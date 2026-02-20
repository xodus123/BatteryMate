"""VLM 체계적 평가 스크립트 (TensorBoard 포함)

CNN test.py와 동일한 test split으로 VLM(Qwen3-VL, Gemini)을 평가하여
성능 비교가 가능한 동일 포맷의 메트릭과 시각화를 생성합니다.

CT 5클래스:
    0: cell_normal, 1: cell_porosity, 2: module_normal,
    3: module_porosity, 4: module_resin_overflow

RGB 3클래스:
    0: normal, 1: pollution, 2: damaged

사용법:
    # CT 평가 (500샘플, Qwen3-VL 8B)
    python models/vlm/test_vlm_eval.py --config vlm_eval

    # RGB 평가 (500샘플, Qwen3-VL 2B)
    python models/vlm/test_vlm_eval.py --config vlm_eval_rgb --model-size 2b

    # Gemini로 평가
    python models/vlm/test_vlm_eval.py --config vlm_eval --model-type gemini

    # 샘플 수 변경
    python models/vlm/test_vlm_eval.py --config vlm_eval --num-samples 100

    # 전체 평가
    python models/vlm/test_vlm_eval.py --config vlm_eval --full
"""
import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
_project_root = Path(__file__).parent.parent.parent.absolute()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import json
import csv
import re
import time
import argparse
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    confusion_matrix, classification_report, roc_auc_score
)

from models.vlm.prompts import UNIFIED_CLASSES, RGB_CLASSES
from training.configs.config_loader import ConfigLoader

# CT 5클래스 키워드 매핑
CT_KEYWORD_MAP = {
    'cell_normal': ['cell_normal'],
    'cell_porosity': ['cell_porosity'],
    'module_normal': ['module_normal'],
    'module_porosity': ['module_porosity'],
    'module_resin_overflow': ['module_resin_overflow', 'resin_overflow', 'resin overflow'],
}

# RGB 3클래스 키워드 매핑
RGB_KEYWORD_MAP = {
    'normal': ['normal', '정상'],
    'pollution': ['pollution', '오염', 'contamination', 'stain', 'dirty'],
    'damaged': ['damaged', 'mixed', '파손', '손상', 'damage', 'deformation', 'deform'],
}


class VLMEvaluator:
    """VLM 체계적 평가기 (TensorBoard 포함)"""

    def __init__(self, config: dict, model_type: str = None, model_size: str = None):
        """
        Args:
            config: YAML config dict
            model_type: 모델 타입 오버라이드 ('qwen3vl' 또는 'gemini')
            model_size: 모델 크기 오버라이드 ('2b', '4b', '8b', '32b')
        """
        self.config = config
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 모달리티 설정 (클래스 정보보다 먼저 결정)
        data_config = config.get('data', {})
        self.modality = data_config.get('modality', 'ct')

        # 모달리티별 클래스 및 키워드 매핑 설정
        if self.modality == 'rgb':
            default_classes = RGB_CLASSES
            self.KEYWORD_MAP = RGB_KEYWORD_MAP
        else:
            default_classes = UNIFIED_CLASSES
            self.KEYWORD_MAP = CT_KEYWORD_MAP

        # 클래스 정보
        self.class_names = config.get('classes', {}).get('names', default_classes)
        self.num_classes = len(self.class_names)
        self.CLASS_TO_IDX = {name: idx for idx, name in enumerate(self.class_names)}

        # VLM 설정
        vlm_config = config.get('vlm', {})
        self.model_type = model_type or vlm_config.get('model_type', 'qwen3vl')
        self.model_size = model_size or vlm_config.get('model_size', '8b')
        self.use_zero_shot = vlm_config.get('use_zero_shot', True)

        # 데이터 설정
        self.test_split_file = data_config.get('test_split',
            'training/data/splits/ct/resize512/battery_test.txt')

        # 평가 설정
        eval_config = config.get('eval', {})
        self.save_raw_responses = eval_config.get('save_raw_responses', True)

        # 결과 저장 디렉토리
        self.results_dir = Path('models/vlm/results')
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.log_dir = Path(config.get('logging', {}).get('tensorboard', {}).get(
            'log_dir', 'models/vlm/logs'))
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Test split 로드
        self.test_data = self._load_test_split()
        print(f"  Test 데이터: {len(self.test_data)}개")

        # VLM 모델 로드
        self.vlm = self._load_vlm()

        # TensorBoard Writer (evaluate 시 초기화)
        self.writer = None
        self.tb_log_dir = None

    def _load_test_split(self) -> List[Tuple[str, int]]:
        """Test split 파일 로드 (image_path, label 쌍)"""
        split_path = _project_root / self.test_split_file
        if not split_path.exists():
            raise FileNotFoundError(f"Test split 파일을 찾을 수 없습니다: {split_path}")

        data = []
        with open(split_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                if len(parts) == 2:
                    image_path, label = parts[0], int(parts[1])
                    data.append((image_path, label))

        return data

    def _load_vlm(self):
        """VLM 모델 로드"""
        if self.model_type == 'gemini':
            from models.vlm.inference_gemini import GeminiVLMInference
            return GeminiVLMInference()
        else:
            from models.vlm.inference import VLMInference
            return VLMInference(model_size=self.model_size)

    def _sample_stratified(self, num_samples: int, seed: int) -> List[Tuple[str, int]]:
        """클래스별 균등 샘플링 (stratified)"""
        rng = np.random.RandomState(seed)

        # 클래스별 데이터 분리
        class_data = {i: [] for i in range(self.num_classes)}
        for img_path, label in self.test_data:
            class_data[label].append((img_path, label))

        # 클래스별 샘플 수 계산
        samples_per_class = num_samples // self.num_classes
        remainder = num_samples % self.num_classes

        sampled = []
        for cls_idx in range(self.num_classes):
            cls_samples = class_data[cls_idx]
            n = samples_per_class + (1 if cls_idx < remainder else 0)

            if len(cls_samples) <= n:
                sampled.extend(cls_samples)
            else:
                indices = rng.choice(len(cls_samples), size=n, replace=False)
                sampled.extend([cls_samples[i] for i in indices])

        # 셔플
        rng.shuffle(sampled)

        # 클래스별 수 출력
        sampled_counts = np.bincount([s[1] for s in sampled], minlength=self.num_classes)
        print(f"\n  샘플링 결과 ({len(sampled)}장):")
        for i, name in enumerate(self.class_names):
            total = len(class_data[i])
            print(f"    {name}: {sampled_counts[i]}/{total}")

        return sampled

    def evaluate(self, num_samples: int = 500, seed: int = 42, full: bool = False) -> dict:
        """
        VLM 평가 실행

        Args:
            num_samples: 샘플 수 (full=True이면 무시)
            seed: 랜덤 시드
            full: 전체 test split 평가 여부

        Returns:
            평가 결과 딕셔너리
        """
        # 평가 대상 선택
        if full:
            eval_data = self.test_data
            eval_mode = 'full'
            print(f"\n전체 평가 모드: {len(eval_data)}장")
        else:
            eval_data = self._sample_stratified(num_samples, seed)
            eval_mode = f'sampled_{len(eval_data)}'

        # TensorBoard 초기화
        run_name = f'vlm_{self.model_type}_{self.model_size}_{eval_mode}_{self.timestamp}'
        self.tb_log_dir = self.log_dir / run_name
        self.tb_log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.tb_log_dir))
        print(f"  TensorBoard 로그: {self.tb_log_dir}")

        print(f"\n{'='*60}")
        print(f"VLM 평가 시작 ({self.model_type} {self.model_size})")
        print(f"  평가 샘플: {len(eval_data)}장")
        print(f"  모달리티: {self.modality}")
        print(f"{'='*60}\n")

        all_labels = []
        all_preds = []
        all_probs = []
        all_confidences = []
        raw_responses = []
        parse_failures = 0
        start_time = time.time()

        # 실시간 로깅 간격 (샘플 수에 따라 자동 조절)
        total = len(eval_data)
        if total <= 10:
            log_interval = 1
        elif total <= 100:
            log_interval = 10
        else:
            log_interval = max(10, total // 20)

        for step, (img_path, label) in enumerate(tqdm(eval_data, desc="VLM 평가"), 1):
            # VLM 추론
            pred_idx, confidence, pseudo_probs, raw_resp = self._infer_single(img_path)

            all_labels.append(label)
            all_preds.append(pred_idx)
            all_probs.append(pseudo_probs)
            all_confidences.append(confidence)

            if self.save_raw_responses:
                raw_responses.append({
                    'image_path': img_path,
                    'true_label': label,
                    'pred_label': pred_idx,
                    'confidence': confidence,
                    'raw_response': raw_resp,
                })

            if pred_idx == -1:
                parse_failures += 1

            # 실시간 TensorBoard 로깅
            if step % log_interval == 0 or step == total:
                _labels = np.array(all_labels)
                _preds = np.array(all_preds)
                # 파싱 실패(-1)는 임시로 제외하고 계산
                _valid = _preds != -1
                if _valid.sum() > 0:
                    _acc = accuracy_score(_labels[_valid], _preds[_valid])
                    self.writer.add_scalar('Live/Accuracy', _acc, step)
                    self.writer.add_scalar('Live/Confidence_mean', np.mean(all_confidences), step)
                    self.writer.add_scalar('Live/Parse_Failure_Rate',
                                          parse_failures / step, step)
                    self.writer.add_scalar('Live/Samples_Per_Second',
                                          step / (time.time() - start_time), step)
                    # 클래스별 누적 정답률
                    for ci, cn in enumerate(self.class_names):
                        mask = _labels == ci
                        if mask.sum() > 0:
                            cls_acc = (_preds[mask] == ci).sum() / mask.sum()
                            self.writer.add_scalar(f'Live/ClassAcc/{cn}', cls_acc, step)
                    self.writer.flush()

        elapsed = time.time() - start_time

        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)

        # 파싱 실패 처리: -1은 가장 빈번한 클래스로 대체 (최악의 경우)
        if parse_failures > 0:
            print(f"\n  파싱 실패: {parse_failures}/{len(eval_data)} ({parse_failures/len(eval_data)*100:.1f}%)")
            most_common = np.bincount(all_labels, minlength=self.num_classes).argmax()
            fail_mask = all_preds == -1
            all_preds[fail_mask] = most_common
            # 실패한 케이스의 확률은 균등 분포
            all_probs[fail_mask] = 1.0 / self.num_classes

        # 메트릭 계산
        metrics = self._calculate_metrics(all_labels, all_preds, all_probs)
        metrics['parse_failures'] = parse_failures
        metrics['elapsed_seconds'] = elapsed
        metrics['samples_per_second'] = len(eval_data) / elapsed if elapsed > 0 else 0

        # 결과 출력
        self._print_results(metrics, all_labels, all_preds, elapsed)

        # TensorBoard 로깅
        self._log_to_tensorboard(metrics, all_labels, all_preds, all_probs, all_confidences)

        # 결과 파일 저장
        self._save_results(metrics, all_labels, all_preds, all_probs,
                          raw_responses, eval_mode)

        return {
            'metrics': metrics,
            'predictions': {
                'labels': all_labels,
                'preds': all_preds,
                'probs': all_probs,
                'confidences': np.array(all_confidences),
            }
        }

    def _infer_single(self, image_path: str) -> Tuple[int, float, np.ndarray, str]:
        """
        단일 이미지 VLM 추론

        Returns:
            (predicted_class_idx, confidence, pseudo_probabilities, raw_response)
        """
        try:
            if self.model_type == 'gemini':
                result = self.vlm.analyze_image(image_path, modality=self.modality)
                raw_resp = result.get('raw_response', '')
                confidence = result.get('confidence', 50.0)

                # Gemini 응답에서 클래스 매핑
                pred_idx = self._map_gemini_to_label(result, image_path)

            else:
                # Qwen3-VL
                if self.use_zero_shot:
                    result = self.vlm.zero_shot_classify(image_path, modality=self.modality)
                    raw_resp = result.get('raw_response', '')
                    confidence = result.get('confidence', 50.0) or 50.0

                    # zero_shot의 classification 필드에서 매핑
                    classification = result.get('classification', '')
                    pred_idx = self._map_prediction_to_label(classification, image_path)

                    # classification 매핑 실패 시 prediction 필드로 fallback
                    if pred_idx == -1:
                        prediction = result.get('prediction', '')
                        pred_idx = self._map_prediction_to_label(prediction, image_path)
                else:
                    result = self.vlm.analyze_image(
                        image_path, modality=self.modality, detailed=False)
                    raw_resp = result.get('raw_response', '')
                    confidence = result.get('confidence', 50.0) or 50.0

                    prediction = result.get('prediction', '')
                    pred_idx = self._map_prediction_to_label(prediction, image_path)

            # confidence를 float로 보장
            if confidence is None:
                confidence = 50.0
            confidence = float(confidence)

            # pseudo-probability 생성
            pseudo_probs = self._create_pseudo_probabilities(pred_idx, confidence)

            return pred_idx, confidence, pseudo_probs, raw_resp

        except Exception as e:
            # 추론 실패 시 안전한 기본값
            print(f"\n  추론 오류 ({Path(image_path).name}): {e}")
            uniform_probs = np.ones(self.num_classes) / self.num_classes
            return -1, 0.0, uniform_probs, str(e)

    def _map_prediction_to_label(self, prediction_text: str, image_path: str) -> int:
        """
        VLM 텍스트 응답을 클래스 인덱스로 매핑

        Args:
            prediction_text: VLM 응답 텍스트 (classification 또는 prediction 필드)
            image_path: 이미지 경로 (CT: cell/module 구분용)

        Returns:
            클래스 인덱스, 매칭 실패 시 -1
        """
        if not prediction_text:
            return -1

        text = prediction_text.lower().strip()

        # 1단계: 정확 매칭
        for class_name, idx in self.CLASS_TO_IDX.items():
            if class_name == text:
                return idx

        # 2단계: 키워드 포함 매칭
        for class_name, keywords in self.KEYWORD_MAP.items():
            for keyword in keywords:
                if keyword in text:
                    return self.CLASS_TO_IDX[class_name]

        # 3단계: modality별 추가 매핑
        if self.modality == 'rgb':
            return self._map_rgb_fallback(text)
        else:
            return self._map_ct_fallback(text, image_path)

    def _map_rgb_fallback(self, text: str) -> int:
        """RGB 모달리티 추가 매핑 (일반 키워드 → 3클래스)"""
        # 정상 키워드
        if 'clean' in text or 'good' in text or 'no defect' in text:
            return self.CLASS_TO_IDX['normal']

        # 결함 일반 키워드 → damaged (가장 광범위한 결함 클래스)
        if 'defect' in text or '불량' in text or '결함' in text:
            return self.CLASS_TO_IDX['damaged']

        return -1

    def _map_ct_fallback(self, text: str, image_path: str) -> int:
        """CT 모달리티 추가 매핑 (파일명 기반 cell/module 구분)"""
        is_cell = 'CT_cell' in image_path or 'ct_cell' in image_path.lower()
        is_module = 'CT_module' in image_path or 'ct_module' in image_path.lower()

        # 정상 키워드
        if 'normal' in text or '정상' in text:
            if is_cell:
                return self.CLASS_TO_IDX['cell_normal']
            elif is_module:
                return self.CLASS_TO_IDX['module_normal']
            return self.CLASS_TO_IDX.get('cell_normal', 0)

        # porosity 키워드
        if 'porosity' in text or '기공' in text or 'poros' in text:
            if is_cell:
                return self.CLASS_TO_IDX['cell_porosity']
            elif is_module:
                return self.CLASS_TO_IDX['module_porosity']
            return self.CLASS_TO_IDX.get('cell_porosity', 1)

        # resin overflow 키워드
        if 'resin' in text or '레진' in text or 'overflow' in text:
            return self.CLASS_TO_IDX['module_resin_overflow']

        # defect 일반 키워드
        if 'defect' in text or '불량' in text or '결함' in text:
            if is_cell:
                return self.CLASS_TO_IDX['cell_porosity']
            elif is_module:
                return self.CLASS_TO_IDX['module_porosity']
            return -1

        return -1

    def _map_gemini_to_label(self, result: dict, image_path: str) -> int:
        """Gemini 응답을 클래스 인덱스로 매핑"""
        # 먼저 defect_type에서 구체적 결함 확인
        defect_type = result.get('defect_type', '') or ''
        if defect_type:
            mapped = self._map_prediction_to_label(defect_type, image_path)
            if mapped != -1:
                return mapped

        # prediction (normal/defect) + 파일명으로 판단
        prediction = result.get('prediction', '')
        return self._map_prediction_to_label(prediction, image_path)

    def _create_pseudo_probabilities(self, pred_class: int, confidence: float) -> np.ndarray:
        """
        VLM confidence로 pseudo-probability 벡터 생성

        Args:
            pred_class: 예측 클래스 인덱스
            confidence: VLM 신뢰도 (0-100)

        Returns:
            (num_classes,) 확률 벡터 (합 = 1.0)
        """
        probs = np.zeros(self.num_classes)

        if pred_class == -1:
            # 파싱 실패 시 균등 분포
            probs[:] = 1.0 / self.num_classes
            return probs

        # confidence를 0-1 범위로 변환
        conf = np.clip(confidence / 100.0, 0.1, 0.99)

        # 예측 클래스에 confidence, 나머지에 잔여분 균등 분배
        probs[pred_class] = conf
        remaining = (1.0 - conf) / max(self.num_classes - 1, 1)
        for i in range(self.num_classes):
            if i != pred_class:
                probs[i] = remaining

        return probs

    def _calculate_metrics(self, labels: np.ndarray, preds: np.ndarray,
                          probs: np.ndarray) -> dict:
        """메트릭 계산 (CNN test.py와 동일 포맷)"""
        accuracy = accuracy_score(labels, preds)
        f1_macro = f1_score(labels, preds, average='macro', zero_division=0)
        f1_weighted = f1_score(labels, preds, average='weighted', zero_division=0)
        precision_macro = precision_score(labels, preds, average='macro', zero_division=0)
        recall_macro = recall_score(labels, preds, average='macro', zero_division=0)

        all_labels_list = list(range(self.num_classes))
        f1_per_class = f1_score(labels, preds, average=None, labels=all_labels_list, zero_division=0)
        precision_per_class = precision_score(labels, preds, average=None, labels=all_labels_list, zero_division=0)
        recall_per_class = recall_score(labels, preds, average=None, labels=all_labels_list, zero_division=0)

        cm = confusion_matrix(labels, preds, labels=range(self.num_classes))

        try:
            roc_auc_ovr = roc_auc_score(labels, probs, multi_class='ovr', average='macro')
        except Exception:
            roc_auc_ovr = None

        class_counts = np.bincount(labels, minlength=self.num_classes)

        return {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'roc_auc_ovr': roc_auc_ovr,
            'f1_per_class': f1_per_class.tolist(),
            'precision_per_class': precision_per_class.tolist(),
            'recall_per_class': recall_per_class.tolist(),
            'confusion_matrix': cm.tolist(),
            'class_counts': class_counts.tolist(),
            'total_samples': len(labels),
        }

    def _print_results(self, metrics: dict, labels: np.ndarray, preds: np.ndarray,
                      elapsed: float):
        """결과 출력"""
        print(f"\n{'='*60}")
        print(f"VLM 평가 결과 ({self.model_type} {self.model_size})")
        print(f"{'='*60}")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1 (macro): {metrics['f1_macro']:.4f}")
        print(f"  F1 (weighted): {metrics['f1_weighted']:.4f}")
        print(f"  Precision (macro): {metrics['precision_macro']:.4f}")
        print(f"  Recall (macro): {metrics['recall_macro']:.4f}")
        if metrics['roc_auc_ovr'] is not None:
            print(f"  ROC-AUC (OvR macro): {metrics['roc_auc_ovr']:.4f}")
        print(f"  파싱 실패: {metrics['parse_failures']}건")
        print(f"  소요 시간: {elapsed:.1f}초 ({metrics['samples_per_second']:.2f} samples/sec)")

        print(f"\n  클래스별 성능:")
        print("-" * 60)
        report = classification_report(
            labels, preds,
            labels=list(range(self.num_classes)),
            target_names=self.class_names,
            zero_division=0
        )
        print(report)

        print(f"\n  클래스별 샘플 수:")
        for i, (name, count) in enumerate(zip(self.class_names, metrics['class_counts'])):
            print(f"    {i}: {name}: {count}")

        print(f"{'='*60}\n")

    def _log_to_tensorboard(self, metrics: dict, labels: np.ndarray, preds: np.ndarray,
                           probs: np.ndarray, confidences: list):
        """TensorBoard에 결과 로깅"""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import io
        from PIL import Image

        print("TensorBoard 로깅 중...")

        # 1. Scalars - 기본 지표
        self.writer.add_scalar('Test/Accuracy', metrics['accuracy'], 0)
        self.writer.add_scalar('Test/F1_macro', metrics['f1_macro'], 0)
        self.writer.add_scalar('Test/F1_weighted', metrics['f1_weighted'], 0)
        self.writer.add_scalar('Test/Precision_macro', metrics['precision_macro'], 0)
        self.writer.add_scalar('Test/Recall_macro', metrics['recall_macro'], 0)
        if metrics['roc_auc_ovr'] is not None:
            self.writer.add_scalar('Test/ROC_AUC_OvR', metrics['roc_auc_ovr'], 0)
        self.writer.add_scalar('Test/Parse_Failures', metrics['parse_failures'], 0)
        self.writer.add_scalar('Test/Samples_Per_Second', metrics['samples_per_second'], 0)

        # 2. 클래스별 메트릭
        for i, class_name in enumerate(self.class_names):
            self.writer.add_scalar(f'Test/PerClass/F1/{class_name}', metrics['f1_per_class'][i], 0)
            self.writer.add_scalar(f'Test/PerClass/Precision/{class_name}', metrics['precision_per_class'][i], 0)
            self.writer.add_scalar(f'Test/PerClass/Recall/{class_name}', metrics['recall_per_class'][i], 0)

        # 3. PR Curve (클래스별 One-vs-Rest)
        for i, class_name in enumerate(self.class_names):
            binary_labels = (labels == i).astype(int)
            class_probs = probs[:, i]
            self.writer.add_pr_curve(
                f'Test/PR_Curve/{class_name}',
                binary_labels,
                class_probs,
                global_step=0
            )

        # 4. Confusion Matrix
        self._log_confusion_matrix(metrics['confusion_matrix'])

        # 5. Error Summary Table
        self._log_error_summary_table(np.array(metrics['confusion_matrix']))

        # 6. 클래스별 확률 히스토그램
        for i, class_name in enumerate(self.class_names):
            class_probs = probs[:, i]
            self.writer.add_histogram(f'Test/Probabilities/{class_name}/all', class_probs, 0)

            true_mask = labels == i
            if true_mask.sum() > 0:
                self.writer.add_histogram(
                    f'Test/Probabilities/{class_name}/true_samples', class_probs[true_mask], 0)

            false_mask = labels != i
            if false_mask.sum() > 0:
                self.writer.add_histogram(
                    f'Test/Probabilities/{class_name}/false_samples', class_probs[false_mask], 0)

        # 7. 신뢰도 히스토그램 (VLM 고유)
        confidences_arr = np.array(confidences)
        correct_mask = preds == labels

        if correct_mask.sum() > 0:
            self.writer.add_histogram('Test/Confidence/correct', confidences_arr[correct_mask], 0)
        if (~correct_mask).sum() > 0:
            self.writer.add_histogram('Test/Confidence/incorrect', confidences_arr[~correct_mask], 0)
        self.writer.add_histogram('Test/Confidence/all', confidences_arr, 0)

        # 8. 클래스 분포
        self._log_class_distribution(labels)

        # 9. ROC Curve (클래스별 One-vs-Rest)
        self._log_roc_curve(labels, probs)

        # 10. 클래스별 F1/Precision/Recall 바 차트
        self._log_per_class_bar_chart(metrics)

        self.writer.flush()
        print("TensorBoard 로깅 완료")

    def _log_confusion_matrix(self, cm_list: list):
        """Confusion Matrix 이미지 로깅"""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import io
        from PIL import Image

        cm = np.array(cm_list)
        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(cm, cmap='Blues', interpolation='nearest')
        ax.figure.colorbar(im, ax=ax)

        ax.set(
            xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            ylabel='True Label',
            xlabel='Predicted Label',
            title=f'Confusion Matrix - VLM ({self.model_type} {self.model_size})'
        )
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black")

        fig.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        buf.seek(0)
        cm_image = Image.open(buf)
        cm_array = np.array(cm_image)
        self.writer.add_image('Test/Confusion_Matrix', cm_array, 0, dataformats='HWC')
        plt.close(fig)

    def _log_error_summary_table(self, cm: np.ndarray):
        """FP/FN 요약 테이블 이미지"""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import io
        from PIL import Image

        try:
            data = []
            for i, class_name in enumerate(self.class_names):
                tp = cm[i, i]
                fn = cm[i, :].sum() - tp
                fp = cm[:, i].sum() - tp
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                data.append([class_name[:20], int(tp), int(fp), int(fn),
                           f'{precision:.3f}', f'{recall:.3f}', f'{f1:.3f}'])

            fig, ax = plt.subplots(figsize=(14, 5))
            ax.axis('off')

            table = ax.table(
                cellText=data,
                colLabels=['Class', 'TP', 'FP', 'FN', 'Precision', 'Recall', 'F1'],
                cellLoc='center',
                loc='center',
                colWidths=[0.28, 0.1, 0.1, 0.1, 0.12, 0.12, 0.12]
            )
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.8)

            for j in range(7):
                table[(0, j)].set_facecolor('#4472C4')
                table[(0, j)].set_text_props(color='white', weight='bold')

            plt.title(f'VLM ({self.model_type} {self.model_size}) - Error Summary',
                     fontsize=12, fontweight='bold', pad=20)
            plt.tight_layout()

            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            image = Image.open(buf)
            image_array = np.array(image)
            self.writer.add_image('Test/Error_Summary_Table', image_array, 0, dataformats='HWC')
            plt.close(fig)

        except Exception as e:
            print(f"  Error Summary Table 로깅 실패: {e}")

    def _log_class_distribution(self, labels: np.ndarray):
        """클래스 분포 시각화"""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import io
        from PIL import Image

        try:
            class_counts = np.bincount(labels, minlength=self.num_classes)

            fig, ax = plt.subplots(figsize=(12, 6))
            colors = plt.cm.tab10(np.linspace(0, 1, self.num_classes))
            bars = ax.bar(range(self.num_classes), class_counts, color=colors)

            ax.set_xlabel('Class')
            ax.set_ylabel('Count')
            ax.set_title(f'Evaluation Set - Class Distribution ({len(labels)} samples)')
            ax.set_xticks(range(self.num_classes))
            ax.set_xticklabels(self.class_names, rotation=45, ha='right')

            total = class_counts.sum()
            for bar, count in zip(bars, class_counts):
                height = bar.get_height()
                ax.annotate(f'{count:,}\n({count/total*100:.1f}%)',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)

            plt.tight_layout()

            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            image = Image.open(buf)
            image_array = np.array(image)
            self.writer.add_image('Test/Class_Distribution', image_array, 0, dataformats='HWC')
            plt.close(fig)

        except Exception as e:
            print(f"  Class Distribution 로깅 실패: {e}")

    def _log_roc_curve(self, labels: np.ndarray, probs: np.ndarray):
        """ROC Curve (클래스별 One-vs-Rest) 이미지 로깅"""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import io
        from PIL import Image
        from sklearn.metrics import roc_curve, auc

        try:
            fig, ax = plt.subplots(figsize=(10, 8))
            colors = plt.cm.tab10(np.linspace(0, 1, self.num_classes))

            for i, (class_name, color) in enumerate(zip(self.class_names, colors)):
                binary_labels = (labels == i).astype(int)
                if binary_labels.sum() == 0 or binary_labels.sum() == len(binary_labels):
                    continue
                fpr, tpr, _ = roc_curve(binary_labels, probs[:, i])
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, color=color, lw=2,
                       label=f'{class_name} (AUC={roc_auc:.3f})')

            ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(f'ROC Curve - VLM ({self.model_type} {self.model_size})')
            ax.legend(loc='lower right', fontsize=9)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()

            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            image = Image.open(buf)
            image_array = np.array(image)
            self.writer.add_image('Test/ROC_Curve', image_array, 0, dataformats='HWC')
            plt.close(fig)

        except Exception as e:
            print(f"  ROC Curve 로깅 실패: {e}")

    def _log_per_class_bar_chart(self, metrics: dict):
        """클래스별 F1/Precision/Recall 바 차트 이미지 로깅"""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import io
        from PIL import Image

        try:
            x = np.arange(self.num_classes)
            width = 0.25

            fig, ax = plt.subplots(figsize=(14, 7))
            bars_f1 = ax.bar(x - width, metrics['f1_per_class'], width, label='F1', color='#4472C4')
            bars_p = ax.bar(x, metrics['precision_per_class'], width, label='Precision', color='#ED7D31')
            bars_r = ax.bar(x + width, metrics['recall_per_class'], width, label='Recall', color='#70AD47')

            ax.set_xlabel('Class')
            ax.set_ylabel('Score')
            ax.set_title(f'Per-Class Metrics - VLM ({self.model_type} {self.model_size})')
            ax.set_xticks(x)
            ax.set_xticklabels(self.class_names, rotation=45, ha='right')
            ax.set_ylim([0.0, 1.1])
            ax.legend()
            ax.grid(True, axis='y', alpha=0.3)

            # 바 위에 수치 표시
            for bars in [bars_f1, bars_p, bars_r]:
                for bar in bars:
                    height = bar.get_height()
                    if height > 0:
                        ax.annotate(f'{height:.2f}',
                                   xy=(bar.get_x() + bar.get_width() / 2, height),
                                   xytext=(0, 2), textcoords="offset points",
                                   ha='center', va='bottom', fontsize=7)

            fig.tight_layout()

            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            image = Image.open(buf)
            image_array = np.array(image)
            self.writer.add_image('Test/PerClass_Metrics', image_array, 0, dataformats='HWC')
            plt.close(fig)

        except Exception as e:
            print(f"  Per-Class Bar Chart 로깅 실패: {e}")

    def _save_results(self, metrics: dict, labels: np.ndarray, preds: np.ndarray,
                     probs: np.ndarray, raw_responses: list, eval_mode: str):
        """결과를 JSON, CSV, Confusion Matrix PNG로 저장"""
        run_name = f'vlm_{self.model_type}_{self.model_size}_{eval_mode}'

        # 1. JSON 결과 (CNN test.py와 동일 포맷)
        json_path = self.results_dir / f'test_{run_name}_{self.timestamp}.json'

        json_data = {
            'model_type': self.model_type,
            'model_size': self.model_size,
            'eval_mode': eval_mode,
            'timestamp': self.timestamp,
            'num_classes': self.num_classes,
            'class_names': self.class_names,
            'total_samples': metrics['total_samples'],
            'class_counts': dict(zip(self.class_names, metrics['class_counts'])),
            'metrics': {
                'accuracy': metrics['accuracy'],
                'f1_macro': metrics['f1_macro'],
                'f1_weighted': metrics['f1_weighted'],
                'precision_macro': metrics['precision_macro'],
                'recall_macro': metrics['recall_macro'],
                'roc_auc_ovr': metrics['roc_auc_ovr'],
            },
            'per_class_metrics': {
                name: {
                    'f1': metrics['f1_per_class'][i],
                    'precision': metrics['precision_per_class'][i],
                    'recall': metrics['recall_per_class'][i],
                    'support': metrics['class_counts'][i]
                }
                for i, name in enumerate(self.class_names)
            },
            'confusion_matrix': metrics['confusion_matrix'],
            'parse_failures': metrics['parse_failures'],
            'elapsed_seconds': metrics['elapsed_seconds'],
            'samples_per_second': metrics['samples_per_second'],
        }

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        print(f"JSON 결과 저장: {json_path}")

        # 2. CSV 요약
        csv_path = self.log_dir / f'test_{run_name}_{self.timestamp}.csv'

        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['metric', 'value'])
            writer.writerow(['model_type', self.model_type])
            writer.writerow(['model_size', self.model_size])
            writer.writerow(['eval_mode', eval_mode])
            writer.writerow(['timestamp', self.timestamp])
            writer.writerow(['total_samples', metrics['total_samples']])
            writer.writerow(['accuracy', f"{metrics['accuracy']:.4f}"])
            writer.writerow(['f1_macro', f"{metrics['f1_macro']:.4f}"])
            writer.writerow(['f1_weighted', f"{metrics['f1_weighted']:.4f}"])
            writer.writerow(['precision_macro', f"{metrics['precision_macro']:.4f}"])
            writer.writerow(['recall_macro', f"{metrics['recall_macro']:.4f}"])
            if metrics['roc_auc_ovr'] is not None:
                writer.writerow(['roc_auc_ovr', f"{metrics['roc_auc_ovr']:.4f}"])
            writer.writerow(['parse_failures', metrics['parse_failures']])
            writer.writerow(['elapsed_seconds', f"{metrics['elapsed_seconds']:.1f}"])

            writer.writerow([])
            writer.writerow(['--- Per Class Metrics ---', ''])
            for i, name in enumerate(self.class_names):
                writer.writerow([f'{name}_f1', f"{metrics['f1_per_class'][i]:.4f}"])
                writer.writerow([f'{name}_precision', f"{metrics['precision_per_class'][i]:.4f}"])
                writer.writerow([f'{name}_recall', f"{metrics['recall_per_class'][i]:.4f}"])
                writer.writerow([f'{name}_support', metrics['class_counts'][i]])

        print(f"CSV 결과 저장: {csv_path}")

        # 3. Confusion Matrix 이미지 저장
        self._save_confusion_matrix_image(metrics['confusion_matrix'], run_name)

        # 4. Raw responses 저장 (선택)
        if raw_responses:
            raw_path = self.results_dir / f'raw_responses_{run_name}_{self.timestamp}.json'
            with open(raw_path, 'w', encoding='utf-8') as f:
                json.dump(raw_responses, f, indent=1, ensure_ascii=False)
            print(f"Raw responses 저장: {raw_path}")

    def _save_confusion_matrix_image(self, cm_list: list, run_name: str):
        """Confusion Matrix 이미지 저장"""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        cm = np.array(cm_list)
        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(cm, cmap='Blues', interpolation='nearest')
        ax.figure.colorbar(im, ax=ax)

        ax.set(
            xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            ylabel='True Label',
            xlabel='Predicted Label',
            title=f'Confusion Matrix - VLM ({self.model_type} {self.model_size})'
        )
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black")

        fig.tight_layout()

        img_path = self.results_dir / f'confusion_matrix_{run_name}_{self.timestamp}.png'
        plt.savefig(img_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Confusion Matrix 이미지 저장: {img_path}")

    def close(self):
        """TensorBoard Writer 닫기"""
        if self.writer is not None:
            self.writer.close()


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='VLM 체계적 평가 (TensorBoard 포함)')
    parser.add_argument(
        '--config', type=str, default='vlm_eval',
        help='Config 파일 이름 또는 경로 (기본: vlm_eval)'
    )
    parser.add_argument(
        '--model-type', type=str, default=None, choices=['qwen3vl', 'gemini'],
        help='VLM 모델 타입 (config 오버라이드)'
    )
    parser.add_argument(
        '--model-size', type=str, default=None, choices=['2b', '4b', '8b', '32b'],
        help='Qwen3-VL 모델 크기 (config 오버라이드)'
    )
    parser.add_argument(
        '--num-samples', type=int, default=None,
        help='평가 샘플 수 (기본: config의 eval.num_samples)'
    )
    parser.add_argument(
        '--full', action='store_true',
        help='전체 test split 평가 (샘플링 없이)'
    )
    parser.add_argument(
        '--seed', type=int, default=None,
        help='랜덤 시드 (기본: config의 eval.seed)'
    )
    parser.add_argument(
        '--no-tensorboard', action='store_true',
        help='TensorBoard 로깅 비활성화'
    )

    args = parser.parse_args()

    # Config 로드
    config = ConfigLoader.load(args.config)

    # TensorBoard 비활성화 처리
    if args.no_tensorboard:
        config.setdefault('logging', {}).setdefault('tensorboard', {})['enabled'] = False

    # 평가 파라미터
    eval_config = config.get('eval', {})
    num_samples = args.num_samples or eval_config.get('num_samples', 500)
    seed = args.seed or eval_config.get('seed', 42)

    # Evaluator 생성
    evaluator = VLMEvaluator(
        config=config,
        model_type=args.model_type,
        model_size=args.model_size,
    )

    # 평가 실행
    results = evaluator.evaluate(
        num_samples=num_samples,
        seed=seed,
        full=args.full,
    )

    # 추가 분석 출력
    preds_data = results['predictions']
    labels = preds_data['labels']
    preds = preds_data['preds']
    confidences = preds_data['confidences']

    print("\n상세 분석:")

    # 클래스별 오분류 분석
    print("\n  클래스별 오분류:")
    for i, class_name in enumerate(evaluator.class_names):
        fn_count = ((labels == i) & (preds != i)).sum()
        fp_count = ((labels != i) & (preds == i)).sum()
        print(f"    {class_name}: FN={fn_count}, FP={fp_count}")

    # 신뢰도 분석
    correct_mask = preds == labels
    print(f"\n  신뢰도 분석:")
    print(f"    정답 평균 신뢰도: {confidences[correct_mask].mean():.1f}%"
          f" (std: {confidences[correct_mask].std():.1f})")
    if (~correct_mask).sum() > 0:
        print(f"    오답 평균 신뢰도: {confidences[~correct_mask].mean():.1f}%"
              f" (std: {confidences[~correct_mask].std():.1f})")

    # Writer 닫기
    evaluator.close()

    print(f"\n{'='*60}")
    print(f"평가 완료!")
    if evaluator.tb_log_dir:
        print(f"  TensorBoard: tensorboard --logdir={evaluator.tb_log_dir}")
    print(f"  결과 파일: {evaluator.results_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
