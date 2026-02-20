"""CT AutoEncoder 테스트 스크립트

학습된 CT AE 모델로 테스트 셋 평가
- Anomaly Detection 성능 측정 (ROC-AUC, Precision, Recall)
- Threshold 기반 정상/결함 분류
"""
import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
_project_root = Path(__file__).parent.parent.parent.absolute()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
import numpy as np
from tqdm import tqdm
import argparse
from datetime import datetime
import json

from models.rgb_ae.model import ConvAutoEncoder, create_model
from training.configs.config_loader import ConfigLoader
from training.data.dataset import BatteryDataset
from training.data.transforms import get_transforms, build_transforms_from_config
from sklearn.metrics import (
    roc_curve, roc_auc_score, precision_recall_curve,
    confusion_matrix, classification_report,
    precision_score, recall_score, f1_score
)


class CTAETester:
    """CT AutoEncoder 테스트"""

    def __init__(self, config: dict, checkpoint_path: str):
        self.config = config
        self.device = torch.device(config['training'].get('device', 'cuda'))
        self.normal_classes = config['classes'].get('normal_classes', [0, 2])
        self.class_names = config['classes'].get('names', [])

        # 모델 로드
        self.model = self._load_model(checkpoint_path)
        self.model.eval()

        # Threshold 로드
        self.threshold = self._load_threshold(checkpoint_path)

        # 테스트 데이터 로더
        self.test_loader = self._create_test_loader()

        print(f"CT AE Tester 초기화 완료")
        print(f"   - Device: {self.device}")
        print(f"   - Normal classes: {self.normal_classes}")
        print(f"   - Threshold: {self.threshold:.4f}")
        print(f"   - Test samples: {len(self.test_loader.dataset)}")

    def _load_model(self, checkpoint_path: str) -> nn.Module:
        """모델 로드"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        model_config = self.config['model']
        encoder_config = model_config.get('encoder', {})
        decoder_config = model_config.get('decoder', {})

        model = ConvAutoEncoder(
            image_size=self.config['data']['image_size'],
            latent_dim=model_config.get('latent_dim', 512),
            encoder_channels=encoder_config.get('channels', [3, 64, 128, 256, 512]),
            decoder_channels=decoder_config.get('channels', [512, 256, 128, 64, 3]),
            dropout=model_config.get('dropout', 0.2)
        )

        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        print(f"   - Model loaded from: {checkpoint_path}")

        return model

    def _load_threshold(self, checkpoint_path: str) -> float:
        """Threshold 로드"""
        # 체크포인트에서 먼저 시도
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if 'threshold' in checkpoint and checkpoint['threshold'] is not None:
            return checkpoint['threshold']

        # threshold.json 파일에서 로드
        checkpoint_dir = Path(checkpoint_path).parent
        threshold_path = checkpoint_dir / 'threshold.json'
        if threshold_path.exists():
            with open(threshold_path, 'r') as f:
                data = json.load(f)
                return data.get('threshold', 0.0)

        print("   - Warning: Threshold not found, using default 0.0")
        return 0.0

    def _create_test_loader(self) -> DataLoader:
        """테스트 데이터 로더 생성"""
        data_config = self.config['data']
        image_size = data_config['image_size']
        preprocessed = data_config.get('preprocessed', True)

        # Val transform 사용 (augmentation 없음)
        augmentation_config = data_config.get('augmentation', None)
        if augmentation_config is not None:
            val_aug = augmentation_config.get('val', [])
            transform = build_transforms_from_config(val_aug, 'ct', image_size, preprocessed)
        else:
            transform = get_transforms('ct', 'val', image_size, preprocessed)

        test_dataset = BatteryDataset(
            split_file=data_config['test_split'],
            transform=transform,
            modality='ct',
            preprocessed=preprocessed
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=data_config['batch_size'],
            shuffle=False,
            num_workers=data_config.get('num_workers', 4),
            pin_memory=True
        )

        return test_loader

    @torch.no_grad()
    def test(self, save_results: bool = True) -> dict:
        """테스트 실행"""
        self.model.eval()
        all_scores = []
        all_labels = []
        all_preds = []

        print(f"\n{'='*60}")
        print("CT AutoEncoder 테스트")
        print(f"{'='*60}")

        for images, labels in tqdm(self.test_loader, desc="Testing"):
            images = images.to(self.device)
            scores = self.model.get_anomaly_score(images)

            # Threshold 기반 예측 (정상: 0, 결함: 1)
            preds = (scores > self.threshold).long()

            all_scores.extend(scores.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_preds.extend(preds.cpu().numpy())

        all_scores = np.array(all_scores)
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)

        # 이진 레이블 (정상: 0, 결함: 1)
        binary_labels = (~np.isin(all_labels, self.normal_classes)).astype(int)

        # 메트릭 계산
        results = self._compute_metrics(all_scores, all_labels, all_preds, binary_labels)

        # 결과 출력
        self._print_results(results)

        # 결과 저장
        if save_results:
            self._save_results(results, all_scores, all_labels, all_preds)

        return results

    def _compute_metrics(self, scores, labels, preds, binary_labels) -> dict:
        """메트릭 계산"""
        # 정상 vs 결함 점수 분포
        normal_mask = np.isin(labels, self.normal_classes)
        normal_scores = scores[normal_mask]
        defect_scores = scores[~normal_mask]

        results = {
            'threshold': self.threshold,
            'normal_classes': self.normal_classes,
            'total_samples': len(labels),
            'normal_samples': int(normal_mask.sum()),
            'defect_samples': int((~normal_mask).sum()),
            'score_stats': {
                'normal_mean': float(normal_scores.mean()) if len(normal_scores) > 0 else 0,
                'normal_std': float(normal_scores.std()) if len(normal_scores) > 0 else 0,
                'defect_mean': float(defect_scores.mean()) if len(defect_scores) > 0 else 0,
                'defect_std': float(defect_scores.std()) if len(defect_scores) > 0 else 0,
            }
        }

        # Binary classification metrics
        if len(defect_scores) > 0:
            # ROC-AUC
            try:
                results['roc_auc'] = float(roc_auc_score(binary_labels, scores))
            except Exception:
                results['roc_auc'] = 0.0

            # Precision, Recall, F1
            results['precision'] = float(precision_score(binary_labels, preds, zero_division=0))
            results['recall'] = float(recall_score(binary_labels, preds, zero_division=0))
            results['f1'] = float(f1_score(binary_labels, preds, zero_division=0))

            # Confusion Matrix (binary)
            cm = confusion_matrix(binary_labels, preds)
            results['confusion_matrix_binary'] = cm.tolist()

            # Per-class breakdown
            tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
            results['true_negative'] = int(tn)
            results['false_positive'] = int(fp)
            results['false_negative'] = int(fn)
            results['true_positive'] = int(tp)

        # 클래스별 분포
        class_stats = {}
        for cls_idx, cls_name in enumerate(self.class_names):
            cls_mask = labels == cls_idx
            if cls_mask.sum() > 0:
                cls_scores = scores[cls_mask]
                cls_preds = preds[cls_mask]
                is_normal = cls_idx in self.normal_classes

                class_stats[cls_name] = {
                    'count': int(cls_mask.sum()),
                    'is_normal': is_normal,
                    'score_mean': float(cls_scores.mean()),
                    'score_std': float(cls_scores.std()),
                    'predicted_normal': int((cls_preds == 0).sum()),
                    'predicted_defect': int((cls_preds == 1).sum()),
                    'accuracy': float((cls_preds == (0 if is_normal else 1)).mean())
                }
        results['class_stats'] = class_stats

        return results

    def _print_results(self, results: dict):
        """결과 출력"""
        print(f"\n{'='*60}")
        print("테스트 결과")
        print(f"{'='*60}")

        print(f"\n[전체 통계]")
        print(f"  Total: {results['total_samples']}")
        print(f"  Normal: {results['normal_samples']}")
        print(f"  Defect: {results['defect_samples']}")
        print(f"  Threshold: {results['threshold']:.4f}")

        print(f"\n[점수 분포]")
        stats = results['score_stats']
        print(f"  Normal: {stats['normal_mean']:.4f} +/- {stats['normal_std']:.4f}")
        print(f"  Defect: {stats['defect_mean']:.4f} +/- {stats['defect_std']:.4f}")

        if 'roc_auc' in results:
            print(f"\n[Binary Classification 성능]")
            print(f"  ROC-AUC:   {results['roc_auc']:.4f}")
            print(f"  Precision: {results['precision']:.4f}")
            print(f"  Recall:    {results['recall']:.4f}")
            print(f"  F1-Score:  {results['f1']:.4f}")

            print(f"\n[Confusion Matrix (Binary)]")
            print(f"             Pred Normal  Pred Defect")
            print(f"  Normal:    {results['true_negative']:>10}  {results['false_positive']:>10}")
            print(f"  Defect:    {results['false_negative']:>10}  {results['true_positive']:>10}")

        print(f"\n[클래스별 성능]")
        for cls_name, stats in results.get('class_stats', {}).items():
            marker = "(Normal)" if stats['is_normal'] else "(Defect)"
            print(f"  {cls_name} {marker}:")
            print(f"    Count: {stats['count']}, Score: {stats['score_mean']:.4f} +/- {stats['score_std']:.4f}")
            print(f"    Accuracy: {stats['accuracy']:.4f}")

        print(f"\n{'='*60}")

    def _save_results(self, results: dict, scores, labels, preds):
        """결과 저장"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_dir = Path('models/ct_ae/results')
        results_dir.mkdir(parents=True, exist_ok=True)

        # JSON 결과 저장
        results_path = results_dir / f'test_ct_ae_{timestamp}.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"  Results saved to: {results_path}")

        # 예측 결과 CSV 저장
        csv_path = results_dir / f'predictions_ct_ae_{timestamp}.csv'
        with open(csv_path, 'w') as f:
            f.write('score,label,pred_binary,label_binary,class_name\n')
            for i in range(len(scores)):
                label_binary = 0 if labels[i] in self.normal_classes else 1
                cls_name = self.class_names[labels[i]] if labels[i] < len(self.class_names) else f'class_{labels[i]}'
                f.write(f'{scores[i]:.6f},{labels[i]},{preds[i]},{label_binary},{cls_name}\n')
        print(f"  Predictions saved to: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description='CT AutoEncoder Test')
    parser.add_argument('--config', type=str, default='autoencoder_ct',
                       help='Config 파일 이름')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='체크포인트 경로')
    args = parser.parse_args()

    # Config 로드
    config_loader = ConfigLoader()
    config = config_loader.load(args.config)

    # Tester 생성 및 테스트
    tester = CTAETester(config, args.checkpoint)
    tester.test()


if __name__ == "__main__":
    main()
