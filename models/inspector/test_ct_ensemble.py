"""
CT 앙상블 테스트 스크립트

패치 전략 데이터 + Outline crop 데이터로 앙상블 평가
- CNN+Metadata: 패치 이미지 (512x512)
- AutoEncoder: Outline crop 이미지 (1024x1024)
"""

import sys
from pathlib import Path

_project_root = Path(__file__).parent.parent.parent.absolute()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import argparse
import json
import re
from datetime import datetime
from typing import List, Tuple, Optional, Dict
from tqdm import tqdm
import numpy as np

import torch
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)

from models.inspector.ct_ensemble_inspector import CTEnsembleInspector


CLASS_NAMES = [
    'cell_normal', 'cell_porosity',
    'module_normal', 'module_porosity', 'module_resin_overflow'
]


def extract_original_slice_name(patch_filename: str) -> str:
    """
    패치 파일명에서 원본 슬라이스 이름 추출

    CT_cell_pouch_101_y_033_p00.jpg → CT_cell_pouch_101_y_033.jpg
    CT_module_pouch_044_z_521_p02.jpg → CT_module_pouch_044_z_521.jpg
    """
    # _pXX 패턴 제거
    name = re.sub(r'_p\d+\.jpg$', '.jpg', patch_filename)
    return name


def load_test_data(
    patch_split_file: str,
    outline_base_dir: str,
    max_samples: Optional[int] = None
) -> Tuple[List[str], List[str], List[int]]:
    """
    테스트 데이터 로드 및 매칭

    Args:
        patch_split_file: 패치 split 파일 경로
        outline_base_dir: Outline crop 베이스 디렉토리
        max_samples: 최대 샘플 수 (None이면 전체)

    Returns:
        (patch_paths, outline_paths, labels)
    """
    patch_paths = []
    outline_paths = []
    labels = []
    skipped = 0

    outline_base = Path(outline_base_dir)

    with open(patch_split_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 헤더 스킵
    if lines and lines[0].startswith('#'):
        lines = lines[1:]

    for line in tqdm(lines, desc="Loading data"):
        line = line.strip()
        if not line:
            continue

        parts = line.split('\t')
        if len(parts) < 2:
            continue

        patch_path = parts[0]
        label = int(parts[1])

        # 원본 슬라이스 이름 추출
        patch_filename = Path(patch_path).name
        original_filename = extract_original_slice_name(patch_filename)

        # Training/Validation 폴더 판별
        if 'Training' in patch_path:
            outline_path = outline_base / 'Training' / original_filename
        else:
            outline_path = outline_base / 'Validation' / original_filename

        # Outline 파일 존재 확인
        if not outline_path.exists():
            skipped += 1
            continue

        patch_paths.append(patch_path)
        outline_paths.append(str(outline_path))
        labels.append(label)

        if max_samples and len(labels) >= max_samples:
            break

    print(f"로드 완료: {len(labels)}개 (스킵: {skipped}개)")
    return patch_paths, outline_paths, labels


def evaluate_ensemble(
    inspector: CTEnsembleInspector,
    patch_paths: List[str],
    outline_paths: List[str],
    labels: List[int],
    save_dir: Optional[str] = None
) -> Dict:
    """
    앙상블 평가 실행
    """
    predictions = []
    cnn_predictions = []
    ae_anomalies = []
    agreements = {'both_defect': 0, 'both_normal': 0, 'cnn_only_defect': 0, 'ae_only_anomaly': 0}

    results_detail = []

    print(f"\n앙상블 평가 시작: {len(labels)}개 샘플")

    for patch_path, outline_path, label in tqdm(
        zip(patch_paths, outline_paths, labels),
        total=len(labels),
        desc="Evaluating"
    ):
        try:
            result = inspector.predict(patch_path, outline_path)

            pred_class = result['verdict_class']
            predictions.append(pred_class)
            cnn_predictions.append(result['cnn_result']['class_idx'])
            ae_anomalies.append(result['ae_result']['is_anomaly'])

            agreement = result['details'].get('agreement', 'unknown')
            if agreement in agreements:
                agreements[agreement] += 1

            results_detail.append({
                'patch_path': patch_path,
                'outline_path': outline_path,
                'true_label': label,
                'true_class': CLASS_NAMES[label],
                'pred_label': pred_class,
                'pred_class': result['verdict'],
                'confidence': result['confidence'],
                'cnn_class': result['cnn_result']['class_name'],
                'cnn_confidence': result['cnn_result']['confidence'],
                'cnn_defect_prob': result['cnn_result']['defect_probability'],
                'ae_anomaly_score': result['ae_result']['anomaly_score'],
                'ae_reconstruction_error': result['ae_result']['reconstruction_error'],
                'ae_is_anomaly': result['ae_result']['is_anomaly'],
                'agreement': agreement,
                'correct': pred_class == label
            })

        except Exception as e:
            print(f"Error processing {patch_path}: {e}")
            predictions.append(-1)
            cnn_predictions.append(-1)
            ae_anomalies.append(False)

    # 유효한 예측만 필터링
    valid_mask = [p >= 0 for p in predictions]
    valid_labels = [l for l, v in zip(labels, valid_mask) if v]
    valid_predictions = [p for p, v in zip(predictions, valid_mask) if v]
    valid_cnn_predictions = [p for p, v in zip(cnn_predictions, valid_mask) if v]

    # 메트릭 계산
    ensemble_accuracy = accuracy_score(valid_labels, valid_predictions)
    ensemble_f1_macro = f1_score(valid_labels, valid_predictions, average='macro')
    ensemble_f1_per_class = f1_score(valid_labels, valid_predictions, average=None, labels=range(5))
    ensemble_precision = precision_score(valid_labels, valid_predictions, average='macro')
    ensemble_recall = recall_score(valid_labels, valid_predictions, average='macro')
    ensemble_cm = confusion_matrix(valid_labels, valid_predictions, labels=range(5))

    # CNN만 사용했을 때
    cnn_accuracy = accuracy_score(valid_labels, valid_cnn_predictions)
    cnn_f1_macro = f1_score(valid_labels, valid_cnn_predictions, average='macro')
    cnn_f1_per_class = f1_score(valid_labels, valid_cnn_predictions, average=None, labels=range(5))

    # 결과 출력
    print("\n" + "=" * 60)
    print("앙상블 평가 결과")
    print("=" * 60)

    print(f"\n[전체 메트릭]")
    print(f"  Ensemble Accuracy: {ensemble_accuracy:.4f}")
    print(f"  Ensemble F1 Macro: {ensemble_f1_macro:.4f}")
    print(f"  Ensemble Precision: {ensemble_precision:.4f}")
    print(f"  Ensemble Recall: {ensemble_recall:.4f}")

    print(f"\n[CNN만 사용 시]")
    print(f"  CNN Accuracy: {cnn_accuracy:.4f}")
    print(f"  CNN F1 Macro: {cnn_f1_macro:.4f}")

    print(f"\n[앙상블 효과]")
    print(f"  F1 변화: {cnn_f1_macro:.4f} → {ensemble_f1_macro:.4f} ({(ensemble_f1_macro - cnn_f1_macro)*100:+.2f}%)")

    print(f"\n[클래스별 F1]")
    print(f"  {'Class':<25} {'CNN':>8} {'Ensemble':>10} {'Change':>10}")
    print(f"  {'-'*55}")
    for i, name in enumerate(CLASS_NAMES):
        cnn_f1 = cnn_f1_per_class[i] if i < len(cnn_f1_per_class) else 0
        ens_f1 = ensemble_f1_per_class[i] if i < len(ensemble_f1_per_class) else 0
        change = (ens_f1 - cnn_f1) * 100
        print(f"  {name:<25} {cnn_f1:>8.4f} {ens_f1:>10.4f} {change:>+10.2f}%")

    print(f"\n[Agreement 분포]")
    total = sum(agreements.values())
    for key, count in agreements.items():
        pct = count / total * 100 if total > 0 else 0
        print(f"  {key}: {count} ({pct:.1f}%)")

    print(f"\n[Confusion Matrix]")
    print(ensemble_cm)

    # 결과 저장
    metrics = {
        'timestamp': datetime.now().isoformat(),
        'num_samples': len(valid_labels),
        'ensemble': {
            'accuracy': ensemble_accuracy,
            'f1_macro': ensemble_f1_macro,
            'precision_macro': ensemble_precision,
            'recall_macro': ensemble_recall,
            'f1_per_class': {CLASS_NAMES[i]: float(ensemble_f1_per_class[i]) for i in range(5)},
            'confusion_matrix': ensemble_cm.tolist()
        },
        'cnn_only': {
            'accuracy': cnn_accuracy,
            'f1_macro': cnn_f1_macro,
            'f1_per_class': {CLASS_NAMES[i]: float(cnn_f1_per_class[i]) for i in range(5)}
        },
        'improvement': {
            'f1_macro': ensemble_f1_macro - cnn_f1_macro,
            'accuracy': ensemble_accuracy - cnn_accuracy
        },
        'agreements': agreements
    }

    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 메트릭 JSON 저장
        metrics_file = save_path / f"ensemble_test_{timestamp}.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\n메트릭 저장: {metrics_file}")

        # 상세 결과 CSV 저장
        import csv
        csv_file = save_path / f"ensemble_test_detail_{timestamp}.csv"
        if results_detail:
            with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=results_detail[0].keys())
                writer.writeheader()
                writer.writerows(results_detail)
            print(f"상세 결과 저장: {csv_file}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description='CT 앙상블 테스트')
    parser.add_argument('--patch-split', type=str,
                        default='training/data/splits/ct/patch/battery_test.txt',
                        help='패치 테스트 split 파일')
    parser.add_argument('--outline-dir', type=str,
                        default='/mnt/d/battery-cropped-v2',
                        help='Outline crop 디렉토리')
    parser.add_argument('--cnn-checkpoint', type=str,
                        default='models/ct_cnn/checkpoints/metadata_best_20260129_232820.pt',
                        help='CNN+Metadata 체크포인트')
    parser.add_argument('--ae-checkpoint', type=str,
                        default='models/ct_ae/checkpoints/ct_ae_best_20260127_190546.pt',
                        help='AutoEncoder 체크포인트')
    parser.add_argument('--ae-threshold', type=str,
                        default='models/ct_ae/checkpoints/threshold.json',
                        help='AE threshold 파일')
    parser.add_argument('--cnn-weight', type=float, default=0.7,
                        help='CNN 앙상블 가중치')
    parser.add_argument('--ae-weight', type=float, default=0.3,
                        help='AE 앙상블 가중치')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='최대 샘플 수 (빠른 테스트용)')
    parser.add_argument('--save-dir', type=str,
                        default='models/inspector/results',
                        help='결과 저장 디렉토리')

    args = parser.parse_args()

    print("=" * 60)
    print("CT 앙상블 테스트")
    print("=" * 60)
    print(f"패치 split: {args.patch_split}")
    print(f"Outline 디렉토리: {args.outline_dir}")
    print(f"CNN 체크포인트: {args.cnn_checkpoint}")
    print(f"AE 체크포인트: {args.ae_checkpoint}")
    print(f"앙상블 가중치: CNN={args.cnn_weight}, AE={args.ae_weight}")
    if args.max_samples:
        print(f"최대 샘플: {args.max_samples}")
    print("=" * 60)

    # 데이터 로드
    patch_paths, outline_paths, labels = load_test_data(
        args.patch_split,
        args.outline_dir,
        args.max_samples
    )

    if len(labels) == 0:
        print("테스트할 데이터가 없습니다.")
        return

    # 앙상블 검사기 생성
    inspector = CTEnsembleInspector(
        cnn_checkpoint=args.cnn_checkpoint,
        ae_checkpoint=args.ae_checkpoint,
        ae_threshold_path=args.ae_threshold,
        cnn_config='cnn_ct_metadata',
        ae_config='autoencoder_ct',
        ensemble_weights=(args.cnn_weight, args.ae_weight)
    )

    # 평가 실행
    metrics = evaluate_ensemble(
        inspector,
        patch_paths,
        outline_paths,
        labels,
        save_dir=args.save_dir
    )

    print("\n완료!")


if __name__ == "__main__":
    main()
