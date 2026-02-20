"""HD-CNN 테스트 스크립트

계층적 추론:
- Coarse: Normal vs Defect
- Fine: Normal이면 cell/module_normal 중 선택, Defect이면 세부 분류
"""
import sys
from pathlib import Path

_project_root = Path(__file__).parent.parent.parent.absolute()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import argparse
import json
import csv
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    confusion_matrix, classification_report
)

from models.ct_cnn.model_hdcnn import HDCNN, HDCNNLoss, create_hdcnn_model
from training.configs.config_loader import ConfigLoader
from training.data.dataset_metadata import BatteryMetadataDataset
from training.data.transforms import build_transforms_from_config, get_transforms


class HDCNNTester:
    """HD-CNN 테스터"""

    def __init__(
        self,
        checkpoint_path: str,
        config: dict = None,
        enable_tensorboard: bool = True
    ):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.checkpoint_path = checkpoint_path
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 체크포인트 로드
        print(f"{'='*60}")
        print(f"HD-CNN 체크포인트 로딩: {checkpoint_path}")
        print(f"{'='*60}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        self.config = checkpoint.get('config', config)
        if self.config is None:
            raise ValueError("Config를 찾을 수 없습니다.")

        # 클래스 정보
        self.class_names = self.config.get('classes', {}).get('names',
            ['cell_normal', 'cell_porosity', 'module_normal', 'module_porosity', 'module_resin_overflow'])
        self.num_classes = len(self.class_names)

        # 모델 로드
        self.model = create_hdcnn_model(
            num_fine_classes=self.num_classes,
            pretrained=False,
            dropout=self.config['model'].get('dropout', 0.5)
        ).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # Loss
        self.criterion = HDCNNLoss()

        # DataLoader
        self._create_test_dataloader()

        # 결과 디렉토리
        self.results_dir = Path('models/ct_cnn/results')
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.log_dir = Path('models/ct_cnn/logs')
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # TensorBoard
        self.writer = None
        self.tb_log_dir = None
        if enable_tensorboard:
            checkpoint_name = Path(checkpoint_path).stem
            self.tb_log_dir = self.log_dir / f'test_hdcnn_{checkpoint_name}_{self.timestamp}'
            self.tb_log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=str(self.tb_log_dir))

        print(f"\n모델 로드 완료")
        print(f"  - Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"  - Best F1: {checkpoint.get('best_metric', 'N/A')}")
        print(f"  - Device: {self.device}")
        print(f"  - Test 데이터: {len(self.test_dataset)}개\n")

    def _create_test_dataloader(self):
        """Test DataLoader 생성"""
        config = self.config
        image_size = config['data']['image_size']
        batch_size = config['data']['batch_size']
        num_workers = config['data']['num_workers']
        preprocessed = config['data'].get('preprocessed', False)

        aug_config = config['data'].get('augmentation', None)
        if aug_config:
            test_transform = build_transforms_from_config(
                aug_config.get('val', []), 'ct', image_size, preprocessed
            )
        else:
            test_transform = get_transforms('ct', 'val', image_size, preprocessed)

        test_split = config['data']['test_split']
        if not Path(test_split).is_absolute():
            test_split = str(_project_root / test_split)

        label_base = config['data'].get('label_base', None)
        label_dirs = config['data'].get('label_dirs', None)

        self.test_dataset = BatteryMetadataDataset(
            split_file=test_split,
            modality='ct',
            mode='test',
            transform=test_transform,
            image_size=image_size,
            preprocessed=preprocessed,
            label_base=label_base,
            label_dirs=label_dirs
        )

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

    def test(self) -> dict:
        """테스트 실행"""
        print(f"{'='*60}")
        print(f"HD-CNN 테스트 시작")
        print(f"{'='*60}\n")

        all_labels = []
        all_final_preds = []
        all_coarse_labels = []
        all_coarse_preds = []

        with torch.no_grad():
            for images, metadata, labels in tqdm(self.test_loader, desc="Testing"):
                images = images.to(self.device)
                labels = labels.to(self.device).long()

                # 계층적 예측
                final_preds, coarse_preds, _, _, _ = self.model.predict(images)

                all_labels.extend(labels.cpu().numpy())
                all_final_preds.extend(final_preds.cpu().numpy())

                # Coarse labels
                coarse_labels = self.criterion.get_coarse_labels(labels)
                all_coarse_labels.extend(coarse_labels.cpu().numpy())
                all_coarse_preds.extend(coarse_preds.cpu().numpy())

        # 메트릭 계산
        all_labels = np.array(all_labels)
        all_final_preds = np.array(all_final_preds)
        all_coarse_labels = np.array(all_coarse_labels)
        all_coarse_preds = np.array(all_coarse_preds)

        metrics = self._calculate_metrics(
            all_labels, all_final_preds,
            all_coarse_labels, all_coarse_preds
        )

        # 결과 출력
        self._print_results(metrics, all_labels, all_final_preds)

        # 저장
        self._save_results(metrics, all_labels, all_final_preds)

        if self.writer:
            self._log_to_tensorboard(metrics, all_labels, all_final_preds)
            self.writer.close()

        return metrics

    def _calculate_metrics(
        self,
        labels: np.ndarray,
        preds: np.ndarray,
        coarse_labels: np.ndarray,
        coarse_preds: np.ndarray
    ) -> dict:
        """메트릭 계산"""
        # Fine metrics
        f1_macro = f1_score(labels, preds, average='macro', zero_division=0)
        f1_weighted = f1_score(labels, preds, average='weighted', zero_division=0)
        accuracy = accuracy_score(labels, preds)
        precision_macro = precision_score(labels, preds, average='macro', zero_division=0)
        recall_macro = recall_score(labels, preds, average='macro', zero_division=0)

        f1_per_class = f1_score(labels, preds, average=None, zero_division=0)
        precision_per_class = precision_score(labels, preds, average=None, zero_division=0)
        recall_per_class = recall_score(labels, preds, average=None, zero_division=0)

        cm = confusion_matrix(labels, preds, labels=range(self.num_classes))

        # Coarse metrics
        coarse_accuracy = accuracy_score(coarse_labels, coarse_preds)
        coarse_f1 = f1_score(coarse_labels, coarse_preds, average='binary', zero_division=0)

        class_counts = np.bincount(labels, minlength=self.num_classes)

        return {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_per_class': f1_per_class.tolist(),
            'precision_per_class': precision_per_class.tolist(),
            'recall_per_class': recall_per_class.tolist(),
            'confusion_matrix': cm.tolist(),
            'class_counts': class_counts.tolist(),
            'total_samples': len(labels),
            'coarse_accuracy': coarse_accuracy,
            'coarse_f1': coarse_f1
        }

    def _print_results(self, metrics: dict, labels: np.ndarray, preds: np.ndarray):
        """결과 출력"""
        print(f"\n{'='*60}")
        print(f"HD-CNN 테스트 결과")
        print(f"{'='*60}")
        print(f"  Coarse (Normal vs Defect):")
        print(f"    - Accuracy: {metrics['coarse_accuracy']:.4f}")
        print(f"    - F1: {metrics['coarse_f1']:.4f}")
        print(f"\n  Fine (5 classes):")
        print(f"    - Accuracy: {metrics['accuracy']:.4f}")
        print(f"    - F1 (macro): {metrics['f1_macro']:.4f}")
        print(f"    - F1 (weighted): {metrics['f1_weighted']:.4f}")
        print(f"    - Precision (macro): {metrics['precision_macro']:.4f}")
        print(f"    - Recall (macro): {metrics['recall_macro']:.4f}")

        print(f"\n  클래스별 성능:")
        print("-" * 60)
        report = classification_report(
            labels, preds,
            labels=list(range(len(self.class_names))),
            target_names=self.class_names,
            zero_division=0
        )
        print(report)

    def _save_results(self, metrics: dict, labels: np.ndarray, preds: np.ndarray):
        """결과 저장"""
        checkpoint_name = Path(self.checkpoint_path).stem

        # JSON
        json_path = self.results_dir / f'test_hdcnn_{checkpoint_name}_{self.timestamp}.json'
        json_data = {
            'model_type': 'hdcnn',
            'checkpoint': self.checkpoint_path,
            'timestamp': self.timestamp,
            'num_classes': self.num_classes,
            'class_names': self.class_names,
            'total_samples': metrics['total_samples'],
            'coarse_metrics': {
                'accuracy': metrics['coarse_accuracy'],
                'f1': metrics['coarse_f1']
            },
            'fine_metrics': {
                'accuracy': metrics['accuracy'],
                'f1_macro': metrics['f1_macro'],
                'f1_weighted': metrics['f1_weighted'],
                'precision_macro': metrics['precision_macro'],
                'recall_macro': metrics['recall_macro']
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
            'confusion_matrix': metrics['confusion_matrix']
        }

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        print(f"\n✅ JSON 저장: {json_path}")

        # Confusion Matrix 이미지
        self._save_confusion_matrix(metrics['confusion_matrix'], checkpoint_name)

    def _save_confusion_matrix(self, cm: list, checkpoint_name: str):
        """Confusion Matrix 이미지 저장"""
        import matplotlib.pyplot as plt

        cm = np.array(cm)
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
            title=f'HD-CNN Confusion Matrix - {checkpoint_name}'
        )
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black")

        fig.tight_layout()

        img_path = self.results_dir / f'confusion_matrix_hdcnn_{checkpoint_name}_{self.timestamp}.png'
        plt.savefig(img_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"✅ Confusion Matrix 저장: {img_path}")

    def _log_to_tensorboard(self, metrics: dict, labels: np.ndarray, preds: np.ndarray):
        """TensorBoard 로깅"""
        self.writer.add_scalar('Test/Coarse_Accuracy', metrics['coarse_accuracy'], 0)
        self.writer.add_scalar('Test/Coarse_F1', metrics['coarse_f1'], 0)
        self.writer.add_scalar('Test/Fine_Accuracy', metrics['accuracy'], 0)
        self.writer.add_scalar('Test/Fine_F1_macro', metrics['f1_macro'], 0)

        for i, name in enumerate(self.class_names):
            self.writer.add_scalar(f'Test/F1/{name}', metrics['f1_per_class'][i], 0)

        self.writer.flush()


def main():
    parser = argparse.ArgumentParser(description='HD-CNN Test')
    parser.add_argument('--checkpoint', type=str, required=True, help='체크포인트 경로')
    parser.add_argument('--config', type=str, default=None, help='Config 파일')
    parser.add_argument('--no-tensorboard', action='store_true', help='TensorBoard 비활성화')

    args = parser.parse_args()

    config = None
    if args.config:
        config = ConfigLoader.load(args.config)

    tester = HDCNNTester(
        checkpoint_path=args.checkpoint,
        config=config,
        enable_tensorboard=not args.no_tensorboard
    )
    tester.test()


if __name__ == "__main__":
    main()
