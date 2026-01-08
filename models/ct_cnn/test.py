"""CT CNN 테스트 스크립트"""
import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
_project_root = Path(__file__).parent.parent.parent.absolute()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import torch
import torch.nn as nn
from pathlib import Path
import numpy as np
from tqdm import tqdm
import argparse
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from models.ct_cnn.model import create_model
from training.configs.config_loader import ConfigLoader
from training.data.dataloader import create_test_dataloader
from training.evaluation.metrics import calculate_metrics, print_metrics


class CNNTester:
    """CNN 테스터"""

    def __init__(self, checkpoint_path: str, config: dict = None, enable_tensorboard: bool = True):
        """
        Args:
            checkpoint_path: 체크포인트 파일 경로
            config: YAML config dict (None이면 checkpoint에서 로드)
            enable_tensorboard: TensorBoard 로깅 활성화 여부
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.checkpoint_path = checkpoint_path

        # 체크포인트 로드
        print(f"✅ 체크포인트 로딩: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # Config 로드 (checkpoint 우선, 없으면 외부에서 받음)
        self.config = checkpoint.get('config', config)
        if self.config is None:
            raise ValueError("Config를 찾을 수 없습니다. checkpoint에 config가 없거나 외부 config를 제공하지 않았습니다.")

        # 모델 생성 및 가중치 로드
        self.model = create_model(self.config).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # Threshold 로드
        self.threshold = self.config['criteria'].get('threshold', 0.5)

        # Loss function (다중분류)
        self.criterion = nn.CrossEntropyLoss()

        # Test DataLoader
        self.test_loader = create_test_dataloader(
            test_split_file=self.config['data']['test_split'],
            batch_size=self.config['data']['batch_size'],
            num_workers=self.config['data']['num_workers'],
            image_size=self.config['data']['image_size']
        )

        # TensorBoard Writer 설정
        self.writer = None
        if enable_tensorboard:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            checkpoint_name = Path(checkpoint_path).stem  # 예: resnet18_best_baseline_th03
            log_dir = Path('models/ct_cnn/logs') / f'test_{checkpoint_name}_{timestamp}'
            log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=str(log_dir))
            print(f"✅ TensorBoard 로그 디렉토리: {log_dir}")

        print(f"✅ 모델 로드 완료")
        print(f"   - Epoch: {checkpoint.get('epoch', 'N/A')}")
        val_f1 = checkpoint.get('val_f1_macro', checkpoint.get('val_f1', 'N/A'))
        print(f"   - Val F1: {val_f1:.4f}" if isinstance(val_f1, float) else f"   - Val F1: {val_f1}")
        print(f"   - Threshold: {self.threshold}")
        print(f"   - Device: {self.device}")
        print(f"   - Test 데이터: {len(self.test_loader.dataset)}개\n")

    def test(self) -> dict:
        """Test 데이터 평가"""
        print(f"{'='*60}")
        print(f"Test 데이터 평가 시작")
        print(f"{'='*60}\n")

        total_loss = 0.0
        num_batches = 0

        all_labels = []
        all_preds = []
        all_probs = []
        all_images = []  # 첫 배치만 저장 (Error Samples용)

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(tqdm(self.test_loader, desc="Testing")):
                images = images.to(self.device)
                labels_tensor = labels.to(self.device).long()  # (B,) 다중분류

                # Forward
                outputs = self.model(images)  # (B, num_classes)
                loss = self.criterion(outputs, labels_tensor)

                total_loss += loss.item()
                num_batches += 1

                # 예측 (Softmax + Argmax)
                probs = torch.softmax(outputs, dim=1)  # (B, num_classes)
                preds = outputs.argmax(dim=1)  # (B,)

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_probs.append(probs.cpu().numpy())  # 다중클래스 확률

                # 첫 배치 이미지 저장 (Error Samples 로깅용)
                if batch_idx == 0:
                    all_images = images.cpu()
                    first_batch_labels = labels.cpu().numpy()
                    first_batch_preds = preds.cpu().numpy()
                    first_batch_probs = probs.cpu().numpy()

        avg_loss = total_loss / num_batches
        all_probs = np.vstack(all_probs)  # (N, num_classes)

        # Metrics 계산 (다중분류 5클래스)
        class_names = self.config.get('class_names', ['cell_normal', 'cell_porosity', 'module_normal', 'module_porosity', 'module_resin_overflow'])
        metrics = calculate_metrics(
            y_true=all_labels,
            y_pred=all_preds,
            y_proba=all_probs,
            num_classes=5,
            class_names=class_names
        )

        # 결과 출력
        print(f"\n{'='*60}")
        print(f"Test 결과")
        print(f"{'='*60}")
        print(f"  Test Loss: {avg_loss:.4f}")
        print_metrics(metrics, prefix="  ")
        print(f"{'='*60}\n")

        # TensorBoard 로깅
        if self.writer is not None:
            self._log_to_tensorboard(
                avg_loss,
                metrics,
                np.array(all_labels),
                np.array(all_preds),
                np.array(all_probs),
                all_images if len(all_images) > 0 else None,
                first_batch_labels if len(all_images) > 0 else None,
                first_batch_preds if len(all_images) > 0 else None,
                first_batch_probs if len(all_images) > 0 else None
            )

        return {
            'loss': avg_loss,
            'metrics': metrics,
            'predictions': {
                'labels': np.array(all_labels),
                'preds': np.array(all_preds),
                'probs': np.array(all_probs)
            }
        }

    def _log_to_tensorboard(self, loss, metrics, labels, preds, probs,
                           images=None, batch_labels=None, batch_preds=None, batch_probs=None):
        """TensorBoard에 테스트 결과 로깅"""
        from sklearn.metrics import confusion_matrix, roc_curve
        import matplotlib.pyplot as plt
        import io
        from PIL import Image

        print("📊 TensorBoard 로깅 중...")

        # 1. Scalars - 기본 지표
        self.writer.add_scalar('Test/Loss', loss, 0)
        self.writer.add_scalar('Test/Accuracy', metrics['accuracy'], 0)
        self.writer.add_scalar('Test/F1', metrics['f1'], 0)
        self.writer.add_scalar('Test/Precision', metrics['precision'], 0)
        self.writer.add_scalar('Test/Recall', metrics['recall'], 0)
        if 'roc_auc' in metrics:
            self.writer.add_scalar('Test/ROC-AUC', metrics['roc_auc'], 0)

        # 2. PR Curve
        self.writer.add_pr_curve(
            tag='Test/PR_Curve',
            labels=labels,
            predictions=probs,
            global_step=0
        )

        # 3. Confusion Matrix
        cm = confusion_matrix(labels, preds)
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, cmap='Blues', interpolation='nearest')
        ax.figure.colorbar(im, ax=ax)

        classes = ['Normal', 'Defect']
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=classes,
               yticklabels=classes,
               ylabel='True Label',
               xlabel='Predicted Label',
               title='Confusion Matrix (Test Set)')

        # 숫자 표시
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > cm.max() / 2 else "black")

        fig.tight_layout()

        # Figure를 이미지로 변환
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        buf.seek(0)
        cm_image = Image.open(buf)
        cm_array = np.array(cm_image)
        self.writer.add_image('Test/Confusion_Matrix', cm_array, 0, dataformats='HWC')
        plt.close(fig)

        # 4. ROC Curve
        fpr, tpr, _ = roc_curve(labels, probs)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {metrics.get("roc_auc", 0):.4f})')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve (Test Set)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        buf.seek(0)
        roc_image = Image.open(buf)
        roc_array = np.array(roc_image)
        self.writer.add_image('Test/ROC_Curve', roc_array, 0, dataformats='HWC')
        plt.close(fig)

        # 5. Prediction Histograms (정상/불량 확률 분포)
        normal_probs = probs[labels == 0]
        defect_probs = probs[labels == 1]

        self.writer.add_histogram('Test/Prediction_Prob_Normal', normal_probs, 0)
        self.writer.add_histogram('Test/Prediction_Prob_Defect', defect_probs, 0)

        # 6. Error Samples (첫 배치에서)
        if images is not None and batch_labels is not None:
            self._log_error_samples(images, batch_labels, batch_preds, batch_probs)

        self.writer.flush()
        print("✅ TensorBoard 로깅 완료")

    def _log_error_samples(self, images, labels, preds, probs, max_samples=8):
        """오답 샘플 시각화"""
        import torchvision.utils as vutils

        # False Negative (불량을 정상으로 오판)
        fn_mask = (labels == 1) & (preds == 0)
        fn_indices = np.where(fn_mask)[0]

        if len(fn_indices) > 0:
            fn_images = images[fn_indices[:max_samples]]
            fn_probs = probs[fn_indices[:max_samples]]

            # 확률 텍스트 추가 (간단히 grid만 표시)
            grid = vutils.make_grid(fn_images, nrow=4, normalize=True, scale_each=True)
            self.writer.add_image('Test/Error_FalseNegative', grid, 0)

        # False Positive (정상을 불량으로 오판)
        fp_mask = (labels == 0) & (preds == 1)
        fp_indices = np.where(fp_mask)[0]

        if len(fp_indices) > 0:
            fp_images = images[fp_indices[:max_samples]]
            fp_probs = probs[fp_indices[:max_samples]]

            grid = vutils.make_grid(fp_images, nrow=4, normalize=True, scale_each=True)
            self.writer.add_image('Test/Error_FalsePositive', grid, 0)

    def close(self):
        """TensorBoard Writer 닫기"""
        if self.writer is not None:
            self.writer.close()


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='CT CNN Test')
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='체크포인트 파일 경로 (예: models/ct_cnn/checkpoints/resnet18_best.pt)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Config 파일 이름 (예: cnn). 생략하면 checkpoint에서 로드'
    )

    args = parser.parse_args()

    # Config 로드 (선택사항)
    config = None
    if args.config:
        config = ConfigLoader.load(args.config)

    # Tester 생성 및 평가
    tester = CNNTester(checkpoint_path=args.checkpoint, config=config)
    results = tester.test()

    # 추가 분석 (선택사항)
    print("\n📊 상세 분석:")
    preds = results['predictions']

    # 오답 분석
    fn_indices = np.where((preds['labels'] == 1) & (preds['preds'] == 0))[0]
    fp_indices = np.where((preds['labels'] == 0) & (preds['preds'] == 1))[0]

    print(f"  False Negatives: {len(fn_indices)}개 (불량을 정상으로 오판)")
    print(f"  False Positives: {len(fp_indices)}개 (정상을 불량으로 오판)")

    # 확률 분포 분석
    normal_probs = preds['probs'][preds['labels'] == 0]
    defect_probs = preds['probs'][preds['labels'] == 1]

    print(f"\n  정상 샘플 예측 확률:")
    print(f"    - 평균: {normal_probs.mean():.4f}")
    print(f"    - 표준편차: {normal_probs.std():.4f}")
    print(f"    - 최소/최대: {normal_probs.min():.4f} / {normal_probs.max():.4f}")

    print(f"\n  불량 샘플 예측 확률:")
    print(f"    - 평균: {defect_probs.mean():.4f}")
    print(f"    - 표준편차: {defect_probs.std():.4f}")
    print(f"    - 최소/최대: {defect_probs.min():.4f} / {defect_probs.max():.4f}")

    # TensorBoard Writer 닫기
    tester.close()
    print("\n✅ 평가 완료!")


if __name__ == "__main__":
    main()
