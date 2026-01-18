"""RGB AutoEncoder 테스트 스크립트 (TensorBoard 로깅 포함)"""
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
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import numpy as np
from tqdm import tqdm
import argparse
from datetime import datetime
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, f1_score, accuracy_score, confusion_matrix, precision_score, recall_score

from models.rgb_ae.model import create_model, ConvAutoEncoder
from training.configs.config_loader import ConfigLoader
from training.data.dataset import BatteryDataset
from training.data.transforms import get_transforms, get_albumentations_transforms


class AETester:
    """AutoEncoder 테스터"""

    def __init__(self, checkpoint_path: str, config: dict = None, log_dir: str = None):
        """
        Args:
            checkpoint_path: 체크포인트 파일 경로
            config: YAML config dict (None이면 checkpoint에서 로드)
            log_dir: TensorBoard 로그 디렉토리
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.checkpoint_path = checkpoint_path

        # 체크포인트 로드
        print(f"✅ 체크포인트 로딩: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # Config 로드
        self.config = checkpoint.get('config', config)
        if self.config is None:
            raise ValueError("Config를 찾을 수 없습니다.")

        # 모델 생성 및 가중치 로드
        self.model = create_model(self.config).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # Threshold 로드
        self.threshold = checkpoint.get('threshold', None)
        if self.threshold is None:
            # threshold.json에서 로드 시도
            threshold_path = Path(self.config['checkpoint']['save_dir']) / 'threshold.json'
            if threshold_path.exists():
                with open(threshold_path, 'r') as f:
                    threshold_data = json.load(f)
                    self.threshold = threshold_data.get('threshold', 0.1)
            else:
                self.threshold = 0.1  # 기본값

        # 테스트 데이터 로더
        self.test_loader = self._create_test_dataloader()

        # TensorBoard Writer
        if log_dir is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_dir = f"models/rgb_ae/logs/test_{timestamp}"
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir=log_dir)

        print(f"✅ 모델 로드 완료")
        print(f"   - Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"   - Val Loss: {checkpoint.get('val_loss', 'N/A'):.4f}")
        print(f"   - Threshold: {self.threshold:.4f}")
        print(f"   - Device: {self.device}")
        print(f"   - Test 데이터: {len(self.test_loader.dataset)}개")
        print(f"   - TensorBoard: {log_dir}\n")

    def _create_test_dataloader(self):
        """테스트 데이터 로더 생성"""
        data_config = self.config['data']
        image_size = data_config['image_size']
        preprocessed = data_config.get('preprocessed', False)
        use_albumentations = data_config.get('use_albumentations', False)

        # Transform 선택
        if use_albumentations:
            transform = get_albumentations_transforms('rgb', 'test', image_size, preprocessed)
        else:
            transform = get_transforms('rgb', 'test', image_size, preprocessed)

        test_dataset = BatteryDataset(
            split_file=data_config['test_split'],
            transform=transform,
            modality='rgb',
            preprocessed=preprocessed
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=data_config['batch_size'],
            shuffle=False,
            num_workers=data_config.get('num_workers', 8),
            pin_memory=True
        )

        return test_loader

    @torch.no_grad()
    def test(self) -> dict:
        """테스트 데이터 평가"""
        print(f"{'='*60}")
        print(f"Test 데이터 평가 시작")
        print(f"{'='*60}\n")

        all_scores = []
        all_labels = []
        all_preds = []

        for images, labels in tqdm(self.test_loader, desc="Testing"):
            images = images.to(self.device)

            # 이상 점수 계산
            scores = self.model.get_anomaly_score(images)

            all_scores.extend(scores.cpu().numpy())
            all_labels.extend(labels.numpy())

        all_scores = np.array(all_scores)
        all_labels = np.array(all_labels)

        # 이진 분류로 변환 (0: normal, 1+: defect)
        binary_labels = (all_labels > 0).astype(int)

        # Threshold 기반 예측
        # 실제 결과: Defect(불량)이 높은 점수, Normal(정상)이 낮은 점수
        # (불량 데이터의 변동성이 커서 모델이 평균 패턴 학습 → 정상이 더 잘 재구성됨)
        # 따라서: 점수 > threshold → defect(1), 점수 <= threshold → normal(0)
        all_preds = (all_scores > self.threshold).astype(int)  # 높은 점수 = defect

        # 메트릭 계산
        metrics = self._calculate_metrics(all_scores, all_labels, binary_labels, all_preds)

        # 결과 출력
        self._print_results(metrics)

        # TensorBoard 로깅
        self._log_to_tensorboard(metrics, all_scores, all_labels, binary_labels, all_preds)

        return {
            'metrics': metrics,
            'scores': all_scores,
            'labels': all_labels,
            'predictions': all_preds
        }

    def _calculate_metrics(self, scores, labels, binary_labels, preds) -> dict:
        """메트릭 계산"""
        metrics = {}

        # 기본 통계
        metrics['num_samples'] = len(labels)
        metrics['num_normal'] = (labels == 0).sum()
        metrics['num_defect'] = (labels > 0).sum()

        # 점수 통계
        normal_scores = scores[labels == 0]
        defect_scores = scores[labels > 0]

        metrics['score_mean'] = scores.mean()
        metrics['score_std'] = scores.std()

        if len(normal_scores) > 0:
            metrics['normal_score_mean'] = normal_scores.mean()
            metrics['normal_score_std'] = normal_scores.std()
        if len(defect_scores) > 0:
            metrics['defect_score_mean'] = defect_scores.mean()
            metrics['defect_score_std'] = defect_scores.std()

        # ROC-AUC (defect를 positive로, 높은 점수 = defect)
        try:
            metrics['roc_auc'] = roc_auc_score(binary_labels, scores)
        except:
            metrics['roc_auc'] = 0.0

        # Accuracy, F1 (threshold 기반)
        metrics['accuracy'] = accuracy_score(binary_labels, preds)
        metrics['f1'] = f1_score(binary_labels, preds, zero_division=0)

        # Confusion Matrix
        cm = confusion_matrix(binary_labels, preds)
        metrics['confusion_matrix'] = cm.tolist()

        # 최적 threshold 찾기
        fpr, tpr, thresholds = roc_curve(binary_labels, scores)
        optimal_idx = np.argmax(tpr - fpr)
        metrics['optimal_threshold'] = thresholds[optimal_idx]

        return metrics

    def _print_results(self, metrics: dict):
        """결과 출력"""
        print(f"\n{'='*60}")
        print(f"Test 결과")
        print(f"{'='*60}")
        print(f"  샘플 수: {metrics['num_samples']}")
        print(f"    - Normal: {metrics['num_normal']}")
        print(f"    - Defect: {metrics['num_defect']}")
        print()
        print(f"  점수 통계:")
        if 'normal_score_mean' in metrics:
            print(f"    - Normal: {metrics['normal_score_mean']:.4f} ± {metrics['normal_score_std']:.4f}")
        if 'defect_score_mean' in metrics:
            print(f"    - Defect: {metrics['defect_score_mean']:.4f} ± {metrics['defect_score_std']:.4f}")
        print()
        print(f"  성능 지표:")
        print(f"    - ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"    - Accuracy: {metrics['accuracy']:.4f}")
        print(f"    - F1 Score: {metrics['f1']:.4f}")
        print(f"    - Threshold: {self.threshold:.4f}")
        print(f"    - Optimal Threshold: {metrics['optimal_threshold']:.4f}")
        print(f"{'='*60}\n")

    def _log_to_tensorboard(self, metrics: dict, scores: np.ndarray, labels: np.ndarray,
                            binary_labels: np.ndarray, preds: np.ndarray):
        """TensorBoard에 결과 로깅"""
        # 1. 스칼라 메트릭 로깅
        self.writer.add_scalar('Test/ROC_AUC', metrics['roc_auc'], 0)
        self.writer.add_scalar('Test/Accuracy', metrics['accuracy'], 0)
        self.writer.add_scalar('Test/F1_Score', metrics['f1'], 0)
        self.writer.add_scalar('Test/Threshold', self.threshold, 0)
        self.writer.add_scalar('Test/Optimal_Threshold', metrics['optimal_threshold'], 0)

        if 'normal_score_mean' in metrics:
            self.writer.add_scalar('Test/Normal_Score_Mean', metrics['normal_score_mean'], 0)
            self.writer.add_scalar('Test/Normal_Score_Std', metrics['normal_score_std'], 0)
        if 'defect_score_mean' in metrics:
            self.writer.add_scalar('Test/Defect_Score_Mean', metrics['defect_score_mean'], 0)
            self.writer.add_scalar('Test/Defect_Score_Std', metrics['defect_score_std'], 0)

        # 2. Score Distribution 히스토그램
        normal_scores = scores[labels == 0]
        defect_scores = scores[labels > 0]

        if len(normal_scores) > 0:
            self.writer.add_histogram('Test/Normal_Scores', normal_scores, 0)
        if len(defect_scores) > 0:
            self.writer.add_histogram('Test/Defect_Scores', defect_scores, 0)

        # 3. Confusion Matrix Figure
        cm = confusion_matrix(binary_labels, preds)
        fig_cm = plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Normal', 'Defect'],
                    yticklabels=['Normal', 'Defect'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix (Threshold={self.threshold:.4f})')
        self.writer.add_figure('Test/Confusion_Matrix', fig_cm, 0)
        plt.close(fig_cm)

        # 4. ROC Curve Figure
        fpr, tpr, _ = roc_curve(binary_labels, scores)
        fig_roc = plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC={metrics["roc_auc"]:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        self.writer.add_figure('Test/ROC_Curve', fig_roc, 0)
        plt.close(fig_roc)

        # 5. Score Distribution Figure
        fig_dist = plt.figure(figsize=(10, 5))
        plt.hist(normal_scores, bins=50, alpha=0.7, label=f'Normal (n={len(normal_scores)})', color='green')
        plt.hist(defect_scores, bins=50, alpha=0.7, label=f'Defect (n={len(defect_scores)})', color='red')
        plt.axvline(self.threshold, color='black', linestyle='--', linewidth=2, label=f'Threshold={self.threshold:.4f}')
        plt.xlabel('Anomaly Score')
        plt.ylabel('Count')
        plt.title('Score Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        self.writer.add_figure('Test/Score_Distribution', fig_dist, 0)
        plt.close(fig_dist)

        # 6. Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(binary_labels, scores)
        fig_pr = plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, 'b-', linewidth=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.grid(True, alpha=0.3)
        self.writer.add_figure('Test/PR_Curve', fig_pr, 0)
        plt.close(fig_pr)

        # 7. 샘플 재구성 결과 이미지 로깅
        self._log_reconstructions_to_tensorboard()

        self.writer.flush()
        print(f"✅ TensorBoard 로깅 완료: {self.log_dir}")

    def _log_reconstructions_to_tensorboard(self, num_samples: int = 8):
        """재구성 결과를 TensorBoard에 로깅"""
        dataiter = iter(self.test_loader)
        images, labels = next(dataiter)
        images = images[:num_samples].to(self.device)

        with torch.no_grad():
            reconstructed, _ = self.model(images)

        # Denormalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)

        images_denorm = torch.clamp(images * std + mean, 0, 1)
        reconstructed_denorm = torch.clamp(reconstructed * std + mean, 0, 1)

        # 원본, 재구성, 차이를 결합
        diff = torch.abs(images_denorm - reconstructed_denorm)

        # Grid 이미지 생성
        import torchvision.utils as vutils

        # 원본 이미지 그리드
        self.writer.add_images('Test/Original', images_denorm.cpu(), 0)
        # 재구성 이미지 그리드
        self.writer.add_images('Test/Reconstructed', reconstructed_denorm.cpu(), 0)
        # 차이 이미지 그리드
        self.writer.add_images('Test/Difference', diff.cpu(), 0)

    def visualize_reconstructions(self, num_samples: int = 8, save_path: str = None):
        """재구성 결과 시각화"""
        self.model.eval()

        # 샘플 가져오기
        dataiter = iter(self.test_loader)
        images, labels = next(dataiter)
        images = images[:num_samples].to(self.device)
        labels = labels[:num_samples]

        with torch.no_grad():
            reconstructed, _ = self.model(images)

        # Denormalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)

        images_denorm = images * std + mean
        reconstructed_denorm = reconstructed * std + mean

        # 클램핑
        images_denorm = torch.clamp(images_denorm, 0, 1)
        reconstructed_denorm = torch.clamp(reconstructed_denorm, 0, 1)

        # 플롯
        fig, axes = plt.subplots(3, num_samples, figsize=(2*num_samples, 6))

        class_names = ['normal', 'pollution', 'mixed']

        for i in range(num_samples):
            # 원본
            axes[0, i].imshow(images_denorm[i].cpu().permute(1, 2, 0).numpy())
            axes[0, i].set_title(f'{class_names[labels[i]]}')
            axes[0, i].axis('off')

            # 재구성
            axes[1, i].imshow(reconstructed_denorm[i].cpu().permute(1, 2, 0).numpy())
            axes[1, i].set_title('Reconstructed')
            axes[1, i].axis('off')

            # 차이
            diff = torch.abs(images_denorm[i] - reconstructed_denorm[i]).mean(dim=0)
            axes[2, i].imshow(diff.cpu().numpy(), cmap='hot')
            axes[2, i].set_title(f'Error: {diff.mean():.3f}')
            axes[2, i].axis('off')

        axes[0, 0].set_ylabel('Original', fontsize=12)
        axes[1, 0].set_ylabel('Reconstructed', fontsize=12)
        axes[2, 0].set_ylabel('Error Map', fontsize=12)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ 시각화 저장: {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_score_distribution(self, scores: np.ndarray, labels: np.ndarray, save_path: str = None):
        """이상 점수 분포 시각화"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # 히스토그램
        normal_scores = scores[labels == 0]
        defect_scores = scores[labels > 0]

        axes[0].hist(normal_scores, bins=50, alpha=0.7, label=f'Normal (n={len(normal_scores)})', color='green')
        axes[0].hist(defect_scores, bins=50, alpha=0.7, label=f'Defect (n={len(defect_scores)})', color='red')
        axes[0].axvline(self.threshold, color='black', linestyle='--', label=f'Threshold={self.threshold:.3f}')
        axes[0].set_xlabel('Anomaly Score')
        axes[0].set_ylabel('Count')
        axes[0].set_title('Score Distribution')
        axes[0].legend()

        # ROC Curve (defect = positive, 높은 점수 = defect)
        binary_labels = (labels > 0).astype(int)
        fpr, tpr, _ = roc_curve(binary_labels, scores)
        auc = roc_auc_score(binary_labels, scores)

        axes[1].plot(fpr, tpr, label=f'ROC (AUC={auc:.3f})')
        axes[1].plot([0, 1], [0, 1], 'k--')
        axes[1].set_xlabel('False Positive Rate')
        axes[1].set_ylabel('True Positive Rate')
        axes[1].set_title('ROC Curve')
        axes[1].legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ 분포 시각화 저장: {save_path}")
        else:
            plt.show()

        plt.close()


def main():
    parser = argparse.ArgumentParser(description='RGB AutoEncoder Testing')
    parser.add_argument('--checkpoint', type=str, required=True, help='체크포인트 파일 경로')
    parser.add_argument('--config', type=str, default='autoencoder_rgb', help='Config 파일 이름')
    parser.add_argument('--visualize', action='store_true', help='재구성 결과 시각화')
    parser.add_argument('--save-dir', type=str, default='models/rgb_ae/results', help='결과 저장 디렉토리')
    parser.add_argument('--log-dir', type=str, default=None, help='TensorBoard 로그 디렉토리')
    args = parser.parse_args()

    # TensorBoard 실행 안내
    print(f"\n{'='*60}")
    print(f"📊 TensorBoard 실행 명령어:")
    print(f"   tensorboard --logdir=models/rgb_ae/logs --port=6007")
    print(f"   http://localhost:6007")
    print(f"{'='*60}\n")

    # Config 로드 (선택적)
    config = None
    if args.config:
        try:
            config_loader = ConfigLoader()
            config = config_loader.load(args.config)
        except:
            pass

    # Tester 생성 및 테스트
    tester = AETester(checkpoint_path=args.checkpoint, config=config, log_dir=args.log_dir)
    results = tester.test()

    # 결과 저장 디렉토리
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 시각화
    if args.visualize:
        tester.visualize_reconstructions(
            num_samples=8,
            save_path=save_dir / 'reconstructions.png'
        )
        tester.plot_score_distribution(
            results['scores'],
            results['labels'],
            save_path=save_dir / 'score_distribution.png'
        )

    # 결과 JSON 저장
    def convert_to_serializable(obj):
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    results_json = {
        'checkpoint': args.checkpoint,
        'metrics': {k: convert_to_serializable(v)
                   for k, v in results['metrics'].items()
                   if k != 'confusion_matrix'},
        'confusion_matrix': results['metrics']['confusion_matrix'],
        'threshold': float(tester.threshold),
        'timestamp': datetime.now().isoformat()
    }

    with open(save_dir / 'test_results.json', 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"✅ 결과 저장: {save_dir / 'test_results.json'}")

    # TensorBoard writer 종료
    tester.writer.close()
    print(f"\n✅ 테스트 완료!")
    print(f"   TensorBoard 확인: tensorboard --logdir=models/rgb_ae/logs --port=6007")


if __name__ == "__main__":
    main()
