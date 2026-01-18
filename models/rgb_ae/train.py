"""RGB AutoEncoder 학습 스크립트"""
import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
_project_root = Path(__file__).parent.parent.parent.absolute()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path
import numpy as np
from tqdm import tqdm
import argparse
from datetime import datetime
import json

from models.rgb_ae.model import create_model, ConvAutoEncoder
from training.configs.config_loader import ConfigLoader
from training.data.dataset import BatteryDataset
from training.data.transforms import get_transforms, get_albumentations_transforms
from training.visualization.tensorboard_logger import TensorBoardLogger
from sklearn.metrics import roc_curve, roc_auc_score


class AETrainer:
    """AutoEncoder 학습 Trainer"""

    def __init__(self, config: dict):
        """
        Args:
            config: YAML config dict
        """
        self.config = config
        self.device = torch.device(config['training'].get('device', 'cuda'))

        # 모델 생성
        self.model = create_model(config).to(self.device)

        # 손실 함수
        loss_type = config['criteria'].get('reconstruction_loss', 'MSE')
        if loss_type == 'MSE':
            self.criterion = nn.MSELoss()
        elif loss_type == 'L1':
            self.criterion = nn.L1Loss()
        else:
            self.criterion = nn.MSELoss()

        # 옵티마이저
        self.optimizer = self._create_optimizer()

        # 스케줄러
        self.scheduler = self._create_scheduler()

        # Mixed Precision
        self.use_amp = config['training'].get('amp', True)
        self.scaler = GradScaler() if self.use_amp else None

        # Early Stopping
        es_config = config['criteria'].get('early_stopping', {})
        self.early_stopping_enabled = es_config.get('enabled', True)
        self.early_stopping_patience = es_config.get('patience', 15)
        self.early_stopping_min_delta = es_config.get('min_delta', 0.0001)
        self.best_val_loss = float('inf')
        self.patience_counter = 0

        # 체크포인트 디렉토리
        self.checkpoint_dir = Path(config['checkpoint']['save_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # TensorBoard Logger
        self.logger = self._create_logger()

        # 데이터 로더
        self.train_loader, self.val_loader = self._create_dataloaders()

        # Threshold (이상 탐지용)
        self.threshold = None

        print(f"✅ AE Trainer 초기화 완료")
        print(f"   - Device: {self.device}")
        print(f"   - Train samples: {len(self.train_loader.dataset)}")
        print(f"   - Val samples: {len(self.val_loader.dataset)}")
        print(f"   - AMP: {self.use_amp}")

    def _create_optimizer(self) -> optim.Optimizer:
        """옵티마이저 생성"""
        train_config = self.config['training']
        optimizer_name = train_config.get('optimizer', 'Adam')

        if optimizer_name == 'Adam':
            return optim.Adam(
                self.model.parameters(),
                lr=train_config['lr'],
                weight_decay=train_config.get('weight_decay', 0.0001)
            )
        elif optimizer_name == 'AdamW':
            return optim.AdamW(
                self.model.parameters(),
                lr=train_config['lr'],
                weight_decay=train_config.get('weight_decay', 0.01)
            )
        else:
            return optim.Adam(self.model.parameters(), lr=train_config['lr'])

    def _create_scheduler(self):
        """스케줄러 생성"""
        sched_config = self.config['training'].get('scheduler', {})
        sched_name = sched_config.get('name', 'ReduceLROnPlateau')

        if sched_name == 'ReduceLROnPlateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=float(sched_config.get('factor', 0.5)),
                patience=int(sched_config.get('patience', 5)),
                min_lr=float(sched_config.get('min_lr', 1e-6))
            )
        elif sched_name == 'CosineAnnealingLR':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['training']['epochs'],
                eta_min=float(sched_config.get('eta_min', 1e-6))
            )
        else:
            return None

    def _create_logger(self):
        """TensorBoard Logger 생성"""
        if not self.config['logging']['tensorboard'].get('enabled', True):
            return None

        # 간단한 TensorBoard writer
        from torch.utils.tensorboard import SummaryWriter
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_dir = Path(self.config['logging']['tensorboard']['log_dir']) / f'rgb_ae_{timestamp}'
        log_dir.mkdir(parents=True, exist_ok=True)
        return SummaryWriter(log_dir=str(log_dir))

    def _create_dataloaders(self):
        """데이터 로더 생성"""
        data_config = self.config['data']
        image_size = data_config['image_size']
        preprocessed = data_config.get('preprocessed', False)
        use_albumentations = data_config.get('use_albumentations', False)

        # Transform 선택
        if use_albumentations:
            train_transform = get_albumentations_transforms('rgb', 'train', image_size, preprocessed)
            val_transform = get_albumentations_transforms('rgb', 'val', image_size, preprocessed)
        else:
            train_transform = get_transforms('rgb', 'train', image_size, preprocessed)
            val_transform = get_transforms('rgb', 'val', image_size, preprocessed)

        # Dataset
        train_dataset = BatteryDataset(
            split_file=data_config['train_split'],
            transform=train_transform,
            modality='rgb',
            preprocessed=preprocessed
        )

        val_dataset = BatteryDataset(
            split_file=data_config['val_split'],
            transform=val_transform,
            modality='rgb',
            preprocessed=preprocessed
        )

        # DataLoader
        train_loader = DataLoader(
            train_dataset,
            batch_size=data_config['batch_size'],
            shuffle=True,
            num_workers=data_config.get('num_workers', 8),
            pin_memory=True,
            drop_last=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=data_config['batch_size'],
            shuffle=False,
            num_workers=data_config.get('num_workers', 8),
            pin_memory=True
        )

        return train_loader, val_loader

    def train_epoch(self, epoch: int) -> float:
        """1 epoch 학습"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} [Train]")
        for images, _ in pbar:  # labels는 사용 안함 (비지도 학습)
            images = images.to(self.device)

            self.optimizer.zero_grad()

            if self.use_amp:
                with autocast():
                    reconstructed, _ = self.model(images)
                    loss = self.criterion(reconstructed, images)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                reconstructed, _ = self.model(images)
                loss = self.criterion(reconstructed, images)
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        return total_loss / num_batches

    @torch.no_grad()
    def validate(self, epoch: int) -> dict:
        """Validation"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        all_anomaly_scores = []
        all_labels = []

        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1} [Val]")
        for images, labels in pbar:
            images = images.to(self.device)

            if self.use_amp:
                with autocast():
                    reconstructed, _ = self.model(images)
                    loss = self.criterion(reconstructed, images)
            else:
                reconstructed, _ = self.model(images)
                loss = self.criterion(reconstructed, images)

            # 이상 점수 계산
            anomaly_scores = self.model.get_anomaly_score(images)

            total_loss += loss.item()
            num_batches += 1
            all_anomaly_scores.extend(anomaly_scores.cpu().numpy())
            all_labels.extend(labels.numpy())

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / num_batches
        all_anomaly_scores = np.array(all_anomaly_scores)
        all_labels = np.array(all_labels)

        # 정상(0) vs 비정상(1,2) 분리
        normal_mask = all_labels == 0
        anomaly_mask = all_labels > 0

        metrics = {
            'val_loss': avg_loss,
            'anomaly_score_mean': all_anomaly_scores.mean(),
            'anomaly_score_std': all_anomaly_scores.std(),
        }

        if normal_mask.sum() > 0:
            metrics['normal_score_mean'] = all_anomaly_scores[normal_mask].mean()
        if anomaly_mask.sum() > 0:
            metrics['anomaly_score_mean_defect'] = all_anomaly_scores[anomaly_mask].mean()

        return metrics

    def train(self):
        """전체 학습 루프"""
        epochs = self.config['training']['epochs']
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        print(f"\n{'='*60}")
        print(f"RGB AutoEncoder 학습 시작")
        print(f"{'='*60}")
        print(f"Epochs: {epochs}")
        print(f"{'='*60}\n")

        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch(epoch)

            # Validate
            val_metrics = self.validate(epoch)
            val_loss = val_metrics['val_loss']

            # 스케줄러 업데이트
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # 현재 학습률
            current_lr = self.optimizer.param_groups[0]['lr']

            # 로깅
            print(f"\nEpoch {epoch+1}/{epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  LR: {current_lr:.6f}")
            if 'normal_score_mean' in val_metrics:
                print(f"  Normal Score: {val_metrics['normal_score_mean']:.4f}")
            if 'anomaly_score_mean_defect' in val_metrics:
                print(f"  Defect Score: {val_metrics['anomaly_score_mean_defect']:.4f}")

            # TensorBoard
            if self.logger is not None:
                self.logger.add_scalar('Loss/train', train_loss, epoch)
                self.logger.add_scalar('Loss/val', val_loss, epoch)
                self.logger.add_scalar('LR', current_lr, epoch)
                for key, value in val_metrics.items():
                    if key != 'val_loss':
                        self.logger.add_scalar(f'Metrics/{key}', value, epoch)

            # Best 모델 저장
            if val_loss < self.best_val_loss - self.early_stopping_min_delta:
                self.best_val_loss = val_loss
                self.patience_counter = 0

                # 체크포인트 저장
                checkpoint_path = self.checkpoint_dir / f'rgb_ae_best_{timestamp}.pt'
                self._save_checkpoint(checkpoint_path, epoch, val_metrics)
                print(f"  ✅ Best model saved: {checkpoint_path}")
            else:
                self.patience_counter += 1

            # Last 모델 저장
            last_path = self.checkpoint_dir / f'rgb_ae_last_{timestamp}.pt'
            self._save_checkpoint(last_path, epoch, val_metrics)

            # Early Stopping
            if self.early_stopping_enabled and self.patience_counter >= self.early_stopping_patience:
                print(f"\n⚠️ Early stopping at epoch {epoch+1}")
                break

        # 학습 완료 후 threshold 계산
        self._compute_threshold()

        print(f"\n{'='*60}")
        print(f"학습 완료!")
        print(f"Best Val Loss: {self.best_val_loss:.4f}")
        print(f"{'='*60}\n")

        if self.logger is not None:
            self.logger.close()

    def _save_checkpoint(self, path: Path, epoch: int, metrics: dict):
        """체크포인트 저장"""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': metrics['val_loss'],
            'config': self.config,
            'threshold': self.threshold,
            'timestamp': datetime.now().isoformat()
        }, path)

    @torch.no_grad()
    def _compute_threshold(self):
        """Threshold 계산 (Validation 데이터 기반, ROC 최적화)"""
        self.model.eval()
        all_scores = []
        all_labels = []

        print("\nThreshold 계산 중 (Validation 데이터 사용)...")
        for images, labels in tqdm(self.val_loader, desc="Computing threshold"):
            images = images.to(self.device)
            scores = self.model.get_anomaly_score(images)
            all_scores.extend(scores.cpu().numpy())
            all_labels.extend(labels.numpy())

        all_scores = np.array(all_scores)
        all_labels = np.array(all_labels)

        # 이진 분류로 변환 (0: normal, 1+: defect)
        binary_labels = (all_labels > 0).astype(int)

        # 점수 통계
        normal_scores = all_scores[all_labels == 0]
        defect_scores = all_scores[all_labels > 0]

        print(f"\n  점수 분포:")
        print(f"    Normal: {normal_scores.mean():.4f} ± {normal_scores.std():.4f} (n={len(normal_scores)})")
        print(f"    Defect: {defect_scores.mean():.4f} ± {defect_scores.std():.4f} (n={len(defect_scores)})")

        # ROC 기반 최적 threshold (Youden's J statistic: TPR - FPR 최대화)
        try:
            fpr, tpr, thresholds = roc_curve(binary_labels, all_scores)
            roc_auc = roc_auc_score(binary_labels, all_scores)

            # Youden's J = TPR - FPR 최대화 지점
            j_scores = tpr - fpr
            optimal_idx = np.argmax(j_scores)
            self.threshold = thresholds[optimal_idx]

            print(f"\n  ROC 분석:")
            print(f"    ROC-AUC: {roc_auc:.4f}")
            print(f"    Optimal Threshold: {self.threshold:.4f}")
            print(f"    TPR at threshold: {tpr[optimal_idx]:.4f}")
            print(f"    FPR at threshold: {fpr[optimal_idx]:.4f}")

        except Exception as e:
            # ROC 계산 실패 시 fallback to k-sigma
            print(f"\n  ⚠️ ROC 계산 실패, k-sigma 방식 사용: {e}")
            threshold_config = self.config.get('threshold', {})
            k = threshold_config.get('k', 2.5)
            self.threshold = all_scores.mean() + k * all_scores.std()
            roc_auc = 0.0

        # Threshold 저장
        threshold_path = self.checkpoint_dir / 'threshold.json'
        with open(threshold_path, 'w') as f:
            json.dump({
                'normal_mean': float(normal_scores.mean()),
                'normal_std': float(normal_scores.std()),
                'defect_mean': float(defect_scores.mean()),
                'defect_std': float(defect_scores.std()),
                'roc_auc': float(roc_auc),
                'threshold': float(self.threshold),
                'note': 'ROC 최적 threshold (Validation 데이터, Youden J 최대화)'
            }, f, indent=2)
        print(f"  Saved to: {threshold_path}")


def main():
    parser = argparse.ArgumentParser(description='RGB AutoEncoder Training')
    parser.add_argument('--config', type=str, default='autoencoder_rgb',
                       help='Config 파일 이름 (training/configs/ 내)')
    args = parser.parse_args()

    # Config 로드
    config_loader = ConfigLoader()
    config = config_loader.load(args.config)

    # TensorBoard 실행 안내
    log_dir = config['logging']['tensorboard'].get('log_dir', 'models/rgb_ae/logs')
    print(f"\n{'='*60}")
    print(f"📊 TensorBoard 실행 명령어:")
    print(f"   tensorboard --logdir={log_dir} --port=6007")
    print(f"   http://localhost:6007")
    print(f"{'='*60}\n")

    # Trainer 생성 및 학습
    trainer = AETrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
