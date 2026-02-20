"""CT AutoEncoder 학습 스크립트

CT 이미지에서 정상 패턴을 학습하여 이상 탐지 (Anomaly Detection)
- 정상 이미지만으로 학습 (normal_classes: 0, 2)
- 결함 이미지 입력 시 높은 재구성 오류 발생
- CNN+Metadata와 앙상블하여 사용
"""
import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
_project_root = Path(__file__).parent.parent.parent.absolute()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path
import numpy as np
from tqdm import tqdm
import argparse
from datetime import datetime
import json
import csv

from models.rgb_ae.model import create_model, ConvAutoEncoder
from training.configs.config_loader import ConfigLoader
from training.data.dataset import BatteryDataset
from training.data.transforms import get_transforms, build_transforms_from_config
from sklearn.metrics import roc_curve, roc_auc_score


class CTAETrainer:
    """CT AutoEncoder 학습 Trainer (Anomaly Detection)"""

    def __init__(self, config: dict):
        """
        Args:
            config: YAML config dict
        """
        self.config = config
        self.device = torch.device(config['training'].get('device', 'cuda'))

        # 모델 생성
        self.model = self._create_model().to(self.device)

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
        self.early_stopping_patience = es_config.get('patience', 10)
        self.early_stopping_min_delta = es_config.get('min_delta', 0.0001)
        self.best_val_loss = float('inf')
        self.patience_counter = 0

        # 체크포인트 디렉토리
        self.checkpoint_dir = Path(config['checkpoint']['save_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # TensorBoard Logger
        self.logger = self._create_logger()

        # CSV Logger
        self.csv_log_path = self._setup_csv_logger()

        # 데이터 로더 (정상 이미지만 학습)
        self.train_loader, self.val_loader, self.val_full_loader = self._create_dataloaders()

        # Normal 클래스 정의
        self.normal_classes = config['classes'].get('normal_classes', [0, 2])

        # Threshold (이상 탐지용)
        self.threshold = None

        print(f"CT AE Trainer 초기화 완료")
        print(f"   - Device: {self.device}")
        print(f"   - Normal classes: {self.normal_classes}")
        print(f"   - Train samples (normal only): {len(self.train_loader.dataset)}")
        print(f"   - Val samples (normal only): {len(self.val_loader.dataset)}")
        print(f"   - Val samples (all): {len(self.val_full_loader.dataset)}")
        print(f"   - AMP: {self.use_amp}")

    def _create_model(self) -> nn.Module:
        """모델 생성"""
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

        return model

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
        elif sched_name == 'CosineAnnealingWarmRestarts':
            return optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=int(sched_config.get('T_0', 10)),
                T_mult=int(sched_config.get('T_mult', 2)),
                eta_min=float(sched_config.get('eta_min', 1e-6))
            )
        else:
            return None

    def _create_logger(self):
        """TensorBoard Logger 생성"""
        tb_config = self.config['logging'].get('tensorboard', {})
        if not tb_config.get('enabled', True):
            return None

        from torch.utils.tensorboard import SummaryWriter
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_dir = Path(tb_config.get('log_dir', 'models/ct_ae/logs')) / f'ct_ae_{timestamp}'
        log_dir.mkdir(parents=True, exist_ok=True)
        print(f"   - TensorBoard: {log_dir}")
        return SummaryWriter(log_dir=str(log_dir))

    def _setup_csv_logger(self) -> Path:
        """CSV Logger 설정"""
        train_log_config = self.config['logging'].get('train_log', {})
        if not train_log_config.get('enabled', True):
            return None

        csv_path = Path(train_log_config.get('save_path', 'models/ct_ae/logs/train_ct_ae.csv'))
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        # 헤더 작성
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'epoch', 'train_loss', 'val_loss', 'lr',
                'normal_score_mean', 'normal_score_std',
                'defect_score_mean', 'defect_score_std',
                'roc_auc', 'best_val_loss', 'timestamp'
            ])

        print(f"   - CSV Log: {csv_path}")
        return csv_path

    def _log_to_csv(self, epoch: int, metrics: dict):
        """CSV 로그 기록"""
        if self.csv_log_path is None:
            return

        with open(self.csv_log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                f"{metrics.get('train_loss', 0):.6f}",
                f"{metrics.get('val_loss', 0):.6f}",
                f"{metrics.get('lr', 0):.8f}",
                f"{metrics.get('normal_score_mean', 0):.6f}",
                f"{metrics.get('normal_score_std', 0):.6f}",
                f"{metrics.get('defect_score_mean', 0):.6f}",
                f"{metrics.get('defect_score_std', 0):.6f}",
                f"{metrics.get('roc_auc', 0):.4f}",
                f"{self.best_val_loss:.6f}",
                datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            ])

    def _create_dataloaders(self):
        """데이터 로더 생성 (정상 이미지만 필터링)"""
        data_config = self.config['data']
        image_size = data_config['image_size']
        preprocessed = data_config.get('preprocessed', True)
        augmentation_config = data_config.get('augmentation', None)
        normal_classes = self.config['classes'].get('normal_classes', [0, 2])

        # Transform 선택
        if augmentation_config is not None:
            train_aug = augmentation_config.get('train', [])
            val_aug = augmentation_config.get('val', [])
            train_transform = build_transforms_from_config(train_aug, 'ct', image_size, preprocessed)
            val_transform = build_transforms_from_config(val_aug, 'ct', image_size, preprocessed)
            print(f"   - Augmentation: Config 기반 ({len(train_aug)}개 transform)")
        else:
            train_transform = get_transforms('ct', 'train', image_size, preprocessed)
            val_transform = get_transforms('ct', 'val', image_size, preprocessed)

        # Full Dataset
        train_dataset_full = BatteryDataset(
            split_file=data_config['train_split'],
            transform=train_transform,
            modality='ct',
            preprocessed=preprocessed
        )

        val_dataset_full = BatteryDataset(
            split_file=data_config['val_split'],
            transform=val_transform,
            modality='ct',
            preprocessed=preprocessed
        )

        # 정상 이미지만 필터링 (Anomaly Detection 학습용)
        # 이미지 로드 없이 라벨만으로 필터링 (빠름)
        print(f"   - Filtering normal images...")

        if data_config.get('train_normal_only', True):
            train_normal_indices = [
                i for i, label in enumerate(train_dataset_full.labels)
                if label in normal_classes
            ]
        else:
            train_normal_indices = list(range(len(train_dataset_full)))

        val_normal_indices = [
            i for i, label in enumerate(val_dataset_full.labels)
            if label in normal_classes
        ]

        print(f"     Train: {len(train_normal_indices)} / {len(train_dataset_full)} (normal only)")
        print(f"     Val: {len(val_normal_indices)} / {len(val_dataset_full)} (normal only)")

        # Normal-only Dataset (재생성하여 인덱스 사용)
        train_dataset_full = BatteryDataset(
            split_file=data_config['train_split'],
            transform=train_transform,
            modality='ct',
            preprocessed=preprocessed
        )

        val_dataset_full = BatteryDataset(
            split_file=data_config['val_split'],
            transform=val_transform,
            modality='ct',
            preprocessed=preprocessed
        )

        train_dataset_normal = Subset(train_dataset_full, train_normal_indices)
        val_dataset_normal = Subset(val_dataset_full, val_normal_indices)

        # DataLoader
        train_loader = DataLoader(
            train_dataset_normal,
            batch_size=data_config['batch_size'],
            shuffle=True,
            num_workers=data_config.get('num_workers', 4),
            pin_memory=True,
            drop_last=True
        )

        val_loader = DataLoader(
            val_dataset_normal,
            batch_size=data_config['batch_size'],
            shuffle=False,
            num_workers=data_config.get('num_workers', 4),
            pin_memory=True
        )

        # Full validation loader (정상 + 결함 모두 - threshold 계산용)
        val_full_loader = DataLoader(
            val_dataset_full,
            batch_size=data_config['batch_size'],
            shuffle=False,
            num_workers=data_config.get('num_workers', 4),
            pin_memory=True
        )

        return train_loader, val_loader, val_full_loader

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
        """Validation (정상 이미지만)"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1} [Val-Normal]")
        for images, _ in pbar:
            images = images.to(self.device)

            if self.use_amp:
                with autocast():
                    reconstructed, _ = self.model(images)
                    loss = self.criterion(reconstructed, images)
            else:
                reconstructed, _ = self.model(images)
                loss = self.criterion(reconstructed, images)

            total_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        return {'val_loss': total_loss / num_batches if num_batches > 0 else 0}

    @torch.no_grad()
    def validate_full(self, epoch: int) -> dict:
        """Full Validation (정상 + 결함 - Anomaly Detection 성능 측정)"""
        self.model.eval()
        all_scores = []
        all_labels = []

        pbar = tqdm(self.val_full_loader, desc=f"Epoch {epoch+1} [Val-Full]")
        for images, labels in pbar:
            images = images.to(self.device)
            scores = self.model.get_anomaly_score(images)
            all_scores.extend(scores.cpu().numpy())
            all_labels.extend(labels.numpy())

        all_scores = np.array(all_scores)
        all_labels = np.array(all_labels)

        # 정상 vs 비정상 분리
        normal_mask = np.isin(all_labels, self.normal_classes)
        defect_mask = ~normal_mask

        metrics = {}

        if normal_mask.sum() > 0:
            metrics['normal_score_mean'] = float(all_scores[normal_mask].mean())
            metrics['normal_score_std'] = float(all_scores[normal_mask].std())
        else:
            metrics['normal_score_mean'] = 0.0
            metrics['normal_score_std'] = 0.0

        if defect_mask.sum() > 0:
            metrics['defect_score_mean'] = float(all_scores[defect_mask].mean())
            metrics['defect_score_std'] = float(all_scores[defect_mask].std())

            # ROC-AUC 계산
            try:
                binary_labels = (~normal_mask).astype(int)  # 0: normal, 1: defect
                roc_auc = roc_auc_score(binary_labels, all_scores)
                metrics['roc_auc'] = float(roc_auc)
            except Exception:
                metrics['roc_auc'] = 0.0
        else:
            metrics['defect_score_mean'] = 0.0
            metrics['defect_score_std'] = 0.0
            metrics['roc_auc'] = 0.0

        return metrics

    def train(self):
        """전체 학습 루프"""
        epochs = self.config['training']['epochs']
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        print(f"\n{'='*60}")
        print(f"CT AutoEncoder 학습 시작 (Anomaly Detection)")
        print(f"{'='*60}")
        print(f"Epochs: {epochs}")
        print(f"Normal Classes: {self.normal_classes}")
        print(f"{'='*60}\n")

        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch(epoch)

            # Validate (normal only)
            val_metrics = self.validate(epoch)
            val_loss = val_metrics['val_loss']

            # Validate Full (for anomaly detection metrics)
            full_metrics = self.validate_full(epoch)

            # 스케줄러 업데이트
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # 현재 학습률
            current_lr = self.optimizer.param_groups[0]['lr']

            # 메트릭 병합
            all_metrics = {
                'train_loss': train_loss,
                'val_loss': val_loss,
                'lr': current_lr,
                **full_metrics
            }

            # 로깅
            print(f"\nEpoch {epoch+1}/{epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss (normal): {val_loss:.4f}")
            print(f"  LR: {current_lr:.6f}")
            if 'normal_score_mean' in full_metrics:
                print(f"  Normal Score: {full_metrics['normal_score_mean']:.4f} +/- {full_metrics['normal_score_std']:.4f}")
            if 'defect_score_mean' in full_metrics:
                print(f"  Defect Score: {full_metrics['defect_score_mean']:.4f} +/- {full_metrics['defect_score_std']:.4f}")
            if 'roc_auc' in full_metrics:
                print(f"  ROC-AUC: {full_metrics['roc_auc']:.4f}")

            # TensorBoard
            if self.logger is not None:
                self.logger.add_scalar('Loss/train', train_loss, epoch)
                self.logger.add_scalar('Loss/val', val_loss, epoch)
                self.logger.add_scalar('LR', current_lr, epoch)
                for key, value in full_metrics.items():
                    self.logger.add_scalar(f'Metrics/{key}', value, epoch)

            # CSV 로깅
            self._log_to_csv(epoch, all_metrics)

            # Best 모델 저장
            if val_loss < self.best_val_loss - self.early_stopping_min_delta:
                self.best_val_loss = val_loss
                self.patience_counter = 0

                # 체크포인트 저장
                checkpoint_path = self.checkpoint_dir / f'ct_ae_best_{timestamp}.pt'
                self._save_checkpoint(checkpoint_path, epoch, all_metrics)
                print(f"  Best model saved: {checkpoint_path}")
            else:
                self.patience_counter += 1

            # Last 모델 저장
            last_path = self.checkpoint_dir / f'ct_ae_last_{timestamp}.pt'
            self._save_checkpoint(last_path, epoch, all_metrics)

            # Early Stopping
            if self.early_stopping_enabled and self.patience_counter >= self.early_stopping_patience:
                print(f"\n Early stopping at epoch {epoch+1}")
                break

        # 학습 완료 후 threshold 계산
        print(f"\n{'='*60}")
        print("Threshold 계산 중...")
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
            'val_loss': metrics.get('val_loss', 0),
            'config': self.config,
            'threshold': self.threshold,
            'normal_classes': self.normal_classes,
            'timestamp': datetime.now().isoformat()
        }, path)

    @torch.no_grad()
    def _compute_threshold(self):
        """Threshold 계산 (Validation 데이터 기반, ROC 최적화)"""
        self.model.eval()
        all_scores = []
        all_labels = []

        print("Validation 데이터로 threshold 계산 중...")
        for images, labels in tqdm(self.val_full_loader, desc="Computing threshold"):
            images = images.to(self.device)
            scores = self.model.get_anomaly_score(images)
            all_scores.extend(scores.cpu().numpy())
            all_labels.extend(labels.numpy())

        all_scores = np.array(all_scores)
        all_labels = np.array(all_labels)

        # 정상 vs 비정상
        normal_mask = np.isin(all_labels, self.normal_classes)
        normal_scores = all_scores[normal_mask]
        defect_scores = all_scores[~normal_mask]

        print(f"\n  점수 분포:")
        print(f"    Normal: {normal_scores.mean():.4f} +/- {normal_scores.std():.4f} (n={len(normal_scores)})")
        if len(defect_scores) > 0:
            print(f"    Defect: {defect_scores.mean():.4f} +/- {defect_scores.std():.4f} (n={len(defect_scores)})")

        # ROC 기반 최적 threshold (Youden's J statistic)
        roc_auc = 0.0
        try:
            if len(defect_scores) > 0:
                binary_labels = (~normal_mask).astype(int)
                fpr, tpr, thresholds = roc_curve(binary_labels, all_scores)
                roc_auc = roc_auc_score(binary_labels, all_scores)

                # Youden's J = TPR - FPR 최대화 지점
                j_scores = tpr - fpr
                optimal_idx = np.argmax(j_scores)
                self.threshold = float(thresholds[optimal_idx])

                print(f"\n  ROC 분석:")
                print(f"    ROC-AUC: {roc_auc:.4f}")
                print(f"    Optimal Threshold: {self.threshold:.4f}")
                print(f"    TPR at threshold: {tpr[optimal_idx]:.4f}")
                print(f"    FPR at threshold: {fpr[optimal_idx]:.4f}")
            else:
                raise ValueError("No defect samples for ROC")

        except Exception as e:
            # ROC 계산 실패 시 fallback to k-sigma
            print(f"\n  ROC 계산 실패, k-sigma 방식 사용: {e}")
            threshold_config = self.config.get('threshold', {})
            k = threshold_config.get('k', 2.5)
            self.threshold = float(normal_scores.mean() + k * normal_scores.std())
            print(f"    k-sigma threshold (k={k}): {self.threshold:.4f}")

        # Threshold 저장
        threshold_path = self.checkpoint_dir / 'threshold.json'
        threshold_data = {
            'normal_mean': float(normal_scores.mean()),
            'normal_std': float(normal_scores.std()),
            'threshold': self.threshold,
            'method': 'roc_youden' if roc_auc > 0 else 'k_sigma',
            'roc_auc': float(roc_auc),
            'normal_classes': self.normal_classes,
            'timestamp': datetime.now().isoformat()
        }

        if len(defect_scores) > 0:
            threshold_data['defect_mean'] = float(defect_scores.mean())
            threshold_data['defect_std'] = float(defect_scores.std())

        with open(threshold_path, 'w') as f:
            json.dump(threshold_data, f, indent=2)
        print(f"  Saved to: {threshold_path}")

        # Best 모델에 threshold 추가 저장
        best_checkpoints = list(self.checkpoint_dir.glob('ct_ae_best_*.pt'))
        if best_checkpoints:
            latest_best = max(best_checkpoints, key=lambda p: p.stat().st_mtime)
            checkpoint = torch.load(latest_best, weights_only=False)
            checkpoint['threshold'] = self.threshold
            torch.save(checkpoint, latest_best)
            print(f"  Threshold added to: {latest_best}")


def main():
    parser = argparse.ArgumentParser(description='CT AutoEncoder Training (Anomaly Detection)')
    parser.add_argument('--config', type=str, default='autoencoder_ct',
                       help='Config 파일 이름 (training/configs/ 내)')
    args = parser.parse_args()

    # Config 로드
    config_loader = ConfigLoader()
    config = config_loader.load(args.config)

    # TensorBoard 실행 안내
    log_dir = config['logging']['tensorboard'].get('log_dir', 'models/ct_ae/logs')
    print(f"\n{'='*60}")
    print(f"TensorBoard 실행 명령어:")
    print(f"   tensorboard --logdir={log_dir} --port=6008")
    print(f"   http://localhost:6008")
    print(f"{'='*60}\n")

    # Trainer 생성 및 학습
    trainer = CTAETrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
