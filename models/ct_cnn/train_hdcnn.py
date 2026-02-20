"""HD-CNN 학습 스크립트

HD-CNN (Hierarchical Deep CNN) 학습:
- Shared Layers: conv1 ~ layer2 (공유)
- Coarse Branch: layer3 ~ fc (독립) → Normal vs Defect
- Fine Branch: layer3 ~ fc (독립) → 5 classes

Loss:
- Coarse Loss: 전체 샘플
- Fine Loss: Defect 샘플만 (조건부)
"""
import sys
from pathlib import Path

# 프로젝트 루트
_project_root = Path(__file__).parent.parent.parent.absolute()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast
from collections import Counter
from pathlib import Path
import numpy as np
from tqdm import tqdm
import argparse
import csv
from datetime import datetime

from models.ct_cnn.model_hdcnn import HDCNN, HDCNNLoss, create_hdcnn_model
from training.configs.config_loader import ConfigLoader
from training.data.dataset_metadata import BatteryMetadataDataset
from training.data.transforms import build_transforms_from_config, get_transforms
from training.visualization.tensorboard_logger import TensorBoardLogger
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix


class HDCNNTrainer:
    """HD-CNN 학습 트레이너"""

    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device(config['training'].get('device', 'cuda'))
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 클래스 정보
        self.class_names = config['classes']['names']
        self.num_classes = config['classes']['num_classes']

        # 모델 생성
        self.model = create_hdcnn_model(
            num_fine_classes=self.num_classes,
            pretrained=config['model'].get('pretrained', True),
            dropout=config['model'].get('dropout', 0.5)
        ).to(self.device)

        # Loss
        class_weights = config['classes'].get('class_weights', None)
        if class_weights:
            class_weights = torch.tensor(class_weights, dtype=torch.float32).to(self.device)

        self.criterion = HDCNNLoss(
            coarse_weight=config['criteria'].get('coarse_weight', 1.0),
            fine_weight=config['criteria'].get('fine_weight', 1.0),
            fine_class_weights=class_weights,
            label_smoothing=config['criteria'].get('label_smoothing', 0.0)
        )

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['training']['lr'],
            weight_decay=config['training'].get('weight_decay', 0.01)
        )

        # DataLoader
        self._create_dataloaders()

        # Scheduler
        self._create_scheduler()

        # AMP
        self.use_amp = config['training'].get('amp', True)
        self.scaler = GradScaler() if self.use_amp else None

        # Gradient Clipping
        self.grad_clip = config['training'].get('gradient_clip', 1.0)

        # Checkpoint
        self.checkpoint_dir = Path(config['checkpoint']['save_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Logging
        self._setup_logging()

        # Best metric tracking
        self.best_metric = 0.0
        self.best_epoch = 0

        # Early stopping
        es_config = config['criteria'].get('early_stopping', {})
        self.early_stopping_enabled = es_config.get('enabled', True)
        self.patience = es_config.get('patience', 5)
        self.patience_counter = 0

    def _create_dataloaders(self):
        """데이터로더 생성"""
        config = self.config
        image_size = config['data']['image_size']
        batch_size = config['data']['batch_size']
        num_workers = config['data']['num_workers']
        preprocessed = config['data'].get('preprocessed', False)

        # Transform
        aug_config = config['data'].get('augmentation', None)
        if aug_config:
            train_transform = build_transforms_from_config(
                aug_config.get('train', []), 'ct', image_size, preprocessed
            )
            val_transform = build_transforms_from_config(
                aug_config.get('val', []), 'ct', image_size, preprocessed
            )
        else:
            train_transform = get_transforms('ct', 'train', image_size, preprocessed)
            val_transform = get_transforms('ct', 'val', image_size, preprocessed)

        # Split 파일
        train_split = str(_project_root / config['data']['train_split'])
        val_split = str(_project_root / config['data']['val_split'])

        # 라벨 경로
        label_base = config['data'].get('label_base', None)
        label_dirs = config['data'].get('label_dirs', None)

        self.train_dataset = BatteryMetadataDataset(
            split_file=train_split,
            modality='ct',
            mode='train',
            transform=train_transform,
            image_size=image_size,
            preprocessed=preprocessed,
            label_base=label_base,
            label_dirs=label_dirs
        )

        self.val_dataset = BatteryMetadataDataset(
            split_file=val_split,
            modality='ct',
            mode='val',
            transform=val_transform,
            image_size=image_size,
            preprocessed=preprocessed,
            label_base=label_base,
            label_dirs=label_dirs
        )

        # Weighted Sampler
        class_balancing = config['data'].get('class_balancing', {})
        train_sampler = None
        shuffle = True

        if class_balancing.get('enabled', False):
            labels = self.train_dataset.labels
            class_counts = Counter(labels)
            weights = [1.0 / class_counts[label] for label in labels]
            train_sampler = WeightedRandomSampler(weights, len(weights))
            shuffle = False
            print("   - WeightedRandomSampler 적용")

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

    def _create_scheduler(self):
        """스케줄러 생성"""
        scheduler_config = self.config['training'].get('scheduler', {})
        name = scheduler_config.get('name', 'CosineAnnealingWarmRestarts')

        if name == 'CosineAnnealingWarmRestarts':
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=int(scheduler_config.get('T_0', 10)),
                T_mult=int(scheduler_config.get('T_mult', 2)),
                eta_min=float(scheduler_config.get('eta_min', 1e-6))
            )
        else:
            self.scheduler = None

    def _setup_logging(self):
        """로깅 설정"""
        log_config = self.config['logging']

        # TensorBoard
        if log_config['tensorboard'].get('enabled', True):
            tb_config = self.config.copy()
            tb_config['model'] = self.config['model'].copy()
            tb_config['model']['name'] = f"hdcnn_{self.timestamp}"
            self.tb_logger = TensorBoardLogger(tb_config)
        else:
            self.tb_logger = None

        # CSV 로그
        train_log = log_config.get('train_log', {})
        if train_log.get('enabled', True):
            base_path = Path(train_log.get('save_path', 'models/ct_cnn/logs/train_hdcnn.csv'))
            self.csv_path = base_path.parent / f"train_hdcnn_{self.timestamp}.csv"
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)

            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'epoch', 'train_loss', 'train_coarse_loss', 'train_fine_loss',
                    'val_loss', 'val_coarse_loss', 'val_fine_loss',
                    'val_f1_macro', 'val_accuracy', 'coarse_accuracy', 'lr'
                ])

    def train_epoch(self, epoch: int) -> dict:
        """1 epoch 학습"""
        self.model.train()
        total_loss = 0.0
        total_coarse_loss = 0.0
        total_fine_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        for images, metadata, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device).long()

            self.optimizer.zero_grad()

            if self.use_amp:
                with autocast():
                    coarse_logits, fine_logits = self.model(images)
                    loss, coarse_loss, fine_loss = self.criterion(
                        coarse_logits, fine_logits, labels
                    )

                self.scaler.scale(loss).backward()
                if self.grad_clip:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                coarse_logits, fine_logits = self.model(images)
                loss, coarse_loss, fine_loss = self.criterion(
                    coarse_logits, fine_logits, labels
                )

                loss.backward()
                if self.grad_clip:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

            total_loss += loss.item()
            total_coarse_loss += coarse_loss.item()
            total_fine_loss += fine_loss.item()
            num_batches += 1

            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'coarse': f"{coarse_loss.item():.4f}",
                'fine': f"{fine_loss.item():.4f}"
            })

        if self.scheduler:
            self.scheduler.step()

        return {
            'loss': total_loss / num_batches,
            'coarse_loss': total_coarse_loss / num_batches,
            'fine_loss': total_fine_loss / num_batches
        }

    @torch.no_grad()
    def validate(self, epoch: int) -> dict:
        """Validation"""
        self.model.eval()
        total_loss = 0.0
        total_coarse_loss = 0.0
        total_fine_loss = 0.0
        num_batches = 0

        all_labels = []
        all_preds = []
        all_coarse_labels = []
        all_coarse_preds = []

        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]")
        for images, metadata, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device).long()

            coarse_logits, fine_logits = self.model(images)
            loss, coarse_loss, fine_loss = self.criterion(
                coarse_logits, fine_logits, labels
            )

            total_loss += loss.item()
            total_coarse_loss += coarse_loss.item()
            total_fine_loss += fine_loss.item()
            num_batches += 1

            # 계층적 예측
            final_preds, coarse_preds, _, _, _ = self.model.predict(images)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(final_preds.cpu().numpy())

            # Coarse labels
            coarse_labels = self.criterion.get_coarse_labels(labels)
            all_coarse_labels.extend(coarse_labels.cpu().numpy())
            all_coarse_preds.extend(coarse_preds.cpu().numpy())

        # Metrics
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        all_coarse_labels = np.array(all_coarse_labels)
        all_coarse_preds = np.array(all_coarse_preds)

        f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        accuracy = accuracy_score(all_labels, all_preds)
        coarse_accuracy = accuracy_score(all_coarse_labels, all_coarse_preds)

        # 클래스별 F1
        f1_per_class = f1_score(all_labels, all_preds, average=None, zero_division=0)

        return {
            'loss': total_loss / num_batches,
            'coarse_loss': total_coarse_loss / num_batches,
            'fine_loss': total_fine_loss / num_batches,
            'f1_macro': f1_macro,
            'accuracy': accuracy,
            'coarse_accuracy': coarse_accuracy,
            'f1_per_class': f1_per_class,
            'confusion_matrix': confusion_matrix(all_labels, all_preds)
        }

    def save_checkpoint(self, epoch: int, metrics: dict, is_best: bool = False):
        """체크포인트 저장"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_metric': self.best_metric,
            'config': self.config
        }

        # Last 저장
        last_path = self.checkpoint_dir / f"hdcnn_last_{self.timestamp}.pt"
        torch.save(checkpoint, last_path)

        # Best 저장
        if is_best:
            best_path = self.checkpoint_dir / f"hdcnn_best_{self.timestamp}.pt"
            torch.save(checkpoint, best_path)
            print(f"   ✅ Best model saved: {best_path}")

    def train(self):
        """전체 학습 루프"""
        epochs = self.config['training']['epochs']

        print(f"\n{'='*60}")
        print(f"HD-CNN 학습 시작")
        print(f"{'='*60}")
        print(f"  - Epochs: {epochs}")
        print(f"  - Device: {self.device}")
        print(f"  - Train samples: {len(self.train_dataset)}")
        print(f"  - Val samples: {len(self.val_dataset)}")
        print(f"  - AMP: {self.use_amp}")
        print(f"{'='*60}\n")

        # 모델 그래프 로깅 (한 번만)
        if self.tb_logger:
            try:
                dummy_input = torch.randn(1, 3, self.config['data']['image_size'],
                                         self.config['data']['image_size']).to(self.device)
                self.tb_logger.log_model_graph(self.model, dummy_input)
            except Exception as e:
                print(f"⚠️ 모델 그래프 로깅 실패: {e}")

        for epoch in range(1, epochs + 1):
            # Train
            train_metrics = self.train_epoch(epoch)

            # Validate
            val_metrics = self.validate(epoch)

            # Current LR
            current_lr = self.optimizer.param_groups[0]['lr']

            # Print results
            print(f"\nEpoch {epoch}/{epochs}:")
            print(f"  Train - Loss: {train_metrics['loss']:.4f} "
                  f"(Coarse: {train_metrics['coarse_loss']:.4f}, Fine: {train_metrics['fine_loss']:.4f})")
            print(f"  Val   - Loss: {val_metrics['loss']:.4f} "
                  f"(Coarse: {val_metrics['coarse_loss']:.4f}, Fine: {val_metrics['fine_loss']:.4f})")
            print(f"  Val   - F1: {val_metrics['f1_macro']:.4f}, "
                  f"Acc: {val_metrics['accuracy']:.4f}, "
                  f"Coarse Acc: {val_metrics['coarse_accuracy']:.4f}")
            print(f"  LR: {current_lr:.6f}")

            # 클래스별 F1
            print(f"  Per-class F1: ", end="")
            for i, name in enumerate(self.class_names):
                print(f"{name[:8]}={val_metrics['f1_per_class'][i]:.3f} ", end="")
            print()

            # Best model check
            is_best = val_metrics['f1_macro'] > self.best_metric
            if is_best:
                self.best_metric = val_metrics['f1_macro']
                self.best_epoch = epoch
                self.patience_counter = 0
                print(f"  ★ New best F1: {self.best_metric:.4f}")
            else:
                self.patience_counter += 1

            # Save checkpoint
            self.save_checkpoint(epoch, val_metrics, is_best)

            # CSV 로깅
            if hasattr(self, 'csv_path'):
                with open(self.csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        epoch,
                        f"{train_metrics['loss']:.4f}",
                        f"{train_metrics['coarse_loss']:.4f}",
                        f"{train_metrics['fine_loss']:.4f}",
                        f"{val_metrics['loss']:.4f}",
                        f"{val_metrics['coarse_loss']:.4f}",
                        f"{val_metrics['fine_loss']:.4f}",
                        f"{val_metrics['f1_macro']:.4f}",
                        f"{val_metrics['accuracy']:.4f}",
                        f"{val_metrics['coarse_accuracy']:.4f}",
                        f"{current_lr:.6f}"
                    ])

            # TensorBoard 로깅
            if self.tb_logger:
                self.tb_logger.log_scalars(epoch, {
                    'Loss/train': train_metrics['loss'],
                    'Loss/train_coarse': train_metrics['coarse_loss'],
                    'Loss/train_fine': train_metrics['fine_loss'],
                    'Loss/val': val_metrics['loss'],
                    'Loss/val_coarse': val_metrics['coarse_loss'],
                    'Loss/val_fine': val_metrics['fine_loss'],
                    'Metrics/val_f1_macro': val_metrics['f1_macro'],
                    'Metrics/val_accuracy': val_metrics['accuracy'],
                    'Metrics/coarse_accuracy': val_metrics['coarse_accuracy'],
                    'LR': current_lr
                })

                # 클래스별 F1
                for i, name in enumerate(self.class_names):
                    self.tb_logger.log_scalars(epoch, {
                        f'F1_PerClass/{name}': val_metrics['f1_per_class'][i]
                    })

                # Confusion Matrix 이미지
                self.tb_logger.log_confusion_matrix(
                    epoch, val_metrics['confusion_matrix'], self.class_names, tag='val'
                )

                # 클래스별 에러 분석 (TP/FP/FN/Precision/Recall)
                self.tb_logger.log_classification_errors(
                    epoch, val_metrics['confusion_matrix'], self.class_names
                )

                # 에러 요약 테이블 이미지
                self.tb_logger.log_error_summary_table(
                    epoch, val_metrics['confusion_matrix'], self.class_names
                )

            # Early stopping
            if self.early_stopping_enabled and self.patience_counter >= self.patience:
                print(f"\n⚠️ Early stopping at epoch {epoch} (patience={self.patience})")
                break

        # TensorBoard 종료
        if self.tb_logger:
            self.tb_logger.close()

        print(f"\n{'='*60}")
        print(f"학습 완료!")
        print(f"  - Best F1: {self.best_metric:.4f} (Epoch {self.best_epoch})")
        print(f"  - Checkpoint: {self.checkpoint_dir}")
        print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description='HD-CNN Training')
    parser.add_argument(
        '--config',
        type=str,
        default='cnn_ct_hdcnn',
        help='Config 파일 이름 (확장자 제외)'
    )
    args = parser.parse_args()

    # Config 로드
    config = ConfigLoader.load(args.config)

    # 학습
    trainer = HDCNNTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
