"""ResNet + Metadata Fusion 학습 스크립트"""
import sys
from pathlib import Path

_project_root = Path(__file__).parent.parent.parent.absolute()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, WeightedRandomSampler
import csv
from tqdm import tqdm
import numpy as np
from datetime import datetime
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
from collections import Counter

from models.ct_cnn.model_metadata import create_metadata_model
from training.configs.config_loader import ConfigLoader
from training.data.dataset_metadata import BatteryMetadataDataset
from training.data.transforms import build_transforms_from_config
from training.visualization.tensorboard_logger import TensorBoardLogger


class FocalLoss(nn.Module):
    """Focal Loss"""
    def __init__(self, gamma=2.0, alpha=None, label_smoothing=0.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        if alpha is not None:
            self.register_buffer('alpha', alpha)
        else:
            self.alpha = None

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(
            inputs, targets, reduction='none', label_smoothing=self.label_smoothing
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class MetadataTrainer:
    """ResNet + Metadata Fusion 학습기"""

    def __init__(self, config_path: str):
        self.config = ConfigLoader.load(config_path)
        self.device = torch.device(
            self.config['training']['device'] if torch.cuda.is_available() else 'cpu'
        )
        self.class_names = self.config['classes']['names']
        self.num_classes = self.config['classes']['num_classes']

        # 모델 생성
        self.model = create_metadata_model(
            num_classes=self.num_classes,
            pretrained=self.config['model'].get('pretrained', True),
            dropout=self.config['model'].get('dropout', 0.5),
            metadata_hidden_dim=self.config['model'].get('metadata_hidden_dim', 32),
            metadata_output_dim=self.config['model'].get('metadata_output_dim', 32),
            metadata_dropout=self.config['model'].get('metadata_dropout', 0.5),
            fusion_hidden_dim=self.config['model'].get('fusion_hidden_dim', 256)
        ).to(self.device)

        # 손실 함수
        class_weights = None
        if self.config['criteria'].get('use_class_weights', False):
            class_weights = torch.tensor(
                self.config['classes']['class_weights'], dtype=torch.float32
            ).to(self.device)

        focal_config = self.config['criteria'].get('focal_loss', {})
        if focal_config.get('enabled', False):
            self.criterion = FocalLoss(
                gamma=focal_config.get('gamma', 2.0),
                alpha=class_weights,
                label_smoothing=self.config['criteria'].get('label_smoothing', 0.0)
            )
        else:
            self.criterion = nn.CrossEntropyLoss(
                weight=class_weights,
                label_smoothing=self.config['criteria'].get('label_smoothing', 0.0)
            )

        # 옵티마이저
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['training']['lr'],
            weight_decay=self.config['training']['weight_decay']
        )

        # AMP
        self.use_amp = self.config['training'].get('amp', False)
        self.scaler = GradScaler('cuda') if self.use_amp else None
        self.grad_clip = self.config['training'].get('gradient_clip', None)

        # 데이터로더 생성
        self._create_dataloaders()

        # 스케줄러
        self._create_scheduler()

        # 로깅
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._setup_logging()

        # Early Stopping
        early_config = self.config['criteria'].get('early_stopping', {})
        self.early_stopping_enabled = early_config.get('enabled', False)
        self.patience = early_config.get('patience', 5)
        self.min_delta = early_config.get('min_delta', 0.001)
        self.best_metric = 0.0
        self.patience_counter = 0

        # 체크포인트
        self.checkpoint_dir = Path(self.config['checkpoint']['save_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

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
            from training.data.transforms import get_transforms
            train_transform = get_transforms('ct', 'train', image_size, preprocessed)
            val_transform = get_transforms('ct', 'val', image_size, preprocessed)

        # Dataset
        train_split = str(_project_root / config['data']['train_split'])
        val_split = str(_project_root / config['data']['val_split'])

        # 라벨 경로 (config에서 가져오기)
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
            self.tb_logger = TensorBoardLogger(self.config)
        else:
            self.tb_logger = None

        # CSV 로그
        train_log = log_config.get('train_log', {})
        if train_log.get('enabled', True):
            base_path = Path(train_log.get('save_path', 'models/ct_cnn/logs/train_metadata.csv'))
            self.csv_path = base_path.parent / f"train_metadata_{self.timestamp}.csv"
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)

            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_f1_macro', 'val_accuracy', 'lr'])

    def train_epoch(self, epoch: int) -> float:
        """1 epoch 학습"""
        self.model.train()
        total_loss = 0.0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        for images, metadata, labels in pbar:
            images = images.to(self.device)
            metadata = metadata.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            if self.use_amp:
                with autocast('cuda'):
                    outputs = self.model(images, metadata)
                    loss = self.criterion(outputs, labels)
                self.scaler.scale(loss).backward()
                if self.grad_clip:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images, metadata)
                loss = self.criterion(outputs, labels)
                loss.backward()
                if self.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def validate(self, epoch: int) -> dict:
        """검증"""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        misclassified_images = []
        misclassified_labels = []
        misclassified_preds = []
        misclassified_probs = []

        for images, metadata, labels in tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]"):
            images = images.to(self.device)
            metadata = metadata.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images, metadata)
            loss = self.criterion(outputs, labels)

            total_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            # 오분류 샘플 수집 (최대 100개)
            incorrect_mask = preds != labels
            if incorrect_mask.any() and len(misclassified_images) < 100:
                incorrect_indices = torch.where(incorrect_mask)[0]
                for idx in incorrect_indices:
                    if len(misclassified_images) >= 100:
                        break
                    img = images[idx].cpu().numpy().transpose(1, 2, 0)  # CHW → HWC
                    misclassified_images.append(img)
                    misclassified_labels.append(labels[idx].cpu().item())
                    misclassified_preds.append(preds[idx].cpu().item())
                    misclassified_probs.append(probs[idx].cpu().numpy())

        val_loss = total_loss / len(self.val_loader)
        val_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        val_acc = accuracy_score(all_labels, all_preds)

        # 클래스별 F1
        per_class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)

        return {
            'loss': val_loss,
            'f1_macro': val_f1,
            'accuracy': val_acc,
            'per_class_f1': per_class_f1,
            'preds': all_preds,
            'labels': all_labels,
            'probs': np.array(all_probs),
            'misclassified': {
                'images': misclassified_images,
                'labels': misclassified_labels,
                'preds': misclassified_preds,
                'probs': np.array(misclassified_probs) if misclassified_probs else np.array([])
            }
        }

    def train(self):
        """전체 학습 루프"""
        num_epochs = self.config['training']['epochs']

        print("=" * 60)
        print("ResNet + Metadata Fusion Training")
        print("=" * 60)
        print(f"  Device: {self.device}")
        print(f"  Model: ResNet18 + Metadata")
        print(f"  Train samples: {len(self.train_dataset)}")
        print(f"  Val samples: {len(self.val_dataset)}")
        print(f"  Epochs: {num_epochs}")
        print(f"  LR: {self.config['training']['lr']}")
        print("=" * 60)

        # 모델 그래프 로깅
        if self.tb_logger:
            image_size = self.config['data']['image_size']
            dummy_image = torch.randn(1, 3, image_size, image_size).to(self.device)
            dummy_metadata = torch.zeros(1, 2).to(self.device)  # [battery_type, axis]
            self.tb_logger.log_model_graph(self.model, (dummy_image, dummy_metadata))

        for epoch in range(1, num_epochs + 1):
            # Train
            train_loss = self.train_epoch(epoch)

            # Validate
            val_metrics = self.validate(epoch)

            # Scheduler step
            if self.scheduler:
                self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]['lr']

            # 로깅
            print(f"\nEpoch {epoch}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f}")
            print(f"  Val F1 Macro: {val_metrics['f1_macro']:.4f}")
            print(f"  Val Accuracy: {val_metrics['accuracy']:.4f}")
            print(f"  LR: {current_lr:.6f}")

            # 클래스별 F1
            print("  Per-class F1:")
            for i, name in enumerate(self.class_names):
                print(f"    {name}: {val_metrics['per_class_f1'][i]:.4f}")

            # TensorBoard
            if self.tb_logger:
                # 기본 메트릭
                metrics = {
                    'Loss/train': train_loss,
                    'Loss/val': val_metrics['loss'],
                    'Metrics/val_f1_macro': val_metrics['f1_macro'],
                    'Metrics/val_accuracy': val_metrics['accuracy'],
                    'LR': current_lr
                }
                # 클래스별 F1 추가
                for i, name in enumerate(self.class_names):
                    metrics[f'PerClass/F1/{name}'] = val_metrics['per_class_f1'][i]

                self.tb_logger.log_scalars(epoch, metrics)

                # Confusion Matrix 로깅
                cm = confusion_matrix(val_metrics['labels'], val_metrics['preds'])
                self.tb_logger.log_confusion_matrix(epoch, cm, self.class_names)

                # Classification Errors (TP/FP/FN)
                self.tb_logger.log_classification_errors(epoch, cm, self.class_names)

                # Error Summary Table
                self.tb_logger.log_error_summary_table(epoch, cm, self.class_names)

                # PR Curves
                self.tb_logger.log_pr_curves(
                    epoch,
                    np.array(val_metrics['labels']),
                    val_metrics['probs'],
                    self.class_names
                )

                # Prediction Confidence
                self.tb_logger.log_prediction_confidence(
                    epoch,
                    val_metrics['probs'],
                    np.array(val_metrics['preds']),
                    np.array(val_metrics['labels'])
                )

                # Misclassified Images (오분류 이미지)
                misclassified = val_metrics['misclassified']
                if len(misclassified['images']) > 0:
                    self.tb_logger.log_misclassified_images(
                        epoch,
                        misclassified['images'],
                        misclassified['labels'],
                        misclassified['preds'],
                        misclassified['probs'],
                        self.class_names,
                        num_samples=4
                    )

            # CSV 로그
            if hasattr(self, 'csv_path'):
                with open(self.csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        epoch, f"{train_loss:.4f}", f"{val_metrics['loss']:.4f}",
                        f"{val_metrics['f1_macro']:.4f}", f"{val_metrics['accuracy']:.4f}",
                        f"{current_lr:.6f}"
                    ])

            # 체크포인트 저장
            if val_metrics['f1_macro'] > self.best_metric:
                self.best_metric = val_metrics['f1_macro']
                self.patience_counter = 0
                self._save_checkpoint(epoch, 'best')
                print(f"  ✅ Best model saved (F1: {self.best_metric:.4f})")
            else:
                self.patience_counter += 1

            # Early Stopping
            if self.early_stopping_enabled and self.patience_counter >= self.patience:
                print(f"\n⚠️ Early Stopping: {self.patience} epochs 동안 개선 없음")
                break

        # 최종 체크포인트
        self._save_checkpoint(epoch, 'last')
        print(f"\n학습 완료! Best F1: {self.best_metric:.4f}")

    def _save_checkpoint(self, epoch: int, tag: str):
        """체크포인트 저장"""
        path = self.checkpoint_dir / f"metadata_{tag}_{self.timestamp}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_metric': self.best_metric,
            'config': self.config
        }, path)


def main():
    parser = argparse.ArgumentParser(description='ResNet + Metadata Fusion Training')
    parser.add_argument('--config', type=str, required=True, help='Config 파일 경로')
    args = parser.parse_args()

    trainer = MetadataTrainer(args.config)
    trainer.train()


if __name__ == "__main__":
    main()
