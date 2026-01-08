"""CT 통합 CNN 학습 스크립트 (5클래스 다중 분류)

클래스:
    0: cell_normal
    1: cell_porosity
    2: module_normal
    3: module_porosity
    4: module_resin_overflow
"""
import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
_project_root = Path(__file__).parent.parent.parent.absolute()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path
import csv
from tqdm import tqdm
import numpy as np
from datetime import datetime
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report, precision_score, recall_score

from models.ct_cnn.model import create_model
from training.configs.config_loader import ConfigLoader
from training.data.dataloader import create_dataloaders, create_test_dataloader
from training.visualization.tensorboard_logger import TensorBoardLogger


class CTUnifiedTrainer:
    """CT 통합 CNN 학습기 (5클래스 다중 분류)"""

    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device(
            config['training']['device'] if torch.cuda.is_available() else 'cpu'
        )

        # 클래스 정보
        self.class_names = config['classes']['names']
        self.num_classes = config['classes']['num_classes']

        # 모델 생성
        self.model = create_model(config).to(self.device)

        # 클래스 가중치
        class_weights = None
        if config['criteria'].get('use_class_weights', False):
            class_weights = torch.tensor(
                config['classes']['class_weights'],
                dtype=torch.float32
            ).to(self.device)

        # Loss Function (다중분류)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Optimizer
        optimizer_name = config['training'].get('optimizer', 'AdamW')
        if optimizer_name == 'AdamW':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=config['training']['lr'],
                weight_decay=config['training']['weight_decay']
            )
        else:
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=config['training']['lr'],
                weight_decay=config['training']['weight_decay']
            )

        # Scheduler
        self.scheduler = self._create_scheduler(config)

        # Mixed Precision
        self.use_amp = config['training'].get('amp', False)
        self.scaler = GradScaler() if self.use_amp else None

        # Gradient Clipping
        self.grad_clip = config['training'].get('gradient_clip', None)

        # DataLoader
        self.train_loader, self.val_loader = create_dataloaders(
            train_split_file=config['data']['train_split'],
            val_split_file=config['data']['val_split'],
            batch_size=config['data']['batch_size'],
            num_workers=config['data']['num_workers'],
            image_size=config['data']['image_size'],
            modality='ct'
        )

        # TensorBoard Logger
        self.tb_logger = TensorBoardLogger(config)

        # TensorBoard URL 출력
        log_dir = config['logging']['tensorboard'].get('log_dir', 'models/ct_cnn/logs')
        print(f"\n{'='*60}")
        print(f"📊 TensorBoard 실행 명령어:")
        print(f"   tensorboard --logdir={log_dir} --port=6006")
        print(f"   http://localhost:6006")
        print(f"{'='*60}\n")

        # 모델 구조 그래프 로깅
        sample_input = torch.randn(1, 3, config['data']['image_size'], config['data']['image_size']).to(self.device)
        self.tb_logger.log_model_graph(self.model, sample_input)

        # Early Stopping
        self.best_f1 = 0.0
        self.patience_counter = 0
        early_stop_config = config['criteria'].get('early_stopping', {})
        self.patience = early_stop_config.get('patience', 10)
        self.min_delta = early_stop_config.get('min_delta', 0.001)

        # Timestamp
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Checkpoint 디렉토리
        self.checkpoint_dir = Path(config['checkpoint']['save_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # CSV 로그
        log_config = config['logging']['train_log']
        if log_config.get('enabled', True):
            base_path = Path(log_config['save_path'])
            self.train_log_path = base_path.parent / f"{base_path.stem}_{self.timestamp}{base_path.suffix}"
            self.train_log_path.parent.mkdir(parents=True, exist_ok=True)
            self._init_csv_log()
        else:
            self.train_log_path = None

        self._print_init_info()

    def _create_scheduler(self, config):
        """스케줄러 생성"""
        scheduler_config = config['training'].get('scheduler', {})
        name = scheduler_config.get('name', 'ReduceLROnPlateau')

        if name == 'CosineAnnealingWarmRestarts':
            return optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=int(scheduler_config.get('T_0', 10)),
                T_mult=int(scheduler_config.get('T_mult', 2)),
                eta_min=float(scheduler_config.get('eta_min', 1e-6))
            )
        elif name == 'ReduceLROnPlateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',  # F1 최대화
                factor=float(scheduler_config.get('factor', 0.5)),
                patience=int(scheduler_config.get('patience', 5)),
                min_lr=float(scheduler_config.get('min_lr', 1e-6)),
                verbose=True
            )
        return None

    def _print_init_info(self):
        """초기화 정보 출력"""
        print(f"\n{'='*60}")
        print(f"CT 통합 CNN Trainer 초기화")
        print(f"{'='*60}")
        print(f"  Device: {self.device}")
        print(f"  Model: {self.config['model']['name']}")
        print(f"  Classes: {self.num_classes}")
        for i, name in enumerate(self.class_names):
            print(f"    {i}: {name}")
        print(f"  Loss: CrossEntropyLoss")
        print(f"  Optimizer: {self.config['training'].get('optimizer', 'AdamW')}")
        print(f"  LR: {self.config['training']['lr']}")
        print(f"  AMP: {self.use_amp}")
        print(f"  Train samples: {len(self.train_loader.dataset)}")
        print(f"  Val samples: {len(self.val_loader.dataset)}")
        print(f"{'='*60}\n")

    def _init_csv_log(self):
        """CSV 로그 초기화"""
        with open(self.train_log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['epoch', 'train_loss', 'val_loss', 'val_f1_macro', 'val_accuracy', 'lr']
            writer.writerow(header)

    def _log_to_csv(self, epoch, train_loss, val_loss, val_f1, val_acc, lr):
        """CSV 로그 기록"""
        if self.train_log_path:
            with open(self.train_log_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch,
                    f"{train_loss:.4f}",
                    f"{val_loss:.4f}",
                    f"{val_f1:.4f}",
                    f"{val_acc:.4f}",
                    f"{lr:.6f}"
                ])

    def train_epoch(self) -> float:
        """1 Epoch 학습"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc="Training")
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)  # (B,) - 클래스 인덱스

            self.optimizer.zero_grad()

            if self.use_amp:
                with autocast():
                    outputs = self.model(images)  # (B, num_classes)
                    loss = self.criterion(outputs, labels)

                self.scaler.scale(loss).backward()

                if self.grad_clip:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()

                if self.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

                self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        return total_loss / num_batches

    def validate(self) -> tuple:
        """Validation"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        all_labels = []
        all_preds = []
        all_probs = []

        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc="Validation"):
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)  # (B, num_classes)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                num_batches += 1

                # 예측 (Softmax + argmax)
                probs = torch.softmax(outputs, dim=1)  # (B, num_classes)
                preds = torch.argmax(probs, dim=1)  # (B,)

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_probs.append(probs.cpu().numpy())

        avg_loss = total_loss / num_batches
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        all_probs = np.vstack(all_probs)

        # 메트릭 계산
        f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        accuracy = accuracy_score(all_labels, all_preds)
        cm = confusion_matrix(all_labels, all_preds, labels=range(self.num_classes))

        # 클래스별 메트릭
        f1_per_class = f1_score(all_labels, all_preds, average=None, zero_division=0)
        precision_per_class = precision_score(all_labels, all_preds, average=None, zero_division=0)
        recall_per_class = recall_score(all_labels, all_preds, average=None, zero_division=0)

        metrics = {
            'f1_macro': f1_macro,
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'f1_per_class': f1_per_class,
            'precision_per_class': precision_per_class,
            'recall_per_class': recall_per_class,
            'all_labels': all_labels,
            'all_preds': all_preds,
            'all_probs': all_probs
        }

        return avg_loss, metrics

    def save_checkpoint(self, epoch: int, val_f1: float, is_best: bool = False):
        """체크포인트 저장"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_f1_macro': val_f1,
            'config': self.config,
            'class_names': self.class_names,
            'timestamp': self.timestamp
        }

        if is_best:
            best_path = self.checkpoint_dir / f'ct_unified_best_{self.timestamp}.pt'
            torch.save(checkpoint, best_path)
            print(f"  ✅ Best 모델 저장: {best_path}")

        if self.config['checkpoint'].get('save_last', True):
            last_path = self.checkpoint_dir / f'ct_unified_last_{self.timestamp}.pt'
            torch.save(checkpoint, last_path)

    def train(self):
        """전체 학습 루프"""
        num_epochs = self.config['training']['epochs']

        print(f"\n{'='*60}")
        print(f"학습 시작: {num_epochs} epochs")
        print(f"{'='*60}\n")

        for epoch in range(1, num_epochs + 1):
            print(f"\n[Epoch {epoch}/{num_epochs}]")

            # Train
            train_loss = self.train_epoch()

            # Validation
            val_loss, metrics = self.validate()

            # 현재 LR
            current_lr = self.optimizer.param_groups[0]['lr']

            # 결과 출력
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Val F1 (macro): {metrics['f1_macro']:.4f}")
            print(f"  Val Accuracy: {metrics['accuracy']:.4f}")
            print(f"  LR: {current_lr:.6f}")

            # 클래스별 성능
            print(f"\n  클래스별 성능:")
            report = classification_report(
                metrics['all_labels'],
                metrics['all_preds'],
                target_names=self.class_names,
                zero_division=0
            )
            for line in report.split('\n')[2:-4]:  # 클래스별 라인만
                if line.strip():
                    print(f"    {line}")

            # TensorBoard 로깅
            self.tb_logger.log_scalars(epoch, {
                'Loss/train': train_loss,
                'Loss/val': val_loss,
                'Metrics/f1_macro': metrics['f1_macro'],
                'Metrics/accuracy': metrics['accuracy'],
                'LR': current_lr
            })

            # Confusion Matrix 로깅
            self.tb_logger.log_confusion_matrix(
                epoch, metrics['confusion_matrix'], self.class_names, 'val'
            )

            # FP/FN 에러 분석 로깅
            self.tb_logger.log_classification_errors(
                epoch, metrics['confusion_matrix'], self.class_names
            )
            self.tb_logger.log_error_summary_table(
                epoch, metrics['confusion_matrix'], self.class_names
            )

            # 클래스별 F1/Precision/Recall 로깅
            self.tb_logger.log_per_class_metrics(
                epoch,
                {
                    'F1': metrics['f1_per_class'],
                    'Precision': metrics['precision_per_class'],
                    'Recall': metrics['recall_per_class']
                },
                self.class_names
            )

            # PR Curve 로깅
            self.tb_logger.log_pr_curves(
                epoch, metrics['all_labels'], metrics['all_probs'], self.class_names
            )

            # 클래스 분포 시각화
            self.tb_logger.log_class_distribution(
                epoch, metrics['all_labels'], self.class_names, 'val'
            )

            # 예측 확률 히스토그램
            self.tb_logger.log_probability_histograms(
                epoch, metrics['all_probs'], metrics['all_labels'], self.class_names
            )

            # 예측 신뢰도 분포
            self.tb_logger.log_prediction_confidence(
                epoch, metrics['all_probs'], metrics['all_preds'], metrics['all_labels']
            )

            # CSV 로깅
            self._log_to_csv(epoch, train_loss, val_loss, metrics['f1_macro'], metrics['accuracy'], current_lr)

            # Scheduler 업데이트
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(metrics['f1_macro'])
                else:
                    self.scheduler.step()

            # Best 모델 저장 & Early Stopping
            if metrics['f1_macro'] > self.best_f1 + self.min_delta:
                self.best_f1 = metrics['f1_macro']
                self.patience_counter = 0
                self.save_checkpoint(epoch, metrics['f1_macro'], is_best=True)
            else:
                self.patience_counter += 1
                self.save_checkpoint(epoch, metrics['f1_macro'], is_best=False)

                if self.patience_counter >= self.patience:
                    print(f"\n⚠️ Early Stopping: {self.patience} epochs 동안 개선 없음")
                    break

        print(f"\n{'='*60}")
        print(f"✅ 학습 완료!")
        print(f"   Best Val F1 (macro): {self.best_f1:.4f}")
        print(f"   체크포인트: {self.checkpoint_dir}")
        print(f"{'='*60}")

        self.tb_logger.close()


def main():
    parser = argparse.ArgumentParser(description='CT 통합 CNN 학습')
    parser.add_argument(
        '--config',
        type=str,
        default='training/configs/cnn_ct_unified.yaml',
        help='Config 파일 경로'
    )
    args = parser.parse_args()

    # Config 로드
    config = ConfigLoader.load(args.config)

    # 시드 설정
    seed = config.get('experiment', {}).get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 학습
    trainer = CTUnifiedTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
