"""Hierarchical CNN 학습 스크립트 - 2단계 분류

1단계 (Coarse): Normal vs Defect
2단계 (Fine): Defect 세부 분류 (Defect 샘플만 학습)

클래스 매핑:
    Coarse 0 (Normal): cell_normal(0), module_normal(2)
    Coarse 1 (Defect): cell_porosity(1), module_porosity(3), module_resin(4)
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
import csv
from tqdm import tqdm
import numpy as np
from datetime import datetime
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report, precision_score, recall_score

from models.ct_cnn.model_hierarchical import HierarchicalResNet18, HierarchicalLoss
from models.ct_cnn.gradcam_hierarchical import HierarchicalGradCAM
from training.configs.config_loader import ConfigLoader
from training.data.dataloader import create_dataloaders
from training.visualization.tensorboard_logger import TensorBoardLogger


class HierarchicalTrainer:
    """Hierarchical CNN 학습기 - 2단계 분류"""

    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device(
            config['training']['device'] if torch.cuda.is_available() else 'cpu'
        )

        # 클래스 정보
        self.class_names = config['classes']['names']
        self.num_classes = config['classes']['num_classes']
        self.coarse_names = ['Normal', 'Defect']

        # Normal/Defect 클래스 매핑
        self.normal_classes = [0, 2]  # cell_normal, module_normal
        self.defect_classes = [1, 3, 4]  # cell_porosity, module_porosity, resin

        # 모델 생성
        self.model = HierarchicalResNet18(
            num_fine_classes=self.num_classes,
            pretrained=config['model'].get('pretrained', True),
            dropout=config['model'].get('dropout', 0.3)
        ).to(self.device)

        # 클래스 가중치
        fine_class_weights = None
        if config['criteria'].get('use_class_weights', False):
            fine_class_weights = torch.tensor(
                config['classes']['class_weights'],
                dtype=torch.float32
            ).to(self.device)

        # Hierarchical Loss
        self.criterion = HierarchicalLoss(
            coarse_weight=config['criteria'].get('coarse_weight', 1.0),
            fine_weight=config['criteria'].get('fine_weight', 1.0),
            fine_class_weights=fine_class_weights,
            label_smoothing=config['criteria'].get('label_smoothing', 0.0)
        )

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
        augmentation_config = config['data'].get('augmentation', None)
        class_balancing = config['data'].get('class_balancing', None)

        self.train_loader, self.val_loader = create_dataloaders(
            train_split_file=config['data']['train_split'],
            val_split_file=config['data']['val_split'],
            batch_size=config['data']['batch_size'],
            num_workers=config['data']['num_workers'],
            image_size=config['data']['image_size'],
            modality='ct',
            preprocessed=config['data'].get('preprocessed', False),
            use_albumentations=config['data'].get('use_albumentations', False),
            augmentation_config=augmentation_config,
            class_balancing=class_balancing
        )

        # TensorBoard Logger
        self.use_tensorboard = config['logging']['tensorboard'].get('enabled', True)
        self.log_grad_cam = config['logging']['tensorboard'].get('log_grad_cam', False)

        if self.use_tensorboard:
            self.tb_logger = TensorBoardLogger(config)
            log_dir = config['logging']['tensorboard'].get('log_dir', 'models/ct_cnn/logs')
            print(f"\n{'='*60}")
            print(f"📊 TensorBoard: tensorboard --logdir={log_dir}")
            print(f"{'='*60}\n")

            # 모델 구조 그래프 로깅
            sample_input = torch.randn(1, 3, config['data']['image_size'], config['data']['image_size']).to(self.device)
            self.tb_logger.log_model_graph(self.model, sample_input)
        else:
            self.tb_logger = None

        # Early Stopping
        early_stop_config = config['criteria'].get('early_stopping', {})
        self.patience = early_stop_config.get('patience', 10)
        self.min_delta = early_stop_config.get('min_delta', 0.001)
        self.monitor_metric = early_stop_config.get('monitor', 'val_f1_macro')
        self.monitor_mode = early_stop_config.get('mode', 'max')

        self.best_metric_value = float('-inf') if self.monitor_mode == 'max' else float('inf')
        self.patience_counter = 0

        # Timestamp & Checkpoint
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.checkpoint_dir = Path(config['checkpoint']['save_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # CSV 로그
        log_config = config['logging']['train_log']
        if log_config.get('enabled', True):
            base_path = Path(log_config['save_path'])
            self.train_log_path = base_path.parent / f"train_hierarchical_{self.timestamp}.csv"
            self.train_log_path.parent.mkdir(parents=True, exist_ok=True)
            self._init_csv_log()
        else:
            self.train_log_path = None

        self._print_init_info()

    def _create_scheduler(self, config):
        """스케줄러 생성"""
        scheduler_config = config['training'].get('scheduler', {})
        name = scheduler_config.get('name', 'CosineAnnealingWarmRestarts')

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
                mode='max',
                factor=float(scheduler_config.get('factor', 0.5)),
                patience=int(scheduler_config.get('patience', 5)),
                min_lr=float(scheduler_config.get('min_lr', 1e-6)),
                verbose=True
            )
        return None

    def _print_init_info(self):
        """초기화 정보 출력"""
        print(f"\n{'='*60}")
        print(f"Hierarchical CNN Trainer 초기화")
        print(f"{'='*60}")
        print(f"  Device: {self.device}")
        print(f"  Model: Hierarchical ResNet18")
        print(f"  Coarse: Normal vs Defect")
        print(f"    - Normal: {[self.class_names[i] for i in self.normal_classes]}")
        print(f"    - Defect: {[self.class_names[i] for i in self.defect_classes]}")
        print(f"  Fine classes: {self.num_classes}")
        print(f"  Train samples: {len(self.train_loader.dataset)}")
        print(f"  Val samples: {len(self.val_loader.dataset)}")
        print(f"{'='*60}\n")

    def _init_csv_log(self):
        """CSV 로그 초기화"""
        with open(self.train_log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            header = [
                'epoch', 'train_loss', 'train_coarse_loss', 'train_fine_loss',
                'val_loss', 'val_coarse_acc', 'val_fine_f1', 'val_final_f1', 'lr'
            ]
            writer.writerow(header)

    def _log_to_csv(self, epoch, metrics, lr):
        """CSV 로그 기록"""
        if self.train_log_path:
            with open(self.train_log_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch,
                    f"{metrics['train_loss']:.4f}",
                    f"{metrics['train_coarse_loss']:.4f}",
                    f"{metrics['train_fine_loss']:.4f}",
                    f"{metrics['val_loss']:.4f}",
                    f"{metrics['val_coarse_acc']:.4f}",
                    f"{metrics['val_fine_f1']:.4f}",
                    f"{metrics['val_final_f1']:.4f}",
                    f"{lr:.6f}"
                ])

    def _get_coarse_labels(self, fine_labels: torch.Tensor) -> torch.Tensor:
        """Fine labels를 Coarse labels로 변환"""
        coarse_labels = torch.zeros_like(fine_labels)
        for defect_class in self.defect_classes:
            coarse_labels[fine_labels == defect_class] = 1
        return coarse_labels

    def _log_gradcam_samples(self, epoch: int, num_samples: int = 8):
        """
        Hierarchical Grad-CAM 시각화 샘플 로깅

        Args:
            epoch: 에폭 번호
            num_samples: 로깅할 샘플 수
        """
        try:
            # Grad-CAM 생성기 초기화
            gradcam = HierarchicalGradCAM(self.model, target_layer='layer4')

            self.model.eval()

            images_list = []
            coarse_heatmaps = []
            fine_heatmaps = []
            labels_list = []
            final_preds_list = []

            sample_count = 0
            for images, labels in self.val_loader:
                if sample_count >= num_samples:
                    break

                images = images.to(self.device)

                batch_size = min(images.size(0), num_samples - sample_count)

                for i in range(batch_size):
                    img = images[i:i+1]
                    label = labels[i].item()

                    # Hierarchical Grad-CAM
                    result = gradcam.generate_both(img)

                    # 이미지 변환
                    img_np = img.squeeze().cpu().numpy()
                    if img_np.ndim == 3:
                        img_np = img_np.transpose(1, 2, 0)
                    img_np = img_np * 0.5 + 0.5
                    img_np = np.clip(img_np, 0, 1)

                    images_list.append(img_np)
                    coarse_heatmaps.append(result['coarse'][0])
                    fine_heatmaps.append(result['fine'][0])
                    labels_list.append(label)
                    final_preds_list.append(result['final_pred'])

                    sample_count += 1

                if sample_count >= num_samples:
                    break

            # TensorBoard에 로깅 (Fine heatmap 사용)
            if images_list:
                self.tb_logger.log_gradcam(
                    epoch,
                    np.array(images_list),
                    np.array(fine_heatmaps),
                    np.array(labels_list),
                    np.array(final_preds_list),
                    self.class_names,
                    num_samples=num_samples // 2
                )

        except Exception as e:
            print(f"⚠️ Hierarchical Grad-CAM 로깅 실패: {e}")

    def train_epoch(self) -> dict:
        """1 Epoch 학습"""
        self.model.train()
        total_loss = 0.0
        total_coarse_loss = 0.0
        total_fine_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc="Training")
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

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
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                coarse_logits, fine_logits = self.model(images)
                loss, coarse_loss, fine_loss = self.criterion(
                    coarse_logits, fine_logits, labels
                )
                loss.backward()

                if self.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

                self.optimizer.step()

            total_loss += loss.item()
            total_coarse_loss += coarse_loss.item()
            total_fine_loss += fine_loss.item() if isinstance(fine_loss, torch.Tensor) else fine_loss
            num_batches += 1

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'coarse': f'{coarse_loss.item():.4f}'
            })

        return {
            'train_loss': total_loss / num_batches,
            'train_coarse_loss': total_coarse_loss / num_batches,
            'train_fine_loss': total_fine_loss / num_batches
        }

    def validate(self) -> dict:
        """Validation - 순차 추론 사용"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        all_labels = []
        all_coarse_labels = []
        all_coarse_preds = []
        all_fine_preds = []
        all_final_preds = []
        all_coarse_probs = []
        all_fine_probs = []

        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc="Validation"):
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Forward
                coarse_logits, fine_logits = self.model(images)
                loss, _, _ = self.criterion(coarse_logits, fine_logits, labels)

                total_loss += loss.item()
                num_batches += 1

                # 순차 추론
                final_preds, coarse_preds, fine_preds, coarse_probs, fine_probs = self.model.predict(images)

                # Coarse labels
                coarse_labels = self._get_coarse_labels(labels)

                all_labels.extend(labels.cpu().numpy())
                all_coarse_labels.extend(coarse_labels.cpu().numpy())
                all_coarse_preds.extend(coarse_preds.cpu().numpy())
                all_fine_preds.extend(fine_preds.cpu().numpy())
                all_final_preds.extend(final_preds.cpu().numpy())
                all_coarse_probs.append(coarse_probs.cpu().numpy())
                all_fine_probs.append(fine_probs.cpu().numpy())

        # 메트릭 계산
        all_labels = np.array(all_labels)
        all_coarse_labels = np.array(all_coarse_labels)
        all_coarse_preds = np.array(all_coarse_preds)
        all_fine_preds = np.array(all_fine_preds)
        all_final_preds = np.array(all_final_preds)
        all_coarse_probs = np.vstack(all_coarse_probs)
        all_fine_probs = np.vstack(all_fine_probs)

        # Coarse accuracy
        coarse_acc = accuracy_score(all_coarse_labels, all_coarse_preds)

        # Fine F1 (Defect만)
        defect_mask = (all_coarse_labels == 1)
        if defect_mask.sum() > 0:
            fine_f1 = f1_score(
                all_labels[defect_mask],
                all_fine_preds[defect_mask],
                average='macro',
                zero_division=0
            )
        else:
            fine_f1 = 0.0

        # Final F1 (순차 추론 결과)
        final_f1 = f1_score(all_labels, all_final_preds, average='macro', zero_division=0)
        final_acc = accuracy_score(all_labels, all_final_preds)

        # 클래스별 메트릭
        f1_per_class = f1_score(all_labels, all_final_preds, average=None, zero_division=0)
        precision_per_class = precision_score(all_labels, all_final_preds, average=None, zero_division=0)
        recall_per_class = recall_score(all_labels, all_final_preds, average=None, zero_division=0)

        # Confusion matrices
        coarse_cm = confusion_matrix(all_coarse_labels, all_coarse_preds, labels=[0, 1])
        final_cm = confusion_matrix(all_labels, all_final_preds, labels=range(self.num_classes))

        return {
            'val_loss': total_loss / num_batches,
            'val_coarse_acc': coarse_acc,
            'val_fine_f1': fine_f1,
            'val_final_f1': final_f1,
            'val_final_acc': final_acc,
            'coarse_cm': coarse_cm,
            'final_cm': final_cm,
            'all_labels': all_labels,
            'all_final_preds': all_final_preds,
            'all_fine_probs': all_fine_probs,
            'f1_per_class': f1_per_class,
            'precision_per_class': precision_per_class,
            'recall_per_class': recall_per_class
        }

    def save_checkpoint(self, epoch: int, metric_value: float, is_best: bool = False):
        """체크포인트 저장"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_metric_name': self.monitor_metric,
            'best_metric_value': metric_value,
            'config': self.config,
            'class_names': self.class_names,
            'timestamp': self.timestamp
        }

        if is_best:
            best_path = self.checkpoint_dir / f'hierarchical_best_{self.timestamp}.pt'
            torch.save(checkpoint, best_path)
            print(f"  ✅ Best 모델 저장: {best_path.name}")

        if self.config['checkpoint'].get('save_last', True):
            last_path = self.checkpoint_dir / f'hierarchical_last_{self.timestamp}.pt'
            torch.save(checkpoint, last_path)

    def train(self):
        """전체 학습 루프"""
        num_epochs = self.config['training']['epochs']

        print(f"\n{'='*60}")
        print(f"Hierarchical 학습 시작: {num_epochs} epochs")
        print(f"{'='*60}\n")

        for epoch in range(1, num_epochs + 1):
            print(f"\n[Epoch {epoch}/{num_epochs}]")

            # Train
            train_metrics = self.train_epoch()

            # Validation
            val_metrics = self.validate()

            # 현재 LR
            current_lr = self.optimizer.param_groups[0]['lr']

            # 결과 출력
            print(f"  Train Loss: {train_metrics['train_loss']:.4f} "
                  f"(Coarse: {train_metrics['train_coarse_loss']:.4f}, "
                  f"Fine: {train_metrics['train_fine_loss']:.4f})")
            print(f"  Val Loss: {val_metrics['val_loss']:.4f}")
            print(f"  Val Coarse Acc: {val_metrics['val_coarse_acc']:.4f}")
            print(f"  Val Fine F1 (Defect only): {val_metrics['val_fine_f1']:.4f}")
            print(f"  Val Final F1 (Sequential): {val_metrics['val_final_f1']:.4f}")
            print(f"  Val Final Acc: {val_metrics['val_final_acc']:.4f}")

            # Coarse Confusion Matrix
            print(f"\n  Coarse CM (Normal vs Defect):")
            print(f"              Pred_N  Pred_D")
            print(f"    Normal    {val_metrics['coarse_cm'][0][0]:6d}  {val_metrics['coarse_cm'][0][1]:6d}")
            print(f"    Defect    {val_metrics['coarse_cm'][1][0]:6d}  {val_metrics['coarse_cm'][1][1]:6d}")

            # 클래스별 성능
            print(f"\n  Fine 클래스별 성능:")
            report = classification_report(
                val_metrics['all_labels'],
                val_metrics['all_final_preds'],
                target_names=self.class_names,
                zero_division=0
            )
            for line in report.split('\n')[2:-4]:
                if line.strip():
                    print(f"    {line}")

            # TensorBoard 로깅
            if self.use_tensorboard:
                self.tb_logger.log_scalars(epoch, {
                    'Loss/train_total': train_metrics['train_loss'],
                    'Loss/train_coarse': train_metrics['train_coarse_loss'],
                    'Loss/train_fine': train_metrics['train_fine_loss'],
                    'Loss/val': val_metrics['val_loss'],
                    'Metrics/coarse_acc': val_metrics['val_coarse_acc'],
                    'Metrics/fine_f1': val_metrics['val_fine_f1'],
                    'Metrics/final_f1': val_metrics['val_final_f1'],
                    'Metrics/final_acc': val_metrics['val_final_acc'],
                    'LR': current_lr
                })

                # Confusion Matrix
                self.tb_logger.log_confusion_matrix(
                    epoch, val_metrics['final_cm'], self.class_names, 'val_final'
                )

                # FP/FN 에러 분석
                self.tb_logger.log_classification_errors(
                    epoch, val_metrics['final_cm'], self.class_names
                )
                self.tb_logger.log_error_summary_table(
                    epoch, val_metrics['final_cm'], self.class_names
                )

                # 클래스별 F1/Precision/Recall
                self.tb_logger.log_per_class_metrics(
                    epoch,
                    {
                        'F1': val_metrics['f1_per_class'],
                        'Precision': val_metrics['precision_per_class'],
                        'Recall': val_metrics['recall_per_class']
                    },
                    self.class_names
                )

                # PR Curve
                self.tb_logger.log_pr_curves(
                    epoch, val_metrics['all_labels'], val_metrics['all_fine_probs'], self.class_names
                )

                # 클래스 분포
                self.tb_logger.log_class_distribution(
                    epoch, val_metrics['all_labels'], self.class_names, 'val'
                )

                # 예측 확률 히스토그램
                self.tb_logger.log_probability_histograms(
                    epoch, val_metrics['all_fine_probs'], val_metrics['all_labels'], self.class_names
                )

                # 예측 신뢰도 분포
                self.tb_logger.log_prediction_confidence(
                    epoch, val_metrics['all_fine_probs'], val_metrics['all_final_preds'], val_metrics['all_labels']
                )

                # Hierarchical Grad-CAM (매 5 에폭마다)
                if self.log_grad_cam and epoch % 5 == 0:
                    self._log_gradcam_samples(epoch)

            # CSV 로깅
            all_metrics = {**train_metrics, **val_metrics}
            self._log_to_csv(epoch, all_metrics, current_lr)

            # Scheduler 업데이트
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['val_final_f1'])
                else:
                    self.scheduler.step()

            # Best 모델 저장 & Early Stopping
            current_metric = val_metrics['val_final_f1']

            if self.monitor_mode == 'max':
                is_improvement = current_metric > self.best_metric_value + self.min_delta
            else:
                is_improvement = current_metric < self.best_metric_value - self.min_delta

            if is_improvement:
                self.best_metric_value = current_metric
                self.patience_counter = 0
                self.save_checkpoint(epoch, current_metric, is_best=True)
            else:
                self.patience_counter += 1
                self.save_checkpoint(epoch, current_metric, is_best=False)

                if self.patience_counter >= self.patience:
                    print(f"\n⚠️ Early Stopping: {self.patience} epochs 동안 개선 없음")
                    break

        print(f"\n{'='*60}")
        print(f"✅ Hierarchical 학습 완료!")
        print(f"   Best Final F1: {self.best_metric_value:.4f}")
        print(f"   체크포인트: {self.checkpoint_dir}")
        print(f"{'='*60}")

        if self.use_tensorboard:
            self.tb_logger.close()


def main():
    parser = argparse.ArgumentParser(description='Hierarchical CNN 학습')
    parser.add_argument(
        '--config',
        type=str,
        default='cnn_ct_hierarchical',
        help='Config 파일 이름 (확장자 제외)'
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
    trainer = HierarchicalTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
