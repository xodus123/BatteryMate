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
from models.inspector.gradcam import GradCAM
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification

    논문: "Focal Loss for Dense Object Detection" (Lin et al., 2017)

    FL(p_t) = -α_t × (1 - p_t)^γ × log(p_t)

    - γ (gamma): focusing parameter - 쉬운 샘플 down-weight (기본: 2.0)
    - α (alpha): class weight - 클래스 불균형 처리 (기본: None)

    γ가 클수록 쉬운 샘플(높은 확률)의 loss가 줄어들고,
    어려운 샘플(낮은 확률)에 더 집중함
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: torch.Tensor = None,
        label_smoothing: float = 0.0,
        reduction: str = 'mean'
    ):
        """
        Args:
            gamma: Focusing parameter (0이면 일반 CE와 동일)
            alpha: 클래스별 가중치 텐서 [num_classes]
            label_smoothing: Label smoothing 값 (0.0 ~ 1.0)
            reduction: 'mean', 'sum', 'none'
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (N, C) - 모델 출력 (logits)
            targets: (N,) - 정답 클래스 인덱스

        Returns:
            Focal Loss 값
        """
        num_classes = inputs.size(1)

        # Label Smoothing 적용
        if self.label_smoothing > 0:
            # One-hot encoding
            targets_one_hot = F.one_hot(targets, num_classes).float()
            # Smooth labels
            targets_smooth = targets_one_hot * (1 - self.label_smoothing) + \
                           self.label_smoothing / num_classes
        else:
            targets_smooth = None

        # Softmax 확률 계산
        p = F.softmax(inputs, dim=1)

        # 정답 클래스의 확률 추출
        if targets_smooth is not None:
            # Label smoothing 사용 시
            p_t = (p * targets_smooth).sum(dim=1)
        else:
            # One-hot 없이 직접 추출
            p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)

        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma

        # Cross entropy (log softmax)
        if targets_smooth is not None:
            ce_loss = -(targets_smooth * F.log_softmax(inputs, dim=1)).sum(dim=1)
        else:
            ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        # Focal Loss
        focal_loss = focal_weight * ce_loss

        # Alpha (class weight) 적용
        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets)
            focal_loss = alpha_t * focal_loss

        # Reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


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

        # Loss Function 설정
        self.criterion, self.loss_name = self._create_loss_function(config, class_weights)

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
        # Augmentation config 가져오기
        augmentation_config = config['data'].get('augmentation', None)
        # Class balancing config 가져오기
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

        # TensorBoard Logger (Config에서 enabled 확인)
        self.use_tensorboard = config['logging']['tensorboard'].get('enabled', True)
        self.log_grad_cam = config['logging']['tensorboard'].get('log_grad_cam', False)

        if self.use_tensorboard:
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
        else:
            self.tb_logger = None
            print(f"\n⚠️ TensorBoard 비활성화됨 (config: tensorboard.enabled=false)")

        # Early Stopping & Best Model 설정
        early_stop_config = config['criteria'].get('early_stopping', {})
        self.patience = early_stop_config.get('patience', 10)
        self.min_delta = early_stop_config.get('min_delta', 0.001)
        self.monitor_metric = early_stop_config.get('monitor', 'val_f1_macro')
        self.monitor_mode = early_stop_config.get('mode', 'max')  # 'max' or 'min'

        # checkpoint.save_best_by도 확인 (early_stopping.monitor와 동일하게 사용)
        self.save_best_by = config['checkpoint'].get('save_best_by', self.monitor_metric)

        # Best 값 초기화 (mode에 따라)
        self.best_metric_value = float('-inf') if self.monitor_mode == 'max' else float('inf')
        self.patience_counter = 0

        # Timestamp
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Checkpoint 디렉토리
        self.checkpoint_dir = Path(config['checkpoint']['save_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Top-K 체크포인트 설정
        self.save_top_k = config['checkpoint'].get('save_top_k', 1)
        self.top_k_checkpoints = []  # [(metric_value, path), ...]

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

    def _create_loss_function(self, config: dict, class_weights: torch.Tensor = None):
        """
        Config 기반 Loss Function 생성

        Args:
            config: 설정 딕셔너리
            class_weights: 클래스 가중치 텐서

        Returns:
            (criterion, loss_name)
        """
        criteria_config = config.get('criteria', {})
        focal_config = criteria_config.get('focal_loss', {})
        label_smoothing = criteria_config.get('label_smoothing', 0.0)

        # Focal Loss 사용 여부 확인
        if focal_config.get('enabled', False):
            gamma = focal_config.get('gamma', 2.0)
            # alpha는 focal_loss config에서 가져오거나 class_weights 사용
            alpha = focal_config.get('alpha', None)
            if alpha is None and class_weights is not None:
                alpha = class_weights

            criterion = FocalLoss(
                gamma=gamma,
                alpha=alpha,
                label_smoothing=label_smoothing,
                reduction='mean'
            )
            loss_name = f"FocalLoss(γ={gamma}, smooth={label_smoothing})"
            print(f"  ✅ Focal Loss 활성화: gamma={gamma}, label_smoothing={label_smoothing}")

        # Label Smoothing만 사용
        elif label_smoothing > 0:
            criterion = nn.CrossEntropyLoss(
                weight=class_weights,
                label_smoothing=label_smoothing
            )
            loss_name = f"CrossEntropyLoss(smooth={label_smoothing})"
            print(f"  ✅ Label Smoothing 활성화: {label_smoothing}")

        # 기본 CrossEntropyLoss
        else:
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            loss_name = "CrossEntropyLoss"

        return criterion, loss_name

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
        print(f"  Loss: {self.loss_name}")
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

    def _get_monitor_value(self, val_loss: float, metrics: dict) -> float:
        """
        모니터링 지표 값 추출

        Args:
            val_loss: Validation loss
            metrics: 메트릭 딕셔너리

        Returns:
            모니터링 지표 값
        """
        metric_map = {
            'val_f1_macro': metrics.get('f1_macro', 0.0),
            'val_accuracy': metrics.get('accuracy', 0.0),
            'val_loss': val_loss,
            'f1_macro': metrics.get('f1_macro', 0.0),
            'accuracy': metrics.get('accuracy', 0.0),
            'loss': val_loss,
        }
        return metric_map.get(self.monitor_metric, metrics.get('f1_macro', 0.0))

    def _check_improvement(self, current_value: float) -> bool:
        """
        개선 여부 확인

        Args:
            current_value: 현재 지표 값

        Returns:
            개선되었으면 True
        """
        if self.monitor_mode == 'max':
            return current_value > self.best_metric_value + self.min_delta
        else:  # min
            return current_value < self.best_metric_value - self.min_delta

    def _log_gradcam_samples(self, epoch: int, metrics: dict, num_samples: int = 8):
        """
        Grad-CAM 시각화 샘플 로깅

        Args:
            epoch: 에폭 번호
            metrics: 메트릭 딕셔너리 (all_labels, all_preds 포함)
            num_samples: 로깅할 샘플 수
        """
        try:
            # Grad-CAM 생성기 초기화 (마지막 Conv 레이어)
            gradcam = GradCAM(self.model, target_layer='layer4')

            # Validation 데이터에서 샘플 추출
            self.model.eval()

            images_list = []
            heatmaps_list = []
            labels_list = []
            preds_list = []

            sample_count = 0
            for images, labels in self.val_loader:
                if sample_count >= num_samples:
                    break

                images = images.to(self.device)
                labels = labels.to(self.device)

                # 배치에서 필요한 만큼만 추출
                batch_size = min(images.size(0), num_samples - sample_count)

                for i in range(batch_size):
                    img = images[i:i+1]
                    label = labels[i].item()

                    # Grad-CAM 계산
                    heatmap, pred = gradcam(img, target_class=None)

                    # 이미지를 numpy로 변환 (denormalize)
                    img_np = img.squeeze().cpu().numpy()
                    if img_np.ndim == 3:
                        img_np = img_np.transpose(1, 2, 0)  # CHW -> HWC
                    # CT 이미지 denormalize (mean=0.5, std=0.5)
                    img_np = img_np * 0.5 + 0.5
                    img_np = np.clip(img_np, 0, 1)

                    images_list.append(img_np)
                    heatmaps_list.append(heatmap)
                    labels_list.append(label)
                    preds_list.append(pred)

                    sample_count += 1

                if sample_count >= num_samples:
                    break

            # TensorBoard에 로깅
            if images_list:
                self.tb_logger.log_gradcam(
                    epoch,
                    np.array(images_list),
                    np.array(heatmaps_list),
                    np.array(labels_list),
                    np.array(preds_list),
                    self.class_names,
                    num_samples=num_samples // 2
                )

        except Exception as e:
            print(f"⚠️ Grad-CAM 로깅 실패: {e}")

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

    def save_checkpoint(self, epoch: int, metric_value: float, is_best: bool = False):
        """체크포인트 저장 (Top-K 지원)"""
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
            # Top-K 체크포인트 관리
            ckpt_path = self.checkpoint_dir / f'ct_unified_top{len(self.top_k_checkpoints)+1}_epoch{epoch}_{self.timestamp}.pt'
            torch.save(checkpoint, ckpt_path)

            # Top-K 리스트에 추가
            self.top_k_checkpoints.append((metric_value, ckpt_path, epoch))

            # Top-K 정렬 (mode에 따라)
            if self.monitor_mode == 'max':
                self.top_k_checkpoints.sort(key=lambda x: x[0], reverse=True)
            else:
                self.top_k_checkpoints.sort(key=lambda x: x[0])

            # K개 초과 시 가장 낮은 성능 체크포인트 삭제
            while len(self.top_k_checkpoints) > self.save_top_k:
                _, old_path, _ = self.top_k_checkpoints.pop()
                if old_path.exists():
                    old_path.unlink()
                    print(f"  🗑️ Top-K 초과 체크포인트 삭제: {old_path.name}")

            # Best 심볼릭 링크 또는 복사
            best_path = self.checkpoint_dir / f'ct_unified_best_{self.timestamp}.pt'
            if best_path.exists():
                best_path.unlink()
            torch.save(checkpoint, best_path)
            print(f"  ✅ Best 모델 저장: {best_path.name} (Top-{min(len(self.top_k_checkpoints), self.save_top_k)} 유지)")

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

            # TensorBoard 로깅 (enabled 시에만)
            if self.use_tensorboard:
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

                # Grad-CAM 시각화 (설정 시에만, 매 5 에폭마다)
                if self.log_grad_cam and epoch % 5 == 0:
                    self._log_gradcam_samples(epoch, metrics)

            # CSV 로깅
            self._log_to_csv(epoch, train_loss, val_loss, metrics['f1_macro'], metrics['accuracy'], current_lr)

            # Scheduler 업데이트
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    # ReduceLROnPlateau는 모니터링 지표 사용
                    scheduler_metric = self._get_monitor_value(val_loss, metrics)
                    self.scheduler.step(scheduler_metric)
                else:
                    self.scheduler.step()

            # Best 모델 저장 & Early Stopping
            current_metric = self._get_monitor_value(val_loss, metrics)
            is_improvement = self._check_improvement(current_metric)

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
        print(f"✅ 학습 완료!")
        print(f"   Best {self.monitor_metric}: {self.best_metric_value:.4f}")
        print(f"   체크포인트: {self.checkpoint_dir}")
        print(f"{'='*60}")

        if self.use_tensorboard:
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
