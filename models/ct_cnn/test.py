"""CT CNN 테스트 스크립트 (5클래스 다중분류)

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

import torch
import torch.nn as nn
from pathlib import Path
import numpy as np
from tqdm import tqdm
import argparse
import json
import csv
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    confusion_matrix, classification_report, roc_auc_score
)

from models.ct_cnn.model import create_model
from training.configs.config_loader import ConfigLoader
from training.data.dataloader import create_test_dataloader
from training.visualization.tensorboard_logger import TensorBoardLogger


class CNNTester:
    """CNN 테스터 (5클래스 다중분류)"""

    def __init__(self, checkpoint_path: str, config: dict = None, enable_tensorboard: bool = True):
        """
        Args:
            checkpoint_path: 체크포인트 파일 경로
            config: YAML config dict (None이면 checkpoint에서 로드)
            enable_tensorboard: TensorBoard 로깅 활성화 여부
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.checkpoint_path = checkpoint_path
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 체크포인트 로드
        print(f"✅ 체크포인트 로딩: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # Config 로드 (checkpoint 우선, 없으면 외부에서 받음)
        self.config = checkpoint.get('config', config)
        if self.config is None:
            raise ValueError("Config를 찾을 수 없습니다. checkpoint에 config가 없거나 외부 config를 제공하지 않았습니다.")

        # 클래스 정보
        self.class_names = self.config.get('classes', {}).get('names',
            ['cell_normal', 'cell_porosity', 'module_normal', 'module_porosity', 'module_resin_overflow'])
        self.num_classes = len(self.class_names)

        # 모델 생성 및 가중치 로드
        self.model = create_model(self.config).to(self.device)

        # 이전 체크포인트 호환성 처리 (01-20 이전: model.fc.weight → model.fc.1.weight)
        state_dict = checkpoint['model_state_dict']
        if 'model.fc.weight' in state_dict and 'model.fc.1.weight' not in state_dict:
            print("⚠️ 구 형식 체크포인트 감지 → state_dict 키 변환 (model.fc → model.fc.1)")
            state_dict['model.fc.1.weight'] = state_dict.pop('model.fc.weight')
            state_dict['model.fc.1.bias'] = state_dict.pop('model.fc.bias')

        self.model.load_state_dict(state_dict)
        self.model.eval()

        # Loss function (다중분류)
        self.criterion = nn.CrossEntropyLoss()

        # Test DataLoader
        # Augmentation config 가져오기 (Test는 augmentation 없이 사용)
        augmentation_config = self.config['data'].get('augmentation', None)

        self.test_loader = create_test_dataloader(
            test_split_file=self.config['data']['test_split'],
            batch_size=self.config['data']['batch_size'],
            num_workers=self.config['data']['num_workers'],
            image_size=self.config['data']['image_size'],
            modality='ct',
            preprocessed=self.config['data'].get('preprocessed', False),
            use_albumentations=self.config['data'].get('use_albumentations', False),
            augmentation_config=augmentation_config
        )

        # 결과 저장 디렉토리
        self.results_dir = Path('models/ct_cnn/results')
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.log_dir = Path('models/ct_cnn/logs')
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # TensorBoard Writer 설정
        self.writer = None
        self.tb_log_dir = None
        if enable_tensorboard:
            checkpoint_name = Path(checkpoint_path).stem
            self.tb_log_dir = self.log_dir / f'test_{checkpoint_name}_{self.timestamp}'
            self.tb_log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=str(self.tb_log_dir))
            print(f"✅ TensorBoard 로그 디렉토리: {self.tb_log_dir}")

        print(f"✅ 모델 로드 완료")
        print(f"   - Epoch: {checkpoint.get('epoch', 'N/A')}")
        val_f1 = checkpoint.get('val_f1_macro', checkpoint.get('val_f1', 'N/A'))
        print(f"   - Val F1: {val_f1:.4f}" if isinstance(val_f1, float) else f"   - Val F1: {val_f1}")
        print(f"   - Num Classes: {self.num_classes}")
        print(f"   - Classes: {self.class_names}")
        print(f"   - Device: {self.device}")
        print(f"   - Test 데이터: {len(self.test_loader.dataset)}개\n")

    def test(self) -> dict:
        """Test 데이터 평가"""
        print(f"{'='*60}")
        print(f"Test 데이터 평가 시작 (5클래스 다중분류)")
        print(f"{'='*60}\n")

        total_loss = 0.0
        num_batches = 0

        all_labels = []
        all_preds = []
        all_probs = []

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(tqdm(self.test_loader, desc="Testing")):
                images = images.to(self.device)
                labels_tensor = labels.to(self.device).long()

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
                all_probs.append(probs.cpu().numpy())

        avg_loss = total_loss / num_batches
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        all_probs = np.vstack(all_probs)  # (N, num_classes)

        # Metrics 계산
        metrics = self._calculate_metrics(all_labels, all_preds, all_probs)
        metrics['loss'] = avg_loss

        # 결과 출력
        self._print_results(metrics, all_labels, all_preds)

        # TensorBoard 로깅
        if self.writer is not None:
            self._log_to_tensorboard(metrics, all_labels, all_preds, all_probs)

        # 결과 파일 저장
        self._save_results(metrics, all_labels, all_preds, all_probs)

        return {
            'metrics': metrics,
            'predictions': {
                'labels': all_labels,
                'preds': all_preds,
                'probs': all_probs
            }
        }

    def _calculate_metrics(self, labels: np.ndarray, preds: np.ndarray, probs: np.ndarray) -> dict:
        """메트릭 계산 (5클래스 다중분류)"""
        # 전체 메트릭
        accuracy = accuracy_score(labels, preds)
        f1_macro = f1_score(labels, preds, average='macro', zero_division=0)
        f1_weighted = f1_score(labels, preds, average='weighted', zero_division=0)
        precision_macro = precision_score(labels, preds, average='macro', zero_division=0)
        recall_macro = recall_score(labels, preds, average='macro', zero_division=0)

        # 클래스별 메트릭
        f1_per_class = f1_score(labels, preds, average=None, zero_division=0)
        precision_per_class = precision_score(labels, preds, average=None, zero_division=0)
        recall_per_class = recall_score(labels, preds, average=None, zero_division=0)

        # Confusion Matrix
        cm = confusion_matrix(labels, preds, labels=range(self.num_classes))

        # ROC-AUC (One-vs-Rest, macro)
        try:
            roc_auc_ovr = roc_auc_score(labels, probs, multi_class='ovr', average='macro')
        except Exception:
            roc_auc_ovr = None

        # 클래스별 샘플 수
        class_counts = np.bincount(labels, minlength=self.num_classes)

        return {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'roc_auc_ovr': roc_auc_ovr,
            'f1_per_class': f1_per_class.tolist(),
            'precision_per_class': precision_per_class.tolist(),
            'recall_per_class': recall_per_class.tolist(),
            'confusion_matrix': cm.tolist(),
            'class_counts': class_counts.tolist(),
            'total_samples': len(labels)
        }

    def _print_results(self, metrics: dict, labels: np.ndarray, preds: np.ndarray):
        """결과 출력"""
        print(f"\n{'='*60}")
        print(f"Test 결과")
        print(f"{'='*60}")
        print(f"  Test Loss: {metrics['loss']:.4f}")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1 (macro): {metrics['f1_macro']:.4f}")
        print(f"  F1 (weighted): {metrics['f1_weighted']:.4f}")
        print(f"  Precision (macro): {metrics['precision_macro']:.4f}")
        print(f"  Recall (macro): {metrics['recall_macro']:.4f}")
        if metrics['roc_auc_ovr'] is not None:
            print(f"  ROC-AUC (OvR macro): {metrics['roc_auc_ovr']:.4f}")

        print(f"\n  클래스별 성능:")
        print("-" * 60)
        report = classification_report(
            labels, preds,
            labels=list(range(len(self.class_names))),
            target_names=self.class_names,
            zero_division=0
        )
        print(report)

        print(f"\n  클래스별 샘플 수:")
        for i, (name, count) in enumerate(zip(self.class_names, metrics['class_counts'])):
            print(f"    {i}: {name}: {count}")

        print(f"{'='*60}\n")

    def _log_to_tensorboard(self, metrics: dict, labels: np.ndarray, preds: np.ndarray, probs: np.ndarray):
        """TensorBoard에 테스트 결과 로깅 (5클래스 다중분류)"""
        import matplotlib.pyplot as plt
        import io
        from PIL import Image

        print("📊 TensorBoard 로깅 중...")

        # 1. Scalars - 기본 지표
        self.writer.add_scalar('Test/Loss', metrics['loss'], 0)
        self.writer.add_scalar('Test/Accuracy', metrics['accuracy'], 0)
        self.writer.add_scalar('Test/F1_macro', metrics['f1_macro'], 0)
        self.writer.add_scalar('Test/F1_weighted', metrics['f1_weighted'], 0)
        self.writer.add_scalar('Test/Precision_macro', metrics['precision_macro'], 0)
        self.writer.add_scalar('Test/Recall_macro', metrics['recall_macro'], 0)
        if metrics['roc_auc_ovr'] is not None:
            self.writer.add_scalar('Test/ROC_AUC_OvR', metrics['roc_auc_ovr'], 0)

        # 2. 클래스별 메트릭
        for i, class_name in enumerate(self.class_names):
            self.writer.add_scalar(f'Test/PerClass/F1/{class_name}', metrics['f1_per_class'][i], 0)
            self.writer.add_scalar(f'Test/PerClass/Precision/{class_name}', metrics['precision_per_class'][i], 0)
            self.writer.add_scalar(f'Test/PerClass/Recall/{class_name}', metrics['recall_per_class'][i], 0)

        # 3. PR Curve (클래스별 One-vs-Rest)
        for i, class_name in enumerate(self.class_names):
            binary_labels = (labels == i).astype(int)
            class_probs = probs[:, i]
            self.writer.add_pr_curve(
                f'Test/PR_Curve/{class_name}',
                binary_labels,
                class_probs,
                global_step=0
            )

        # 4. Confusion Matrix (5x5)
        cm = np.array(metrics['confusion_matrix'])
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
            title='Confusion Matrix (Test Set) - 5 Classes'
        )
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # 숫자 표시
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black")

        fig.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        buf.seek(0)
        cm_image = Image.open(buf)
        cm_array = np.array(cm_image)
        self.writer.add_image('Test/Confusion_Matrix', cm_array, 0, dataformats='HWC')
        plt.close(fig)

        # 5. 클래스별 TP/FP/FN 테이블
        self._log_error_summary_table(cm)

        # 6. 클래스별 확률 히스토그램
        for i, class_name in enumerate(self.class_names):
            class_probs = probs[:, i]

            # 전체 샘플에 대한 해당 클래스 확률
            self.writer.add_histogram(f'Test/Probabilities/{class_name}/all', class_probs, 0)

            # 실제 해당 클래스인 샘플의 확률
            true_mask = labels == i
            if true_mask.sum() > 0:
                self.writer.add_histogram(f'Test/Probabilities/{class_name}/true_samples', class_probs[true_mask], 0)

            # 실제 다른 클래스인 샘플의 확률
            false_mask = labels != i
            if false_mask.sum() > 0:
                self.writer.add_histogram(f'Test/Probabilities/{class_name}/false_samples', class_probs[false_mask], 0)

        # 7. 예측 신뢰도 분포 (정답/오답)
        max_probs = probs.max(axis=1)
        correct_mask = preds == labels

        if correct_mask.sum() > 0:
            self.writer.add_histogram('Test/Confidence/correct', max_probs[correct_mask], 0)
        if (~correct_mask).sum() > 0:
            self.writer.add_histogram('Test/Confidence/incorrect', max_probs[~correct_mask], 0)
        self.writer.add_histogram('Test/Confidence/all', max_probs, 0)

        # 8. 클래스 분포 시각화
        self._log_class_distribution(labels)

        self.writer.flush()
        print("✅ TensorBoard 로깅 완료")

    def _log_error_summary_table(self, cm: np.ndarray):
        """FP/FN 요약 테이블 이미지로 로깅"""
        import matplotlib.pyplot as plt
        import io
        from PIL import Image

        try:
            data = []
            for i, class_name in enumerate(self.class_names):
                tp = cm[i, i]
                fn = cm[i, :].sum() - tp
                fp = cm[:, i].sum() - tp
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                data.append([class_name[:20], int(tp), int(fp), int(fn), f'{precision:.3f}', f'{recall:.3f}', f'{f1:.3f}'])

            fig, ax = plt.subplots(figsize=(14, 5))
            ax.axis('off')

            table = ax.table(
                cellText=data,
                colLabels=['Class', 'TP', 'FP', 'FN', 'Precision', 'Recall', 'F1'],
                cellLoc='center',
                loc='center',
                colWidths=[0.28, 0.1, 0.1, 0.1, 0.12, 0.12, 0.12]
            )
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.8)

            for j in range(7):
                table[(0, j)].set_facecolor('#4472C4')
                table[(0, j)].set_text_props(color='white', weight='bold')

            plt.title('Test Set - Classification Errors by Class', fontsize=12, fontweight='bold', pad=20)
            plt.tight_layout()

            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            image = Image.open(buf)
            image_array = np.array(image)

            self.writer.add_image('Test/Error_Summary_Table', image_array, 0, dataformats='HWC')
            plt.close(fig)

        except Exception as e:
            print(f"⚠️ Error Summary Table 로깅 실패: {e}")

    def _log_class_distribution(self, labels: np.ndarray):
        """클래스 분포 시각화"""
        import matplotlib.pyplot as plt
        import io
        from PIL import Image

        try:
            class_counts = np.bincount(labels, minlength=self.num_classes)

            fig, ax = plt.subplots(figsize=(12, 6))
            colors = plt.cm.tab10(np.linspace(0, 1, self.num_classes))
            bars = ax.bar(range(self.num_classes), class_counts, color=colors)

            ax.set_xlabel('Class')
            ax.set_ylabel('Count')
            ax.set_title('Test Set - Class Distribution')
            ax.set_xticks(range(self.num_classes))
            ax.set_xticklabels(self.class_names, rotation=45, ha='right')

            total = class_counts.sum()
            for bar, count in zip(bars, class_counts):
                height = bar.get_height()
                ax.annotate(f'{count:,}\n({count/total*100:.1f}%)',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)

            plt.tight_layout()

            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            image = Image.open(buf)
            image_array = np.array(image)

            self.writer.add_image('Test/Class_Distribution', image_array, 0, dataformats='HWC')
            plt.close(fig)

        except Exception as e:
            print(f"⚠️ Class Distribution 로깅 실패: {e}")

    def _save_results(self, metrics: dict, labels: np.ndarray, preds: np.ndarray, probs: np.ndarray):
        """결과를 JSON과 CSV 파일로 저장"""
        checkpoint_name = Path(self.checkpoint_path).stem

        # 1. JSON 파일 저장 (상세 결과)
        json_path = self.results_dir / f'test_{checkpoint_name}_{self.timestamp}.json'

        json_data = {
            'checkpoint': self.checkpoint_path,
            'timestamp': self.timestamp,
            'num_classes': self.num_classes,
            'class_names': self.class_names,
            'total_samples': metrics['total_samples'],
            'class_counts': dict(zip(self.class_names, metrics['class_counts'])),
            'metrics': {
                'loss': metrics['loss'],
                'accuracy': metrics['accuracy'],
                'f1_macro': metrics['f1_macro'],
                'f1_weighted': metrics['f1_weighted'],
                'precision_macro': metrics['precision_macro'],
                'recall_macro': metrics['recall_macro'],
                'roc_auc_ovr': metrics['roc_auc_ovr'],
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
        print(f"✅ JSON 결과 저장: {json_path}")

        # 2. CSV 파일 저장 (요약)
        csv_path = self.log_dir / f'test_{checkpoint_name}_{self.timestamp}.csv'

        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # 헤더
            header = ['metric', 'value']
            writer.writerow(header)

            # 전체 메트릭
            writer.writerow(['checkpoint', self.checkpoint_path])
            writer.writerow(['timestamp', self.timestamp])
            writer.writerow(['total_samples', metrics['total_samples']])
            writer.writerow(['loss', f"{metrics['loss']:.4f}"])
            writer.writerow(['accuracy', f"{metrics['accuracy']:.4f}"])
            writer.writerow(['f1_macro', f"{metrics['f1_macro']:.4f}"])
            writer.writerow(['f1_weighted', f"{metrics['f1_weighted']:.4f}"])
            writer.writerow(['precision_macro', f"{metrics['precision_macro']:.4f}"])
            writer.writerow(['recall_macro', f"{metrics['recall_macro']:.4f}"])
            if metrics['roc_auc_ovr'] is not None:
                writer.writerow(['roc_auc_ovr', f"{metrics['roc_auc_ovr']:.4f}"])

            writer.writerow([])  # 빈 줄
            writer.writerow(['--- Per Class Metrics ---', ''])

            # 클래스별 메트릭
            for i, name in enumerate(self.class_names):
                writer.writerow([f'{name}_f1', f"{metrics['f1_per_class'][i]:.4f}"])
                writer.writerow([f'{name}_precision', f"{metrics['precision_per_class'][i]:.4f}"])
                writer.writerow([f'{name}_recall', f"{metrics['recall_per_class'][i]:.4f}"])
                writer.writerow([f'{name}_support', metrics['class_counts'][i]])

        print(f"✅ CSV 결과 저장: {csv_path}")

        # 3. Confusion Matrix 이미지 저장
        self._save_confusion_matrix_image(metrics['confusion_matrix'], checkpoint_name)

    def _save_confusion_matrix_image(self, cm: list, checkpoint_name: str):
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
            title=f'Confusion Matrix - {checkpoint_name}'
        )
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black")

        fig.tight_layout()

        img_path = self.results_dir / f'confusion_matrix_{checkpoint_name}_{self.timestamp}.png'
        plt.savefig(img_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"✅ Confusion Matrix 이미지 저장: {img_path}")

    def close(self):
        """TensorBoard Writer 닫기"""
        if self.writer is not None:
            self.writer.close()


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='CT CNN Test (5-class classification)')
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='체크포인트 파일 경로 (예: models/ct_cnn/checkpoints/ct_unified_best_20260105_140553.pt)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Config 파일 경로 (생략하면 checkpoint에서 로드)'
    )
    parser.add_argument(
        '--no-tensorboard',
        action='store_true',
        help='TensorBoard 로깅 비활성화'
    )

    args = parser.parse_args()

    # Config 로드 (선택사항)
    config = None
    if args.config:
        config = ConfigLoader.load(args.config)

    # Tester 생성 및 평가
    tester = CNNTester(
        checkpoint_path=args.checkpoint,
        config=config,
        enable_tensorboard=not args.no_tensorboard
    )
    results = tester.test()

    # 추가 분석
    print("\n📊 상세 분석:")
    preds_data = results['predictions']
    labels = preds_data['labels']
    preds = preds_data['preds']
    probs = preds_data['probs']

    # 클래스별 오답 분석
    print("\n  클래스별 오분류 분석:")
    for i, class_name in enumerate(tester.class_names):
        # 해당 클래스를 다른 것으로 잘못 예측 (FN)
        fn_mask = (labels == i) & (preds != i)
        fn_count = fn_mask.sum()

        # 다른 것을 해당 클래스로 잘못 예측 (FP)
        fp_mask = (labels != i) & (preds == i)
        fp_count = fp_mask.sum()

        print(f"    {class_name}:")
        print(f"      - FN (놓친 것): {fn_count}개")
        print(f"      - FP (잘못 예측): {fp_count}개")

    # 예측 신뢰도 분석
    max_probs = probs.max(axis=1)
    correct_mask = preds == labels

    print(f"\n  예측 신뢰도 분석:")
    print(f"    정답 예측 신뢰도: {max_probs[correct_mask].mean():.4f} (std: {max_probs[correct_mask].std():.4f})")
    if (~correct_mask).sum() > 0:
        print(f"    오답 예측 신뢰도: {max_probs[~correct_mask].mean():.4f} (std: {max_probs[~correct_mask].std():.4f})")

    # TensorBoard Writer 닫기
    tester.close()

    print(f"\n{'='*60}")
    print(f"✅ 평가 완료!")
    print(f"   - TensorBoard: tensorboard --logdir={tester.tb_log_dir}")
    print(f"   - 결과 파일: {tester.results_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
