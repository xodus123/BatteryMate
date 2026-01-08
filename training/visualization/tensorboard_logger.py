"""TensorBoard Logger - 다중분류 학습 시각화"""
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from datetime import datetime
import torch
import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image


class TensorBoardLogger:
    """TensorBoard 로깅 (다중분류 지원)"""

    def __init__(self, config: dict):
        """
        Args:
            config: YAML config dict
        """
        if not config['logging']['tensorboard'].get('enabled', True):
            self.writer = None
            return

        # Run 이름 생성
        run_name = self._generate_run_name(config)

        # Log 디렉토리
        log_dir = Path(config['logging']['tensorboard']['log_dir']) / run_name
        log_dir.mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(log_dir=str(log_dir))
        print(f"✅ TensorBoard 로거 초기화: {log_dir}")

    def _generate_run_name(self, config: dict) -> str:
        """실험 조건 기반 run 이름 생성"""
        model_name = config['model']['name']
        lr = config['training']['lr']
        bs = config['data']['batch_size']
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"{model_name}_lr{lr}_bs{bs}_{timestamp}"

    def log_scalars(self, epoch: int, metrics: dict):
        """스칼라 로깅 (딕셔너리 형태)"""
        if self.writer is None:
            return

        for name, value in metrics.items():
            self.writer.add_scalar(name, value, epoch)

    def log_confusion_matrix(self, epoch: int, cm: np.ndarray, class_names: list, tag: str = 'val'):
        """Confusion Matrix 이미지 로깅"""
        if self.writer is None:
            return

        try:
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            ax.figure.colorbar(im, ax=ax)

            ax.set(
                xticks=np.arange(cm.shape[1]),
                yticks=np.arange(cm.shape[0]),
                xticklabels=class_names,
                yticklabels=class_names,
                title=f'Confusion Matrix (Epoch {epoch})',
                ylabel='True Label',
                xlabel='Predicted Label'
            )

            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

            # 셀에 숫자 표시
            thresh = cm.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, format(cm[i, j], 'd'),
                           ha="center", va="center",
                           color="white" if cm[i, j] > thresh else "black")

            fig.tight_layout()

            # 이미지로 변환
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            image = Image.open(buf)
            image_array = np.array(image)

            # TensorBoard에 이미지 추가
            self.writer.add_image(f'ConfusionMatrix/{tag}', image_array, epoch, dataformats='HWC')

            plt.close(fig)
            buf.close()

        except Exception as e:
            print(f"⚠️ Confusion Matrix 로깅 실패: {e}")

    def log_per_class_metrics(self, epoch: int, class_metrics: dict, class_names: list):
        """클래스별 메트릭 로깅"""
        if self.writer is None:
            return

        for metric_name, values in class_metrics.items():
            for i, class_name in enumerate(class_names):
                if i < len(values):
                    self.writer.add_scalar(
                        f'PerClass/{metric_name}/{class_name}',
                        values[i],
                        epoch
                    )

    def log_classification_errors(self, epoch: int, cm: np.ndarray, class_names: list):
        """클래스별 TP/FP/FN/TN 로깅"""
        if self.writer is None:
            return

        num_classes = len(class_names)

        for i, class_name in enumerate(class_names):
            # TP: 대각선 (정확히 분류)
            tp = cm[i, i]
            # FN: 해당 행의 다른 열 합 (실제 i인데 다른 것으로 예측)
            fn = cm[i, :].sum() - tp
            # FP: 해당 열의 다른 행 합 (다른 것인데 i로 예측)
            fp = cm[:, i].sum() - tp
            # TN: 나머지
            tn = cm.sum() - tp - fn - fp

            # Precision, Recall 계산
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0

            self.writer.add_scalar(f'Errors/{class_name}/TP', tp, epoch)
            self.writer.add_scalar(f'Errors/{class_name}/FP', fp, epoch)
            self.writer.add_scalar(f'Errors/{class_name}/FN', fn, epoch)
            self.writer.add_scalar(f'Errors/{class_name}/Precision', precision, epoch)
            self.writer.add_scalar(f'Errors/{class_name}/Recall', recall, epoch)

    def log_error_summary_table(self, epoch: int, cm: np.ndarray, class_names: list):
        """FP/FN 요약 테이블 이미지로 로깅"""
        if self.writer is None:
            return

        try:
            num_classes = len(class_names)

            # 데이터 계산
            data = []
            for i, class_name in enumerate(class_names):
                tp = cm[i, i]
                fn = cm[i, :].sum() - tp
                fp = cm[:, i].sum() - tp
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                data.append([class_name[:15], int(tp), int(fp), int(fn), f'{precision:.3f}', f'{recall:.3f}', f'{f1:.3f}'])

            # 테이블 생성
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.axis('off')

            table = ax.table(
                cellText=data,
                colLabels=['Class', 'TP', 'FP', 'FN', 'Precision', 'Recall', 'F1'],
                cellLoc='center',
                loc='center',
                colWidths=[0.22, 0.1, 0.1, 0.1, 0.12, 0.12, 0.12]
            )
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.5)

            # 헤더 스타일
            for j in range(7):
                table[(0, j)].set_facecolor('#4472C4')
                table[(0, j)].set_text_props(color='white', weight='bold')

            # FP/FN 강조 (값이 높으면 빨간색)
            for i in range(1, num_classes + 1):
                fp_val = int(data[i-1][2])
                fn_val = int(data[i-1][3])
                if fp_val > 100:
                    table[(i, 2)].set_facecolor('#FFE0E0')
                if fn_val > 100:
                    table[(i, 3)].set_facecolor('#FFE0E0')

            plt.title(f'Classification Errors (Epoch {epoch})', fontsize=12, fontweight='bold', pad=20)
            plt.tight_layout()

            # 이미지로 변환
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            image = Image.open(buf)
            image_array = np.array(image)

            self.writer.add_image('ErrorSummary/table', image_array, epoch, dataformats='HWC')

            plt.close(fig)
            buf.close()

        except Exception as e:
            print(f"⚠️ Error Summary 로깅 실패: {e}")

    def log_model_graph(self, model, input_sample):
        """모델 구조 그래프 로깅"""
        if self.writer is None:
            return

        try:
            self.writer.add_graph(model, input_sample)
            print(f"✅ 모델 구조 그래프가 TensorBoard에 추가되었습니다.")
        except Exception as e:
            print(f"⚠️ 모델 그래프 로깅 실패: {e}")

    def log_pr_curves(self, epoch: int, labels: np.ndarray, probs: np.ndarray, class_names: list):
        """클래스별 PR Curve 로깅"""
        if self.writer is None:
            return

        try:
            num_classes = len(class_names)

            for i, class_name in enumerate(class_names):
                # One-vs-Rest 방식: 해당 클래스인지 아닌지 (binary)
                binary_labels = (labels == i).astype(int)
                class_probs = probs[:, i]

                self.writer.add_pr_curve(
                    f'PR_Curve/{class_name}',
                    binary_labels,
                    class_probs,
                    global_step=epoch
                )

        except Exception as e:
            print(f"⚠️ PR Curve 로깅 실패: {e}")

    def log_class_distribution(self, epoch: int, labels: np.ndarray, class_names: list, tag: str = 'val'):
        """클래스 분포 시각화 (데이터셋 불균형 확인)"""
        if self.writer is None:
            return

        try:
            num_classes = len(class_names)
            class_counts = np.bincount(labels, minlength=num_classes)

            # 막대 그래프 생성
            fig, ax = plt.subplots(figsize=(10, 6))

            colors = plt.cm.tab10(np.linspace(0, 1, num_classes))
            bars = ax.bar(range(num_classes), class_counts, color=colors)

            ax.set_xlabel('Class')
            ax.set_ylabel('Count')
            ax.set_title(f'Class Distribution - {tag} (Epoch {epoch})')
            ax.set_xticks(range(num_classes))
            ax.set_xticklabels(class_names, rotation=45, ha='right')

            # 각 막대 위에 숫자 표시
            for bar, count in zip(bars, class_counts):
                height = bar.get_height()
                ax.annotate(f'{count:,}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)

            # 비율 표시
            total = class_counts.sum()
            for i, (bar, count) in enumerate(zip(bars, class_counts)):
                pct = count / total * 100
                ax.annotate(f'({pct:.1f}%)',
                           xy=(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2),
                           ha='center', va='center', fontsize=8, color='white', fontweight='bold')

            plt.tight_layout()

            # 이미지로 변환
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            image = Image.open(buf)
            image_array = np.array(image)

            self.writer.add_image(f'Distribution/{tag}', image_array, epoch, dataformats='HWC')

            plt.close(fig)
            buf.close()

        except Exception as e:
            print(f"⚠️ Class Distribution 로깅 실패: {e}")

    def log_probability_histograms(self, epoch: int, probs: np.ndarray, labels: np.ndarray, class_names: list):
        """클래스별 예측 확률 히스토그램 로깅"""
        if self.writer is None:
            return

        try:
            num_classes = len(class_names)

            for i, class_name in enumerate(class_names):
                # 해당 클래스의 예측 확률 분포
                class_probs = probs[:, i]

                # 전체 샘플에 대한 해당 클래스 확률 분포
                self.writer.add_histogram(
                    f'Probabilities/{class_name}/all',
                    class_probs,
                    global_step=epoch
                )

                # 실제 해당 클래스인 샘플들의 확률 분포 (TP가 되어야 할 샘플들)
                true_class_mask = labels == i
                if true_class_mask.sum() > 0:
                    self.writer.add_histogram(
                        f'Probabilities/{class_name}/true_samples',
                        class_probs[true_class_mask],
                        global_step=epoch
                    )

                # 실제 다른 클래스인 샘플들의 확률 분포 (FP가 될 수 있는 샘플들)
                false_class_mask = labels != i
                if false_class_mask.sum() > 0:
                    self.writer.add_histogram(
                        f'Probabilities/{class_name}/false_samples',
                        class_probs[false_class_mask],
                        global_step=epoch
                    )

        except Exception as e:
            print(f"⚠️ Probability Histogram 로깅 실패: {e}")

    def log_prediction_confidence(self, epoch: int, probs: np.ndarray, preds: np.ndarray, labels: np.ndarray):
        """예측 신뢰도 분포 (정답/오답 비교)"""
        if self.writer is None:
            return

        try:
            # 최대 확률값 (예측 신뢰도)
            max_probs = probs.max(axis=1)

            # 정답인 경우와 오답인 경우 분리
            correct_mask = preds == labels
            incorrect_mask = ~correct_mask

            # 정답 예측의 신뢰도 분포
            if correct_mask.sum() > 0:
                self.writer.add_histogram(
                    'Confidence/correct_predictions',
                    max_probs[correct_mask],
                    global_step=epoch
                )

            # 오답 예측의 신뢰도 분포
            if incorrect_mask.sum() > 0:
                self.writer.add_histogram(
                    'Confidence/incorrect_predictions',
                    max_probs[incorrect_mask],
                    global_step=epoch
                )

            # 전체 신뢰도 분포
            self.writer.add_histogram(
                'Confidence/all_predictions',
                max_probs,
                global_step=epoch
            )

        except Exception as e:
            print(f"⚠️ Confidence Histogram 로깅 실패: {e}")

    def close(self):
        """TensorBoard writer 종료"""
        if self.writer:
            self.writer.close()
