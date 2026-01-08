"""평가 지표 계산 (이진 분류 / 다중 분류 지원)"""
import torch
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report
)
import numpy as np
from typing import Dict, List, Optional, Union


def calculate_metrics(
    y_true: Union[np.ndarray, List],
    y_pred: Union[np.ndarray, List],
    y_proba: Optional[np.ndarray] = None,
    num_classes: int = 2,
    class_names: Optional[List[str]] = None
) -> Dict:
    """
    분류 지표 계산 (이진/다중 분류 자동 처리)

    Args:
        y_true: 실제 라벨 (numpy array or list)
        y_pred: 예측 라벨 (numpy array or list)
        y_proba: 예측 확률 (numpy array), 다중분류 시 (N, num_classes) 형태
        num_classes: 클래스 수 (2: 이진분류, 3+: 다중분류)
        class_names: 클래스 이름 리스트 (예: ['normal', 'porosity', 'resin_overflow'])

    Returns:
        dict: 평가 지표들
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 이진 분류 vs 다중 분류
    is_binary = num_classes == 2
    average = 'binary' if is_binary else 'macro'

    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
    }

    # 이진 분류일 때 binary 지표도 추가
    if is_binary:
        metrics['f1'] = f1_score(y_true, y_pred, average='binary', zero_division=0)
        metrics['precision'] = precision_score(y_true, y_pred, average='binary', zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average='binary', zero_division=0)

    # 다중 분류일 때 클래스별 지표 추가
    if not is_binary:
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        # 클래스별 F1
        per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
        if class_names and len(class_names) == len(per_class_f1):
            for i, name in enumerate(class_names):
                metrics[f'f1_{name}'] = per_class_f1[i]
        else:
            for i, f1 in enumerate(per_class_f1):
                metrics[f'f1_class_{i}'] = f1

    # Confusion Matrix
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()

    # ROC-AUC (확률이 있을 경우)
    if y_proba is not None:
        try:
            if is_binary:
                # 이진 분류: 양성 클래스 확률만 사용
                proba = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
                metrics['roc_auc'] = roc_auc_score(y_true, proba)
            else:
                # 다중 분류: One-vs-Rest AUC
                metrics['roc_auc_ovr'] = roc_auc_score(
                    y_true, y_proba, multi_class='ovr', average='macro'
                )
        except Exception as e:
            pass  # 클래스가 부족하면 계산 불가

    return metrics


def calculate_metrics_binary(y_true, y_pred, y_proba=None):
    """이진 분류 지표 계산 (하위 호환성)"""
    return calculate_metrics(y_true, y_pred, y_proba, num_classes=2)


def print_metrics(metrics: Dict, prefix: str = "", class_names: Optional[List[str]] = None):
    """지표 출력"""
    print(f"{prefix}Metrics:")
    print(f"  - Accuracy:       {metrics['accuracy']:.4f}")
    print(f"  - F1 (Macro):     {metrics['f1_macro']:.4f}")
    print(f"  - Precision:      {metrics['precision_macro']:.4f}")
    print(f"  - Recall:         {metrics['recall_macro']:.4f}")

    # 이진 분류 지표
    if 'f1' in metrics:
        print(f"  - F1 (Binary):    {metrics['f1']:.4f}")

    # 다중 분류 지표
    if 'f1_weighted' in metrics:
        print(f"  - F1 (Weighted):  {metrics['f1_weighted']:.4f}")

    # ROC-AUC
    if 'roc_auc' in metrics:
        print(f"  - ROC-AUC:        {metrics['roc_auc']:.4f}")
    if 'roc_auc_ovr' in metrics:
        print(f"  - ROC-AUC (OvR):  {metrics['roc_auc_ovr']:.4f}")

    # 클래스별 F1
    if class_names:
        print(f"  - Per-class F1:")
        for name in class_names:
            key = f'f1_{name}'
            if key in metrics:
                print(f"      {name}: {metrics[key]:.4f}")


def get_classification_report(
    y_true: Union[np.ndarray, List],
    y_pred: Union[np.ndarray, List],
    class_names: Optional[List[str]] = None
) -> str:
    """sklearn classification_report 반환"""
    return classification_report(
        y_true, y_pred,
        target_names=class_names,
        zero_division=0
    )
