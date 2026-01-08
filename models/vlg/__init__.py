"""VLG (Vision Language Grounding) - 결함 위치 탐지 모듈

지원 모델:
- GroundingDINO: 고정밀 오픈 어휘 탐지 (기본)
- YOLO-World: 빠른 오픈 어휘 탐지
"""
from .inference import VLGInference
from .inference_yoloworld import YOLOWorldInference
from .prompts import GroundingPrompts


def create_vlg(model_type: str = 'groundingdino', **kwargs):
    """
    VLG 모델 팩토리 함수

    Args:
        model_type: 'groundingdino' 또는 'yoloworld'
        **kwargs: 모델별 추가 인자

    Returns:
        VLG 인스턴스
    """
    if model_type.lower() in ['groundingdino', 'gdino', 'dino']:
        return VLGInference(**kwargs)
    elif model_type.lower() in ['yoloworld', 'yolo-world', 'yolo']:
        return YOLOWorldInference(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'groundingdino' or 'yoloworld'")


__all__ = [
    'VLGInference',
    'YOLOWorldInference',
    'GroundingPrompts',
    'create_vlg',
]
