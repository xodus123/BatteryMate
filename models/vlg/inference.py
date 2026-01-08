"""GroundingDINO 기반 VLG 추론 모듈"""
import torch
import numpy as np
from pathlib import Path
from typing import Union, Dict, Any, List, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont
from dataclasses import dataclass

from .prompts import GroundingPrompts, UNIFIED_CLASSES


@dataclass
class DetectionResult:
    """결함 탐지 결과"""
    boxes: List[List[float]]  # [[x1, y1, x2, y2], ...]
    labels: List[str]  # 탐지된 결함 라벨
    scores: List[float]  # 신뢰도 점수
    phrases: List[str]  # 매칭된 텍스트 구문


class VLGInference:
    """GroundingDINO를 이용한 결함 위치 탐지"""

    # 사용 가능한 모델
    MODEL_CONFIGS = {
        'swinT': {
            'config': None,  # 패키지 내 config 사용
            'weights': 'models/vlg/weights/groundingdino_swint_ogc.pth',
        },
        'swinB': {
            'config': None,
            'weights': 'models/vlg/weights/groundingdino_swinb_cogcoor.pth',
        },
    }

    def __init__(
        self,
        model_type: str = 'swinT',
        device: str = 'cuda',
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
        weights_path: Optional[str] = None,
    ):
        """
        VLG 모델 초기화

        Args:
            model_type: 모델 타입 ('swinT' 또는 'swinB')
            device: 실행 디바이스
            box_threshold: 바운딩 박스 신뢰도 임계값
            text_threshold: 텍스트 매칭 임계값
            weights_path: 커스텀 가중치 경로
        """
        self.device = device
        self.model_type = model_type
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

        # 프롬프트 템플릿
        self.prompts = GroundingPrompts()

        # 모델 로드
        self.model = self._load_model(model_type, weights_path)

        print(f"VLG model ({model_type}) loaded successfully on {device}")

    def _load_model(self, model_type: str, weights_path: Optional[str] = None):
        """
        GroundingDINO 모델 로드

        Args:
            model_type: 모델 타입
            weights_path: 커스텀 가중치 경로

        Returns:
            로드된 모델
        """
        try:
            from groundingdino.util.inference import load_model
            import groundingdino

            config = self.MODEL_CONFIGS[model_type]

            # 가중치 경로 결정
            if weights_path is None:
                weights_path = config['weights']

            # Config 경로 결정 (패키지 내 config 사용)
            package_dir = Path(groundingdino.__file__).parent
            if model_type == 'swinT':
                config_path = str(package_dir / 'config' / 'GroundingDINO_SwinT_OGC.py')
            else:
                config_path = str(package_dir / 'config' / 'GroundingDINO_SwinB_cfg.py')

            # 모델 로드
            model = load_model(config_path, weights_path, device=self.device)

            return model

        except ImportError:
            print("GroundingDINO not installed. Using mock model for testing.")
            return None
        except Exception as e:
            print(f"Error loading GroundingDINO: {e}")
            return None

    def detect(
        self,
        image: Union[str, Path, Image.Image],
        text_prompt: Optional[str] = None,
        modality: str = 'ct',
        defect_type: str = 'all',
    ) -> DetectionResult:
        """
        이미지에서 결함 탐지

        Args:
            image: 이미지 경로 또는 PIL Image
            text_prompt: 커스텀 텍스트 프롬프트 (없으면 기본 프롬프트 사용)
            modality: 이미지 모달리티 ('ct' 또는 'rgb')
            defect_type: 결함 유형

        Returns:
            DetectionResult 객체
        """
        # 이미지 로드
        if isinstance(image, (str, Path)):
            image_pil = Image.open(image).convert('RGB')
            image_path = str(image)
        else:
            image_pil = image.convert('RGB')
            image_path = None

        # 텍스트 프롬프트 준비
        if text_prompt is None:
            prompts = self.prompts.get_prompts(modality, defect_type)
            text_prompt = self.prompts.to_grounding_text(prompts)

        # 모델이 없으면 더미 결과 반환
        if self.model is None:
            return DetectionResult(
                boxes=[],
                labels=[],
                scores=[],
                phrases=[],
            )

        # 추론 수행
        try:
            from groundingdino.util.inference import predict, load_image

            # 이미지 로드 (tensor 변환)
            if isinstance(image, (str, Path)):
                image_source, image_tensor = load_image(str(image))
            else:
                # PIL Image를 임시 파일로 저장 후 로드
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                    image_pil.save(f.name)
                    image_source, image_tensor = load_image(f.name)
                    import os
                    os.remove(f.name)

            boxes, logits, phrases = predict(
                model=self.model,
                image=image_tensor,
                caption=text_prompt,
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                device=self.device,
            )

            # 결과 변환
            boxes_list = boxes.cpu().numpy().tolist()
            scores_list = logits.cpu().numpy().tolist()

            return DetectionResult(
                boxes=boxes_list,
                labels=[self._phrase_to_label(p) for p in phrases],
                scores=scores_list,
                phrases=phrases,
            )

        except Exception as e:
            print(f"Detection error: {e}")
            import traceback
            traceback.print_exc()
            return DetectionResult(
                boxes=[],
                labels=[],
                scores=[],
                phrases=[],
            )

    def detect_batch(
        self,
        images: List[Union[str, Path, Image.Image]],
        modality: str = 'ct',
        defect_type: str = 'all',
    ) -> List[DetectionResult]:
        """
        배치 이미지 결함 탐지

        Args:
            images: 이미지 리스트
            modality: 이미지 모달리티
            defect_type: 결함 유형

        Returns:
            DetectionResult 리스트
        """
        results = []
        for image in images:
            result = self.detect(image, modality=modality, defect_type=defect_type)
            results.append(result)
        return results

    def analyze_image(
        self,
        image: Union[str, Path, Image.Image],
        modality: str = 'ct',
    ) -> Dict[str, Any]:
        """
        이미지 분석 (VLM 호환 인터페이스)

        Args:
            image: 이미지 경로 또는 PIL Image
            modality: 이미지 모달리티

        Returns:
            분석 결과 딕셔너리
        """
        detection = self.detect(image, modality=modality)

        # 결함 여부 판단
        has_defect = len(detection.boxes) > 0

        # 결함 유형별 집계
        defect_counts = {}
        for label in detection.labels:
            defect_counts[label] = defect_counts.get(label, 0) + 1

        return {
            'prediction': 'defect' if has_defect else 'normal',
            'is_normal': not has_defect,
            'num_defects': len(detection.boxes),
            'defect_types': list(set(detection.labels)),
            'defect_counts': defect_counts,
            'boxes': detection.boxes,
            'scores': detection.scores,
            'phrases': detection.phrases,
            'modality': modality,
            'confidence': max(detection.scores) if detection.scores else 0.0,
        }

    def visualize(
        self,
        image: Union[str, Path, Image.Image],
        detection: DetectionResult,
        output_path: Optional[str] = None,
        show: bool = False,
    ) -> Image.Image:
        """
        탐지 결과 시각화

        Args:
            image: 원본 이미지
            detection: 탐지 결과
            output_path: 저장 경로 (None이면 저장 안함)
            show: 화면에 표시 여부

        Returns:
            시각화된 이미지
        """
        # 이미지 로드
        if isinstance(image, (str, Path)):
            image_pil = Image.open(image).convert('RGB')
        else:
            image_pil = image.copy()

        draw = ImageDraw.Draw(image_pil)
        width, height = image_pil.size

        # 색상 맵
        colors = {
            'porosity': 'red',
            'void': 'red',
            'bubble': 'orange',
            'resin': 'blue',
            'pollution': 'green',
            'scratch': 'yellow',
            'damage': 'purple',
            'default': 'red',
        }

        # 폰트 (시스템 기본 폰트 사용)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        except Exception:
            font = ImageFont.load_default()

        # 바운딩 박스 그리기
        for box, label, score in zip(detection.boxes, detection.labels, detection.scores):
            # 정규화된 좌표를 픽셀 좌표로 변환
            x1 = int(box[0] * width)
            y1 = int(box[1] * height)
            x2 = int(box[2] * width)
            y2 = int(box[3] * height)

            # 색상 선택
            color = 'red'
            for key, c in colors.items():
                if key in label.lower():
                    color = c
                    break

            # 박스 그리기
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

            # 라벨 그리기
            label_text = f"{label}: {score:.2f}"
            draw.rectangle([x1, y1 - 20, x1 + len(label_text) * 8, y1], fill=color)
            draw.text((x1 + 2, y1 - 18), label_text, fill='white', font=font)

        # 저장
        if output_path:
            image_pil.save(output_path)

        # 표시
        if show:
            image_pil.show()

        return image_pil

    def _phrase_to_label(self, phrase: str) -> str:
        """
        탐지된 구문을 5클래스 라벨로 변환

        Args:
            phrase: 탐지된 텍스트 구문

        Returns:
            5클래스 중 하나 (cell_normal, cell_porosity, module_normal,
                          module_porosity, module_resin_overflow)
        """
        # 5클래스 매핑 사용
        return GroundingPrompts.map_to_unified_class(phrase)

    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        return {
            'model_type': self.model_type,
            'device': self.device,
            'box_threshold': self.box_threshold,
            'text_threshold': self.text_threshold,
            'model_loaded': self.model is not None,
        }


def create_vlg_inference(
    model_type: str = 'swinT',
    device: str = 'cuda',
    box_threshold: float = 0.35,
    text_threshold: float = 0.25,
) -> VLGInference:
    """
    VLG 추론 인스턴스 생성 헬퍼

    Args:
        model_type: 모델 타입 ('swinT' 또는 'swinB')
        device: 실행 디바이스
        box_threshold: 바운딩 박스 임계값
        text_threshold: 텍스트 매칭 임계값

    Returns:
        VLGInference 인스턴스
    """
    return VLGInference(
        model_type=model_type,
        device=device,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )
