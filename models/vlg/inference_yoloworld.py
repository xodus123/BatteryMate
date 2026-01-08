"""YOLO-World 기반 VLG 추론 모듈"""
import torch
import numpy as np
from pathlib import Path
from typing import Union, Dict, Any, List, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont
from dataclasses import dataclass

from .prompts import GroundingPrompts, map_to_unified_class


@dataclass
class DetectionResult:
    """결함 탐지 결과"""
    boxes: List[List[float]]  # [[x1, y1, x2, y2], ...]
    labels: List[str]  # 탐지된 결함 라벨
    scores: List[float]  # 신뢰도 점수
    phrases: List[str]  # 매칭된 텍스트 구문


class YOLOWorldInference:
    """YOLO-World를 이용한 결함 위치 탐지"""

    # 사용 가능한 모델
    MODEL_CONFIGS = {
        'yolov8s-world': {
            'weights': 'yolov8s-worldv2.pt',
            'size': 'small',
        },
        'yolov8m-world': {
            'weights': 'yolov8m-worldv2.pt',
            'size': 'medium',
        },
        'yolov8l-world': {
            'weights': 'yolov8l-worldv2.pt',
            'size': 'large',
        },
        'yolov8x-world': {
            'weights': 'yolov8x-worldv2.pt',
            'size': 'xlarge',
        },
    }

    # 결함 탐지용 클래스 정의
    DEFECT_CLASSES = {
        'ct': [
            'porosity', 'pore', 'void', 'bubble', 'hole', 'cavity',
            'resin overflow', 'resin leak', 'excess resin',
            'crack', 'fracture', 'defect', 'anomaly', 'damage'
        ],
        'rgb': [
            'pollution', 'contamination', 'stain', 'dirt', 'residue',
            'scratch', 'abrasion', 'mark', 'scuff',
            'damage', 'dent', 'deformation', 'defect', 'anomaly'
        ],
        'all': [
            'porosity', 'pore', 'void', 'bubble', 'hole', 'cavity',
            'resin overflow', 'resin leak', 'excess resin',
            'pollution', 'contamination', 'stain', 'dirt',
            'scratch', 'abrasion', 'damage', 'crack', 'defect', 'anomaly'
        ]
    }

    def __init__(
        self,
        model_type: str = 'yolov8s-world',
        device: str = 'cuda',
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        weights_path: Optional[str] = None,
    ):
        """
        YOLO-World 모델 초기화

        Args:
            model_type: 모델 타입 ('yolov8s-world', 'yolov8m-world', 'yolov8l-world', 'yolov8x-world')
            device: 실행 디바이스
            conf_threshold: 신뢰도 임계값
            iou_threshold: NMS IoU 임계값
            weights_path: 커스텀 가중치 경로
        """
        self.device = device
        self.model_type = model_type
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        # 프롬프트 템플릿
        self.prompts = GroundingPrompts()

        # 현재 설정된 클래스
        self.current_classes = []

        # 모델 로드
        self.model = self._load_model(model_type, weights_path)

        if self.model is not None:
            print(f"✅ YOLO-World ({model_type}) loaded on {device}")

    def _load_model(self, model_type: str, weights_path: Optional[str] = None):
        """
        YOLO-World 모델 로드

        Args:
            model_type: 모델 타입
            weights_path: 커스텀 가중치 경로

        Returns:
            로드된 모델
        """
        try:
            from ultralytics import YOLO

            config = self.MODEL_CONFIGS.get(model_type, self.MODEL_CONFIGS['yolov8s-world'])

            # 가중치 경로 결정
            if weights_path is None:
                weights_path = config['weights']

            # 로컬 가중치 파일 확인
            local_weights = Path(f'models/vlg/weights/{weights_path}')
            if local_weights.exists():
                weights_path = str(local_weights)

            # 모델 로드
            model = YOLO(weights_path)

            # GPU로 이동
            if self.device == 'cuda' and torch.cuda.is_available():
                model.to('cuda')

            return model

        except ImportError:
            print("⚠️ ultralytics not installed. Run: pip install ultralytics")
            return None
        except Exception as e:
            print(f"❌ Error loading YOLO-World: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _set_classes(self, modality: str = 'all'):
        """탐지할 클래스 설정"""
        classes = self.DEFECT_CLASSES.get(modality, self.DEFECT_CLASSES['all'])

        # 이미 같은 클래스가 설정되어 있으면 스킵
        if classes == self.current_classes:
            return

        self.current_classes = classes

        if self.model is not None:
            try:
                self.model.set_classes(classes)
            except Exception as e:
                print(f"⚠️ Failed to set classes: {e}")

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
            text_prompt: 커스텀 텍스트 프롬프트 (클래스 리스트로 변환됨)
            modality: 이미지 모달리티 ('ct' 또는 'rgb')
            defect_type: 결함 유형

        Returns:
            DetectionResult 객체
        """
        # 모델이 없으면 더미 결과 반환
        if self.model is None:
            return DetectionResult(
                boxes=[],
                labels=[],
                scores=[],
                phrases=[],
            )

        # 클래스 설정
        if text_prompt:
            # 커스텀 프롬프트에서 클래스 추출
            custom_classes = [c.strip() for c in text_prompt.replace('.', ',').split(',') if c.strip()]
            if custom_classes:
                self.model.set_classes(custom_classes)
                self.current_classes = custom_classes
        else:
            self._set_classes(modality)

        # 이미지 준비
        if isinstance(image, Image.Image):
            # PIL Image -> numpy array
            image_input = np.array(image.convert('RGB'))
        else:
            image_input = str(image)

        # 추론 수행
        try:
            results = self.model.predict(
                image_input,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                verbose=False,
            )

            if len(results) == 0 or results[0].boxes is None:
                return DetectionResult(
                    boxes=[],
                    labels=[],
                    scores=[],
                    phrases=[],
                )

            result = results[0]
            boxes = result.boxes

            # 결과 추출
            boxes_xyxy = boxes.xyxy.cpu().numpy().tolist()
            scores_list = boxes.conf.cpu().numpy().tolist()
            class_ids = boxes.cls.cpu().numpy().astype(int).tolist()

            # 클래스 ID를 라벨로 변환
            labels = []
            phrases = []
            for cls_id in class_ids:
                if cls_id < len(self.current_classes):
                    phrase = self.current_classes[cls_id]
                    phrases.append(phrase)
                    labels.append(self._phrase_to_label(phrase, modality))
                else:
                    phrases.append('unknown')
                    labels.append('unknown')

            # 박스 좌표 정규화 (0~1 범위)
            if isinstance(image, Image.Image):
                img_w, img_h = image.size
            else:
                img = Image.open(image)
                img_w, img_h = img.size

            normalized_boxes = []
            for box in boxes_xyxy:
                x1, y1, x2, y2 = box
                normalized_boxes.append([
                    x1 / img_w,
                    y1 / img_h,
                    x2 / img_w,
                    y2 / img_h
                ])

            return DetectionResult(
                boxes=normalized_boxes,
                labels=labels,
                scores=scores_list,
                phrases=phrases,
            )

        except Exception as e:
            print(f"❌ Detection error: {e}")
            import traceback
            traceback.print_exc()
            return DetectionResult(
                boxes=[],
                labels=[],
                scores=[],
                phrases=[],
            )

    def _phrase_to_label(self, phrase: str, modality: str = 'ct') -> str:
        """구문을 통합 라벨로 변환"""
        return map_to_unified_class(phrase, modality)

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

        # 최대 신뢰도
        max_confidence = max(detection.scores) if detection.scores else 0.0

        return {
            'prediction': 'defect' if has_defect else 'normal',
            'is_normal': not has_defect,
            'num_defects': len(detection.boxes),
            'defect_types': list(set(detection.labels)),
            'defect_counts': defect_counts,
            'boxes': detection.boxes,
            'scores': detection.scores,
            'labels': detection.labels,
            'phrases': detection.phrases,
            'confidence': max_confidence,
            'model': f'yolo-world ({self.model_type})',
        }

    def visualize(
        self,
        image: Union[str, Path, Image.Image],
        detection: Optional[DetectionResult] = None,
        modality: str = 'ct',
        show_labels: bool = True,
        show_scores: bool = True,
        box_color: str = 'red',
        text_color: str = 'white',
        line_width: int = 3,
    ) -> Image.Image:
        """
        탐지 결과 시각화

        Args:
            image: 원본 이미지
            detection: DetectionResult (없으면 새로 탐지)
            modality: 이미지 모달리티
            show_labels: 라벨 표시 여부
            show_scores: 신뢰도 표시 여부
            box_color: 박스 색상
            text_color: 텍스트 색상
            line_width: 선 두께

        Returns:
            시각화된 PIL Image
        """
        # 이미지 로드
        if isinstance(image, (str, Path)):
            image_pil = Image.open(image).convert('RGB')
        else:
            image_pil = image.copy().convert('RGB')

        # 탐지 수행
        if detection is None:
            detection = self.detect(image, modality=modality)

        # 결과가 없으면 원본 반환
        if not detection.boxes:
            return image_pil

        # 그리기
        draw = ImageDraw.Draw(image_pil)
        img_w, img_h = image_pil.size

        # 폰트 설정
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        except:
            font = ImageFont.load_default()

        # 결함 유형별 색상
        color_map = {
            'porosity': '#FF6B6B',
            'resin_overflow': '#4ECDC4',
            'pollution': '#FFE66D',
            'scratch': '#95E1D3',
            'damaged': '#F38181',
            'unknown': '#CCCCCC',
        }

        for i, (box, label, score, phrase) in enumerate(zip(
            detection.boxes, detection.labels, detection.scores, detection.phrases
        )):
            # 정규화된 좌표를 픽셀 좌표로 변환
            x1 = int(box[0] * img_w)
            y1 = int(box[1] * img_h)
            x2 = int(box[2] * img_w)
            y2 = int(box[3] * img_h)

            # 색상 결정
            color = color_map.get(label, box_color)

            # 박스 그리기
            draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

            # 라벨 텍스트
            if show_labels or show_scores:
                text_parts = []
                if show_labels:
                    text_parts.append(label)
                if show_scores:
                    text_parts.append(f'{score:.2f}')
                text = ' '.join(text_parts)

                # 텍스트 배경
                bbox = draw.textbbox((x1, y1 - 20), text, font=font)
                draw.rectangle(bbox, fill=color)
                draw.text((x1, y1 - 20), text, fill=text_color, font=font)

        return image_pil

    def compare_with_groundingdino(
        self,
        image: Union[str, Path, Image.Image],
        modality: str = 'ct',
    ) -> Dict[str, Any]:
        """
        GroundingDINO와 비교 분석

        Args:
            image: 이미지
            modality: 모달리티

        Returns:
            비교 결과
        """
        # YOLO-World 결과
        yolo_result = self.detect(image, modality=modality)

        # GroundingDINO 결과
        try:
            from .inference import VLGInference
            gdino = VLGInference(device=self.device)
            gdino_result = gdino.detect(image, modality=modality)
        except Exception as e:
            gdino_result = None
            print(f"⚠️ GroundingDINO comparison failed: {e}")

        return {
            'yolo_world': {
                'num_detections': len(yolo_result.boxes),
                'labels': yolo_result.labels,
                'scores': yolo_result.scores,
            },
            'groundingdino': {
                'num_detections': len(gdino_result.boxes) if gdino_result else 0,
                'labels': gdino_result.labels if gdino_result else [],
                'scores': gdino_result.scores if gdino_result else [],
            } if gdino_result else None,
        }


# 테스트용 메인 함수
def main():
    """테스트 실행"""
    import argparse

    parser = argparse.ArgumentParser(description='YOLO-World VLG Test')
    parser.add_argument('--image', type=str, help='테스트 이미지 경로')
    parser.add_argument('--model', type=str, default='yolov8s-world',
                       choices=['yolov8s-world', 'yolov8m-world', 'yolov8l-world', 'yolov8x-world'])
    parser.add_argument('--modality', type=str, default='ct', choices=['ct', 'rgb', 'all'])
    parser.add_argument('--conf', type=float, default=0.25, help='신뢰도 임계값')
    parser.add_argument('--save', type=str, help='결과 저장 경로')
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("YOLO-World VLG 테스트")
    print(f"{'='*60}\n")

    # 모델 로드
    vlg = YOLOWorldInference(
        model_type=args.model,
        conf_threshold=args.conf,
    )

    if args.image:
        # 이미지 분석
        print(f"이미지 분석: {args.image}")
        result = vlg.analyze_image(args.image, modality=args.modality)

        print(f"\n결과:")
        print(f"  - 예측: {result['prediction']}")
        print(f"  - 결함 수: {result['num_defects']}")
        print(f"  - 결함 유형: {result['defect_types']}")
        print(f"  - 최대 신뢰도: {result['confidence']:.4f}")

        if args.save:
            # 시각화 저장
            detection = DetectionResult(
                boxes=result['boxes'],
                labels=result['labels'],
                scores=result['scores'],
                phrases=result['phrases'],
            )
            vis_image = vlg.visualize(args.image, detection, modality=args.modality)
            vis_image.save(args.save)
            print(f"\n✅ 결과 저장: {args.save}")
    else:
        print("✅ 모델 로드 테스트 완료")
        print(f"   사용법: python -m models.vlg.inference_yoloworld --image <path> --modality ct")


if __name__ == "__main__":
    main()
