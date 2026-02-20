"""Qwen3-VL 기반 VLM 추론 모듈

Qwen3-VL 특징:
- 텍스트 분석 + Bounding Box 출력 지원
- 2D/3D Grounding 지원
- transformers >= 4.57.0 필요
"""
import torch
import json
import re
from pathlib import Path
from typing import Union, Dict, Any, Optional, List, Tuple
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

from .prompts import BatteryDefectPrompts


class VLMInference:
    """Qwen3-VL을 이용한 배터리 결함 분석 (텍스트 + BBox 지원)"""

    # 지원하는 모델 크기
    MODEL_SIZES = {
        '2b': 'Qwen/Qwen3-VL-2B-Instruct',
        '4b': 'Qwen/Qwen3-VL-4B-Instruct',
        '8b': 'Qwen/Qwen3-VL-8B-Instruct',
        '32b': 'Qwen/Qwen3-VL-32B-Instruct',
    }

    # 결함 클래스 매핑
    DEFECT_CLASSES = {
        'ct': ['cell_normal', 'cell_porosity', 'module_normal',
               'module_porosity', 'module_resin_overflow'],
        'rgb': ['normal', 'pollution', 'scratch', 'damage', 'discoloration']
    }

    def __init__(
        self,
        model_size: str = '8b',
        device: str = 'cuda',
        torch_dtype: torch.dtype = torch.bfloat16,
        use_flash_attention: bool = True,
    ):
        """
        VLM 모델 초기화

        Args:
            model_size: 모델 크기 ('2b', '4b', '8b', '32b')
            device: 실행 디바이스
            torch_dtype: 모델 dtype
            use_flash_attention: Flash Attention 2 사용 여부
        """
        self.device = device
        self.model_size = model_size
        self.model_name = self.MODEL_SIZES.get(model_size, self.MODEL_SIZES['8b'])

        print(f"Loading Qwen3-VL model: {self.model_name}")

        # 모델 로드
        model_kwargs = {
            'torch_dtype': torch_dtype,
            'device_map': 'auto' if device == 'cuda' else None,
        }

        # Flash Attention 사용 여부 결정
        if use_flash_attention:
            try:
                import flash_attn
                model_kwargs['attn_implementation'] = 'flash_attention_2'
                print("Using Flash Attention 2")
            except ImportError:
                print("Flash Attention not available, using default attention")
                use_flash_attention = False

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_name,
            **model_kwargs
        )

        # 프로세서 로드
        self.processor = AutoProcessor.from_pretrained(self.model_name)

        # 프롬프트 템플릿
        self.prompts = BatteryDefectPrompts()

        print(f"Qwen3-VL model loaded successfully on {device}")

    def analyze_image(
        self,
        image: Union[str, Path, Image.Image],
        modality: str = 'ct',
        detailed: bool = True,
        max_new_tokens: int = 512,
    ) -> Dict[str, Any]:
        """
        배터리 이미지 분석 (텍스트 분석)

        Args:
            image: 이미지 경로 또는 PIL Image
            modality: 이미지 모달리티 ('ct' 또는 'rgb')
            detailed: 상세 분석 여부
            max_new_tokens: 최대 생성 토큰 수

        Returns:
            분석 결과 딕셔너리
        """
        # 이미지 로드
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')

        # 프롬프트 선택
        prompt = self.prompts.get_prompt(modality, detailed)

        # 메시지 구성
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # 입력 처리
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.device)

        # 추론
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        # 생성된 토큰만 추출
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        # 디코딩
        response = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        # 결과 파싱
        result = self._parse_response(response, modality)
        result['raw_response'] = response
        result['modality'] = modality

        return result

    def detect_defects(
        self,
        image: Union[str, Path, Image.Image],
        modality: str = 'ct',
        max_new_tokens: int = 512,
    ) -> Dict[str, Any]:
        """
        결함 위치 탐지 (Bounding Box 출력)

        Args:
            image: 이미지 경로 또는 PIL Image
            modality: 이미지 모달리티 ('ct' 또는 'rgb')
            max_new_tokens: 최대 생성 토큰 수

        Returns:
            탐지 결과 딕셔너리 (bboxes 포함)
        """
        # 이미지 로드
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')

        img_width, img_height = image.size

        # Grounding 프롬프트 (Qwen3-VL 네이티브 형식)
        if modality == 'ct':
            defect_types = "porosity, void, crack, resin overflow"
            defect_refs = [
                "<|object_ref_start|>porosity<|object_ref_end|>",
                "<|object_ref_start|>void<|object_ref_end|>",
                "<|object_ref_start|>crack<|object_ref_end|>",
                "<|object_ref_start|>resin overflow<|object_ref_end|>",
            ]
        else:
            defect_types = "pollution, scratch, damage, discoloration"
            defect_refs = [
                "<|object_ref_start|>pollution<|object_ref_end|>",
                "<|object_ref_start|>scratch<|object_ref_end|>",
                "<|object_ref_start|>damage<|object_ref_end|>",
                "<|object_ref_start|>discoloration<|object_ref_end|>",
            ]

        # Qwen3-VL grounding 프롬프트
        prompt = f"""Detect and locate all defects in this battery {modality.upper()} image.
Find any: {', '.join(defect_refs)}

For each defect found, output the bounding box using this format:
<|object_ref_start|>defect_type<|object_ref_end|><|box_start|>(x1,y1),(x2,y2)<|box_end|>

Coordinates should be in 0-1000 range (normalized).
If no defects are found, respond with: No defects detected."""

        # 메시지 구성
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # 입력 처리
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.device)

        # 추론
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        # 생성된 토큰만 추출
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        # 디코딩
        response = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        # BBox 파싱
        result = self._parse_detection_response(response, img_width, img_height)
        result['raw_response'] = response
        result['modality'] = modality
        result['image_size'] = (img_width, img_height)

        return result

    def analyze_with_grounding(
        self,
        image: Union[str, Path, Image.Image],
        modality: str = 'ct',
        max_new_tokens: int = 1024,
    ) -> Dict[str, Any]:
        """
        텍스트 분석 + Bounding Box 동시 출력

        Args:
            image: 이미지 경로 또는 PIL Image
            modality: 이미지 모달리티 ('ct' 또는 'rgb')
            max_new_tokens: 최대 생성 토큰 수

        Returns:
            분석 결과 + 탐지 결과 딕셔너리
        """
        # 이미지 로드
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')

        img_width, img_height = image.size

        # 통합 프롬프트 (Qwen3-VL 네이티브 형식)
        if modality == 'ct':
            defect_types = "porosity (기공), void (공극), resin overflow (수지 넘침)"
            defect_refs = "<|object_ref_start|>porosity<|object_ref_end|>, <|object_ref_start|>void<|object_ref_end|>, <|object_ref_start|>resin overflow<|object_ref_end|>"
        else:
            defect_types = "pollution (오염), scratch (스크래치), damage (손상)"
            defect_refs = "<|object_ref_start|>pollution<|object_ref_end|>, <|object_ref_start|>scratch<|object_ref_end|>, <|object_ref_start|>damage<|object_ref_end|>"

        prompt = f"""이 배터리 {modality.upper()} 이미지를 분석하고 결함을 탐지하세요.

탐지 대상: {defect_types}

다음 형식으로 출력하세요:

## 분석 결과
- 판정: [정상/결함]
- 결함 유형: [결함 유형 또는 없음]
- 신뢰도: [0-100]%
- 분석 근거: [설명]

## 결함 위치
결함이 발견되면 위치를 다음 형식으로 표시:
<|object_ref_start|>결함유형<|object_ref_end|><|box_start|>(x1,y1),(x2,y2)<|box_end|>

좌표는 0-1000 범위로 정규화해서 출력하세요.
결함이 없으면 "결함 없음"으로 출력하세요."""

        # 메시지 구성
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # 입력 처리
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.device)

        # 추론
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        # 생성된 토큰만 추출
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        # 디코딩
        response = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        # 텍스트 분석 파싱
        text_result = self._parse_response(response, modality)

        # BBox 파싱
        detection_result = self._parse_detection_response(response, img_width, img_height)

        # 결과 통합
        result = {
            **text_result,
            'detections': detection_result.get('detections', []),
            'raw_response': response,
            'modality': modality,
            'image_size': (img_width, img_height),
        }

        return result

    def analyze_batch(
        self,
        images: List[Union[str, Path, Image.Image]],
        modality: str = 'ct',
        detailed: bool = True,
        batch_size: int = 4,
    ) -> List[Dict[str, Any]]:
        """
        배치 이미지 분석

        Args:
            images: 이미지 리스트
            modality: 이미지 모달리티
            detailed: 상세 분석 여부
            batch_size: 배치 크기

        Returns:
            분석 결과 리스트
        """
        results = []

        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            for img in batch:
                result = self.analyze_image(img, modality, detailed)
                results.append(result)

        return results

    def zero_shot_classify(
        self,
        image: Union[str, Path, Image.Image],
        modality: str = 'ct',
        max_new_tokens: int = 256,
    ) -> Dict[str, Any]:
        """
        Zero-shot 결함 분류 (JSON 형식 출력)

        Args:
            image: 이미지 경로 또는 PIL Image
            modality: 이미지 모달리티 ('ct' 또는 'rgb')
            max_new_tokens: 최대 생성 토큰 수

        Returns:
            분류 결과 딕셔너리
        """
        # 이미지 로드
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')

        # Zero-shot 프롬프트 (modality에 따라 선택)
        if modality == 'rgb':
            prompt = self.prompts.ZERO_SHOT_CLASSIFICATION_RGB
        else:
            prompt = self.prompts.ZERO_SHOT_CLASSIFICATION

        # 메시지 구성
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # 입력 처리
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.device)

        # 추론
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        # 생성된 토큰만 추출
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        # 디코딩
        response = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        # JSON 파싱
        result = self._parse_json_response(response)
        result['raw_response'] = response

        return result

    def _parse_response(self, response: str, modality: str) -> Dict[str, Any]:
        """
        모델 응답 파싱

        Args:
            response: 모델 응답 텍스트
            modality: 이미지 모달리티

        Returns:
            파싱된 결과 딕셔너리
        """
        result = {
            'prediction': None,
            'confidence': None,
            'defect_type': None,
            'location': None,
            'explanation': None,
        }

        lines = response.strip().split('\n')

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # 분류/판정 추출
            if '분류:' in line or '판정:' in line:
                value = line.split(':')[-1].strip().strip('[]')
                result['prediction'] = value

                # 정상 여부 판단
                if '정상' in value.lower() or 'normal' in value.lower():
                    result['is_normal'] = True
                else:
                    result['is_normal'] = False

            # 신뢰도 추출
            elif '신뢰도:' in line:
                match = re.search(r'(\d+)', line)
                if match:
                    result['confidence'] = int(match.group(1))

            # 결함 유형 추출
            elif '결함 유형:' in line:
                value = line.split(':')[-1].strip().strip('[]')
                if value and value != '없음':
                    result['defect_type'] = value

            # 위치 추출
            elif '결함 위치:' in line or '위치:' in line:
                value = line.split(':')[-1].strip().strip('[]')
                if value and value != '없음':
                    result['location'] = value

            # 근거 추출
            elif '분석 근거:' in line or '근거:' in line:
                value = line.split(':')[-1].strip().strip('[]')
                result['explanation'] = value

        return result

    def _parse_detection_response(
        self,
        response: str,
        img_width: int,
        img_height: int
    ) -> Dict[str, Any]:
        """
        Detection 응답에서 BBox 파싱

        Qwen3-VL은 두 가지 형식으로 BBox를 출력:
        1. 특수 토큰: <|object_ref_start|>label<|object_ref_end|><|box_start|>(x1,y1),(x2,y2)<|box_end|>
        2. JSON: {"bbox_2d": [x1, y1, x2, y2], "label": "..."}

        좌표는 0-1000 범위로 정규화되어 있음

        Args:
            response: 모델 응답 텍스트
            img_width: 이미지 너비
            img_height: 이미지 높이

        Returns:
            파싱된 탐지 결과 딕셔너리
        """
        result = {
            'detections': [],
            'is_normal': True,
        }

        detections = []

        # 방법 1: 특수 토큰 형식 파싱
        # <|object_ref_start|>label<|object_ref_end|><|box_start|>(x1,y1),(x2,y2)<|box_end|>
        box_pattern = re.compile(
            r'<\|object_ref_start\|>(.+?)<\|object_ref_end\|>\s*'
            r'<\|box_start\|>\((\d+),(\d+)\),\((\d+),(\d+)\)<\|box_end\|>',
            re.DOTALL
        )
        for match in box_pattern.finditer(response):
            label = match.group(1).strip()
            x1, y1, x2, y2 = map(int, match.groups()[1:])
            detections.append({
                'label': label,
                'bbox_2d': [x1, y1, x2, y2],
                'bbox_pixel': [
                    int(x1 / 1000 * img_width),
                    int(y1 / 1000 * img_height),
                    int(x2 / 1000 * img_width),
                    int(y2 / 1000 * img_height),
                ],
            })

        # 방법 2: box_start만 있는 단순 형식
        # <|box_start|>(x1,y1),(x2,y2)<|box_end|>
        if not detections:
            simple_box_pattern = re.compile(
                r'<\|box_start\|>\((\d+),(\d+)\),\((\d+),(\d+)\)<\|box_end\|>'
            )
            for match in simple_box_pattern.finditer(response):
                x1, y1, x2, y2 = map(int, match.groups())
                detections.append({
                    'label': 'defect',
                    'bbox_2d': [x1, y1, x2, y2],
                    'bbox_pixel': [
                        int(x1 / 1000 * img_width),
                        int(y1 / 1000 * img_height),
                        int(x2 / 1000 * img_width),
                        int(y2 / 1000 * img_height),
                    ],
                })

        # 방법 3: JSON 형식 파싱 (bbox_2d 키 포함)
        if not detections:
            # JSON 배열 또는 객체에서 bbox_2d 찾기
            json_patterns = [
                # {"detections": [...]} 형식
                r'\{\s*"detections"\s*:\s*(\[[\s\S]*?\])\s*\}',
                # 단일 배열 [...] 형식
                r'\[(\{[^[\]]*"bbox_2d"[^[\]]*\}(?:\s*,\s*\{[^[\]]*"bbox_2d"[^[\]]*\})*)\]',
                # 단일 객체 {...} 형식
                r'(\{[^{}]*"bbox_2d"\s*:\s*\[[^\]]+\][^{}]*\})',
            ]

            for pattern in json_patterns:
                matches = re.findall(pattern, response, re.DOTALL)
                if matches:
                    for match_str in matches:
                        try:
                            # 배열인지 확인
                            if match_str.strip().startswith('['):
                                items = json.loads(match_str)
                            else:
                                # 단일 객체를 배열로 감싸기
                                items = json.loads(f'[{match_str}]')

                            for item in items:
                                if 'bbox_2d' in item:
                                    box = item['bbox_2d']
                                    detections.append({
                                        'label': item.get('label', 'defect'),
                                        'bbox_2d': box,
                                        'bbox_pixel': [
                                            int(box[0] / 1000 * img_width),
                                            int(box[1] / 1000 * img_height),
                                            int(box[2] / 1000 * img_width),
                                            int(box[3] / 1000 * img_height),
                                        ],
                                        'confidence': item.get('confidence'),
                                    })
                        except (json.JSONDecodeError, TypeError, KeyError):
                            continue

                    if detections:
                        break

        result['detections'] = detections
        result['is_normal'] = len(detections) == 0

        return result

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """
        JSON 형식 응답 파싱

        Args:
            response: 모델 응답 텍스트

        Returns:
            파싱된 결과 딕셔너리
        """
        result = {
            'prediction': None,
            'defect_type': None,
            'confidence': None,
            'explanation': None,
        }

        # JSON 블록 추출
        json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)

        if json_match:
            try:
                parsed = json.loads(json_match.group())
                result.update(parsed)

                # 정상 여부 판단
                if result.get('prediction'):
                    result['is_normal'] = result['prediction'].lower() == 'normal'
            except json.JSONDecodeError:
                # JSON 파싱 실패 시 텍스트 파싱 시도
                result = self._parse_response(response, 'unknown')
        else:
            result = self._parse_response(response, 'unknown')

        return result

    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        return {
            'model_name': self.model_name,
            'model_size': self.model_size,
            'model_version': 'Qwen3-VL',
            'device': self.device,
            'num_parameters': sum(p.numel() for p in self.model.parameters()),
            'supported_modalities': list(self.DEFECT_CLASSES.keys()),
            'capabilities': ['text_analysis', 'bounding_box', '2d_grounding'],
        }


def create_vlm_inference(
    model_size: str = '8b',
    device: str = 'cuda',
) -> VLMInference:
    """
    VLM 추론 인스턴스 생성 헬퍼

    Args:
        model_size: 모델 크기 ('2b', '4b', '8b', '32b')
        device: 실행 디바이스

    Returns:
        VLMInference 인스턴스
    """
    return VLMInference(
        model_size=model_size,
        device=device,
    )
