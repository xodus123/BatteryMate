"""Qwen2-VL 기반 VLM 추론 모듈"""
import torch
import json
import re
from pathlib import Path
from typing import Union, Dict, Any, Optional, List
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

from .prompts import BatteryDefectPrompts


class VLMInference:
    """Qwen2-VL을 이용한 배터리 결함 분석"""

    # 지원하는 모델 크기
    MODEL_SIZES = {
        '2b': 'Qwen/Qwen2-VL-2B-Instruct',
        '7b': 'Qwen/Qwen2-VL-7B-Instruct',
        '72b': 'Qwen/Qwen2-VL-72B-Instruct',
    }

    # 결함 클래스 매핑
    DEFECT_CLASSES = {
        'ct': ['cell_normal', 'cell_porosity', 'module_normal',
               'module_porosity', 'module_resin_overflow'],
        'rgb': ['normal', 'pollution', 'scratch', 'damage', 'discoloration']
    }

    def __init__(
        self,
        model_size: str = '7b',
        device: str = 'cuda',
        torch_dtype: torch.dtype = torch.bfloat16,
        use_flash_attention: bool = True,
    ):
        """
        VLM 모델 초기화

        Args:
            model_size: 모델 크기 ('2b', '7b', '72b')
            device: 실행 디바이스
            torch_dtype: 모델 dtype
            use_flash_attention: Flash Attention 2 사용 여부
        """
        self.device = device
        self.model_size = model_size
        self.model_name = self.MODEL_SIZES.get(model_size, self.MODEL_SIZES['7b'])

        print(f"Loading VLM model: {self.model_name}")

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
            except ImportError:
                print("Flash Attention not available, using default attention")
                use_flash_attention = False

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_name,
            **model_kwargs
        )

        # 프로세서 로드
        self.processor = AutoProcessor.from_pretrained(self.model_name)

        # 프롬프트 템플릿
        self.prompts = BatteryDefectPrompts()

        print(f"VLM model loaded successfully on {device}")

    def analyze_image(
        self,
        image: Union[str, Path, Image.Image],
        modality: str = 'ct',
        detailed: bool = True,
        max_new_tokens: int = 512,
    ) -> Dict[str, Any]:
        """
        배터리 이미지 분석

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
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt",
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
        max_new_tokens: int = 256,
    ) -> Dict[str, Any]:
        """
        Zero-shot 결함 분류 (JSON 형식 출력)

        Args:
            image: 이미지 경로 또는 PIL Image
            max_new_tokens: 최대 생성 토큰 수

        Returns:
            분류 결과 딕셔너리
        """
        # 이미지 로드
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')

        # Zero-shot 프롬프트
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
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt",
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
            'device': self.device,
            'num_parameters': sum(p.numel() for p in self.model.parameters()),
            'supported_modalities': list(self.DEFECT_CLASSES.keys()),
        }


def create_vlm_inference(
    model_size: str = '7b',
    device: str = 'cuda',
) -> VLMInference:
    """
    VLM 추론 인스턴스 생성 헬퍼

    Args:
        model_size: 모델 크기 ('2b', '7b', '72b')
        device: 실행 디바이스

    Returns:
        VLMInference 인스턴스
    """
    return VLMInference(
        model_size=model_size,
        device=device,
    )
