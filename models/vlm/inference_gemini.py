"""Gemini VLM 추론 모듈 - Google AI API 사용"""

import sys
from pathlib import Path
from typing import Dict, Optional
import google.generativeai as genai
from PIL import Image

# 프로젝트 루트를 sys.path에 추가
_project_root = Path(__file__).parent.parent.parent.absolute()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# 설정 모듈 로드
try:
    from config import settings
except ImportError:
    settings = None


class GeminiVLMInference:
    """Google Gemini Vision API를 사용한 VLM 추론"""

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        """
        Args:
            api_key: Google AI API 키 (없으면 config 또는 환경변수에서 로드)
            model_name: 사용할 모델 (기본: config의 GEMINI_MODEL_NAME)
        """
        # API 키 우선순위: 파라미터 > config > 환경변수
        if api_key:
            self.api_key = api_key
        elif settings and settings.GEMINI_API_KEY:
            self.api_key = settings.GEMINI_API_KEY
        else:
            import os
            self.api_key = os.getenv("GEMINI_API_KEY")

        if not self.api_key:
            raise ValueError("Gemini API 키가 필요합니다. .env 파일 또는 GEMINI_API_KEY 환경변수를 설정하세요.")

        # 모델명 설정
        if model_name:
            self.model_name = model_name
        elif settings:
            self.model_name = settings.GEMINI_MODEL_NAME
        else:
            self.model_name = "gemini-2.0-flash"

        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_name)

        print(f"GeminiVLMInference 초기화 완료")
        print(f"   - Model: {self.model_name}")

    def analyze_image(self, image_path: str, modality: str = 'ct') -> Dict:
        """
        이미지 분석

        Args:
            image_path: 이미지 파일 경로
            modality: 'ct' 또는 'rgb'

        Returns:
            {
                'is_normal': bool,
                'prediction': 'normal' or 'defect',
                'confidence': float,
                'defect_type': str or None,
                'raw_response': str,
                'location': str or None,
            }
        """
        # 이미지 로드
        image = Image.open(image_path)

        # 프롬프트 생성
        prompt = self._get_prompt(modality)

        try:
            # Gemini API 호출
            response = self.model.generate_content([prompt, image])
            raw_response = response.text

            # 응답 파싱
            result = self._parse_response(raw_response, modality)
            result['raw_response'] = raw_response

            return result

        except Exception as e:
            print(f"Gemini API 오류: {e}")
            return {
                'is_normal': True,
                'prediction': 'error',
                'confidence': 0.0,
                'defect_type': None,
                'raw_response': str(e),
                'location': None,
                'error': str(e)
            }

    def _get_prompt(self, modality: str) -> str:
        """모달리티별 프롬프트 생성"""

        if modality == 'ct':
            return """당신은 배터리 품질 검사 전문가입니다. 이 CT 스캔 이미지를 분석하여 내부 결함을 검사하세요.

검사 항목:
1. 기공 (porosity) - 내부 빈 공간
2. 공극 (void) - 가스에 의한 구멍
3. 크랙 (crack) - 균열
4. 레진 오버플로우 (resin overflow) - 수지 넘침

다음 형식으로 정확히 답변하세요:
판정: [정상/불량]
신뢰도: [0-100]%
결함유형: [없음 또는 결함 종류]
위치: [결함 위치 설명 또는 없음]
소견: [상세 분석 내용]"""

        else:  # rgb
            return """당신은 배터리 품질 검사 전문가입니다. 이 RGB 이미지를 분석하여 외관 결함을 검사하세요.

검사 항목:
1. 오염 (pollution/contamination) - 이물질, 얼룩
2. 스크래치 (scratch) - 긁힘
3. 손상 (damage) - 찍힘, 변형
4. 부식 (corrosion) - 녹, 산화

다음 형식으로 정확히 답변하세요:
판정: [정상/불량]
신뢰도: [0-100]%
결함유형: [없음 또는 결함 종류]
위치: [결함 위치 설명 또는 없음]
소견: [상세 분석 내용]"""

    def _parse_response(self, response: str, modality: str) -> Dict:
        """Gemini 응답 파싱"""
        lines = response.strip().split('\n')

        result = {
            'is_normal': True,
            'prediction': 'normal',
            'confidence': 80.0,
            'defect_type': None,
            'location': None,
        }

        for line in lines:
            line = line.strip()

            if line.startswith('판정:'):
                verdict = line.replace('판정:', '').strip()
                if '불량' in verdict:
                    result['is_normal'] = False
                    result['prediction'] = 'defect'
                else:
                    result['is_normal'] = True
                    result['prediction'] = 'normal'

            elif line.startswith('신뢰도:'):
                try:
                    conf_str = line.replace('신뢰도:', '').strip()
                    conf_str = conf_str.replace('%', '').strip()
                    result['confidence'] = float(conf_str)
                except:
                    result['confidence'] = 80.0

            elif line.startswith('결함유형:'):
                defect_type = line.replace('결함유형:', '').strip()
                if defect_type and defect_type != '없음':
                    result['defect_type'] = defect_type

            elif line.startswith('위치:'):
                location = line.replace('위치:', '').strip()
                if location and location != '없음':
                    result['location'] = location

        return result

    def analyze_ensemble(self, ct_path: Optional[str], rgb_path: Optional[str]) -> Dict:
        """
        CT + RGB 앙상블 분석

        Args:
            ct_path: CT 이미지 경로 (optional)
            rgb_path: RGB 이미지 경로 (optional)

        Returns:
            통합 분석 결과
        """
        ct_result = None
        rgb_result = None

        if ct_path:
            ct_result = self.analyze_image(ct_path, 'ct')

        if rgb_path:
            rgb_result = self.analyze_image(rgb_path, 'rgb')

        # 종합 판정
        ct_is_defect = ct_result and not ct_result.get('is_normal', True)
        rgb_is_defect = rgb_result and not rgb_result.get('is_normal', True)

        if ct_is_defect and rgb_is_defect:
            prediction = 'complex_defect'
            verdict = '복합불량'
        elif ct_is_defect:
            prediction = 'internal_defect'
            verdict = '내부불량'
        elif rgb_is_defect:
            prediction = 'external_defect'
            verdict = '외부불량'
        else:
            prediction = 'normal'
            verdict = '정상'

        # 신뢰도 계산
        confidences = []
        if ct_result and 'confidence' in ct_result:
            confidences.append(ct_result['confidence'])
        if rgb_result and 'confidence' in rgb_result:
            confidences.append(rgb_result['confidence'])

        avg_confidence = sum(confidences) / len(confidences) if confidences else 80.0

        # 결함 유형 결합
        defect_types = []
        if ct_result and ct_result.get('defect_type'):
            defect_types.append(ct_result['defect_type'])
        if rgb_result and rgb_result.get('defect_type'):
            defect_types.append(rgb_result['defect_type'])

        return {
            'prediction': prediction,
            'verdict': verdict,
            'confidence': avg_confidence / 100.0,  # 0-1 범위로 변환
            'defect_type': ' + '.join(defect_types) if defect_types else None,
            'ct_analysis': ct_result,
            'rgb_analysis': rgb_result,
            'model': f'Gemini ({self.model_name})'
        }


# 테스트
if __name__ == "__main__":
    print("Gemini VLM 테스트")
    print("=" * 60)

    # API 키 확인 (config에서 자동 로드)
    if settings and not settings.GEMINI_API_KEY:
        print("GEMINI_API_KEY가 설정되지 않았습니다.")
        print("설정 방법:")
        print("  1. .env 파일에 GEMINI_API_KEY=your_api_key 추가")
        print("  2. 또는 환경변수 설정: export GEMINI_API_KEY='your_api_key'")
        exit(1)

    try:
        # config에서 API 키와 모델명을 자동으로 로드
        vlm = GeminiVLMInference()
        print("모델 로드 성공!")

        # 테스트 이미지가 있으면 분석
        test_images = list(Path("data/ct_unified/test").glob("**/*.png"))[:1]
        if test_images:
            print(f"\n테스트 이미지: {test_images[0]}")
            result = vlm.analyze_image(str(test_images[0]), 'ct')
            print(f"결과: {result}")
        else:
            print("테스트 이미지 없음")

    except Exception as e:
        print(f"오류: {e}")
