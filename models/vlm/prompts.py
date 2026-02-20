"""배터리 결함 분석용 프롬프트 템플릿 - CT 5클래스 / RGB 3클래스"""


# CT 통일된 5클래스 정의
UNIFIED_CLASSES = [
    'cell_normal',
    'cell_porosity',
    'module_normal',
    'module_porosity',
    'module_resin_overflow',
]

# RGB 3클래스 정의
RGB_CLASSES = [
    'normal',
    'pollution',
    'damaged',
]


class BatteryDefectPrompts:
    """배터리 결함 분석 프롬프트 - 5클래스 통일"""

    # CT 이미지 분석 프롬프트 (5클래스)
    CT_ANALYSIS = """이 CT 이미지는 배터리 내부를 촬영한 것입니다.
다음 5가지 클래스 중 **반드시 하나**로 분류해주세요:

1. cell_normal: 정상 셀 (결함 없음)
2. cell_porosity: 셀 내부 기공(porosity) 결함
3. module_normal: 정상 모듈 (결함 없음)
4. module_porosity: 모듈 내부 기공 결함
5. module_resin_overflow: 모듈 레진 오버플로우 결함

**중요**: 결함이 보이지 않으면 cell_normal 또는 module_normal로 판정하세요.

응답 형식:
- 분류: [5클래스 중 하나]
- 판정: [정상/불량]
- 신뢰도: [0-100]%
- 결함 위치: [있다면 설명, 없으면 "없음"]
- 분석 근거: [판단 이유]"""

    CT_ANALYSIS_SIMPLE = """이 배터리 CT 이미지를 분석해주세요.

5클래스 중 하나로 분류:
- cell_normal (정상 셀)
- cell_porosity (셀 기공 결함)
- module_normal (정상 모듈)
- module_porosity (모듈 기공 결함)
- module_resin_overflow (레진 오버플로우)

결함이 없으면 반드시 normal 클래스로 판정하세요."""

    # RGB 이미지 분석 프롬프트
    RGB_ANALYSIS = """이 RGB 이미지는 배터리 외관을 촬영한 것입니다.
외부 결함 여부를 분석해주세요.

확인할 결함:
- 오염 (pollution)
- 스크래치
- 손상
- 변색

**중요**: 결함이 보이지 않으면 "정상"으로 판정하세요.

응답 형식:
- 판정: [정상/불량]
- 결함 유형: [있다면, 없으면 "없음"]
- 결함 위치: [있다면 설명, 없으면 "없음"]
- 분석 근거: [판단 이유]"""

    RGB_ANALYSIS_SIMPLE = """이 배터리 외관 이미지를 분석해주세요.
외부 결함(오염, 손상 등)이 있는지 판단해주세요.
결함이 없으면 "정상"으로 판정하세요."""

    # Zero-shot 분류 프롬프트 (5클래스)
    ZERO_SHOT_CLASSIFICATION = """Analyze this battery image and classify it into ONE of these 5 classes:

1. cell_normal - Normal cell (no defect)
2. cell_porosity - Cell with porosity defect
3. module_normal - Normal module (no defect)
4. module_porosity - Module with porosity defect
5. module_resin_overflow - Module with resin overflow defect

IMPORTANT: If no defect is visible, classify as cell_normal or module_normal.

Answer in JSON format:
{
    "classification": "one of 5 classes above",
    "prediction": "normal" or "defect",
    "confidence": 0-100,
    "explanation": "brief explanation"
}"""

    # Zero-shot 분류 프롬프트 (RGB 3클래스)
    ZERO_SHOT_CLASSIFICATION_RGB = """Analyze this battery exterior image and classify it into ONE of these 3 classes:

1. normal - Normal battery surface (no defect)
2. pollution - Surface contamination or stain
3. damaged - Physical damage, mixed defects, or deformation

IMPORTANT: If the surface looks clean with no visible defects, classify as normal.

Answer in JSON format:
{
    "classification": "one of 3 classes above",
    "prediction": "normal" or "defect",
    "confidence": 0-100,
    "explanation": "brief explanation"
}"""

    # Grounding 프롬프트 (위치 특정)
    GROUNDING_PROMPT = """이 배터리 이미지에서 결함이 있는 영역을 찾아주세요.

**중요**: 결함이 없으면 "결함 발견: 아니오"로 응답하세요.

응답 형식:
- 결함 발견: [예/아니오]
- 바운딩 박스: [[x1, y1, x2, y2], ...] 또는 []
- 결함 유형: [각 박스별 결함 유형] 또는 "없음"
- 판정: [정상/불량]"""

    @classmethod
    def get_ct_prompt(cls, detailed: bool = True) -> str:
        """CT 이미지 분석 프롬프트 반환"""
        return cls.CT_ANALYSIS if detailed else cls.CT_ANALYSIS_SIMPLE

    @classmethod
    def get_rgb_prompt(cls, detailed: bool = True) -> str:
        """RGB 이미지 분석 프롬프트 반환"""
        return cls.RGB_ANALYSIS if detailed else cls.RGB_ANALYSIS_SIMPLE

    @classmethod
    def get_prompt(cls, modality: str = 'ct', detailed: bool = True) -> str:
        """모달리티에 따른 프롬프트 반환"""
        if modality.lower() == 'ct':
            return cls.get_ct_prompt(detailed)
        elif modality.lower() == 'rgb':
            return cls.get_rgb_prompt(detailed)
        else:
            return cls.ZERO_SHOT_CLASSIFICATION

    @classmethod
    def get_unified_classes(cls) -> list:
        """통일된 5클래스 목록 반환"""
        return UNIFIED_CLASSES.copy()

    @classmethod
    def is_normal_class(cls, class_name: str) -> bool:
        """정상 클래스인지 확인"""
        return class_name in ['cell_normal', 'module_normal']

    @classmethod
    def is_defect_class(cls, class_name: str) -> bool:
        """결함 클래스인지 확인"""
        return class_name in ['cell_porosity', 'module_porosity', 'module_resin_overflow']
