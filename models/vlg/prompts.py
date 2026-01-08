"""GroundingDINO용 텍스트 프롬프트 정의 - 5클래스 통일"""


# 통일된 5클래스 정의
UNIFIED_CLASSES = [
    'cell_normal',
    'cell_porosity',
    'module_normal',
    'module_porosity',
    'module_resin_overflow',
]


class GroundingPrompts:
    """결함 탐지를 위한 텍스트 프롬프트 - 5클래스 통일"""

    # ============================================================
    # 5클래스별 탐지 키워드
    # ============================================================

    # cell_porosity 탐지용
    CELL_POROSITY = [
        "cell porosity",
        "cell void",
        "cell bubble",
        "small porosity",
        "internal void",
    ]

    # module_porosity 탐지용
    MODULE_POROSITY = [
        "module porosity",
        "large porosity",
        "module void",
        "big bubble",
        "module defect",
    ]

    # module_resin_overflow 탐지용
    MODULE_RESIN_OVERFLOW = [
        "resin overflow",
        "resin leakage",
        "adhesive overflow",
        "excess resin",
        "resin defect",
    ]

    # 전체 결함 탐지용 (porosity + resin)
    ALL_DEFECTS = [
        "porosity",
        "void",
        "bubble",
        "resin overflow",
        "defect",
    ]

    # ============================================================
    # 키워드 → 5클래스 매핑
    # ============================================================

    KEYWORD_TO_CLASS = {
        # cell_porosity 매핑
        'cell porosity': 'cell_porosity',
        'cell void': 'cell_porosity',
        'cell bubble': 'cell_porosity',
        'small porosity': 'cell_porosity',
        'internal void': 'cell_porosity',

        # module_porosity 매핑
        'module porosity': 'module_porosity',
        'large porosity': 'module_porosity',
        'module void': 'module_porosity',
        'big bubble': 'module_porosity',
        'module defect': 'module_porosity',

        # module_resin_overflow 매핑
        'resin overflow': 'module_resin_overflow',
        'resin leakage': 'module_resin_overflow',
        'adhesive overflow': 'module_resin_overflow',
        'excess resin': 'module_resin_overflow',
        'resin defect': 'module_resin_overflow',

        # 일반 키워드 → 기본 매핑 (module_porosity)
        'porosity': 'module_porosity',
        'void': 'module_porosity',
        'bubble': 'module_porosity',
        'defect': 'module_porosity',
        'resin': 'module_resin_overflow',

        # === RGB 외관 결함 매핑 ===
        'pollution': 'pollution',
        'contamination': 'pollution',
        'stain': 'pollution',
        'dirty': 'pollution',
        'scratch': 'scratch',
        'scratched': 'scratch',
        'damage': 'damaged',
        'damaged': 'damaged',
        'dent': 'damaged',
        'crack': 'damaged',
    }

    # ============================================================
    # 유틸리티 메서드
    # ============================================================

    @classmethod
    def get_all_prompts(cls) -> list:
        """전체 결함 탐지 프롬프트"""
        return cls.ALL_DEFECTS

    @classmethod
    def get_prompts_by_class(cls, class_name: str) -> list:
        """특정 클래스 탐지 프롬프트"""
        prompts_map = {
            'cell_porosity': cls.CELL_POROSITY,
            'module_porosity': cls.MODULE_POROSITY,
            'module_resin_overflow': cls.MODULE_RESIN_OVERFLOW,
        }
        return prompts_map.get(class_name, cls.ALL_DEFECTS)

    @classmethod
    def to_grounding_text(cls, prompts: list = None, separator: str = ' . ') -> str:
        """
        프롬프트 리스트를 GroundingDINO 입력 형식으로 변환
        """
        if prompts is None:
            prompts = cls.ALL_DEFECTS
        return separator.join(prompts)

    @classmethod
    def map_to_unified_class(cls, detected_label: str) -> str:
        """
        탐지된 라벨을 결함 클래스로 매핑

        Args:
            detected_label: VLG가 탐지한 라벨

        Returns:
            결함 클래스 (CT: 5클래스, RGB: 외관 결함)
        """
        label_lower = detected_label.lower().strip()

        # 직접 매핑 확인
        if label_lower in cls.KEYWORD_TO_CLASS:
            return cls.KEYWORD_TO_CLASS[label_lower]

        # 부분 매칭 - 외관 결함 (RGB)
        if any(kw in label_lower for kw in ['pollution', 'contamination', 'stain', 'dirty']):
            return 'pollution'
        elif any(kw in label_lower for kw in ['scratch', 'scratched']):
            return 'scratch'
        elif any(kw in label_lower for kw in ['damage', 'damaged', 'dent', 'crack']):
            return 'damaged'

        # 부분 매칭 - 내부 결함 (CT)
        if 'resin' in label_lower:
            return 'module_resin_overflow'
        elif 'cell' in label_lower:
            return 'cell_porosity'
        elif any(kw in label_lower for kw in ['porosity', 'void', 'bubble']):
            return 'module_porosity'

        # 기본값 (매핑 실패 시 원본 라벨 반환)
        return detected_label

    @classmethod
    def get_unified_classes(cls) -> list:
        """통일된 5클래스 목록 반환"""
        return UNIFIED_CLASSES.copy()


# 모듈 레벨 헬퍼 함수 (직접 import 가능)
def map_to_unified_class(detected_label: str, modality: str = 'ct') -> str:
    """
    탐지된 라벨을 통합 클래스로 매핑 (모듈 레벨 함수)

    Args:
        detected_label: VLG가 탐지한 라벨
        modality: 'ct' 또는 'rgb'

    Returns:
        통합 결함 클래스
    """
    return GroundingPrompts.map_to_unified_class(detected_label)
