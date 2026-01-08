"""결함 클래스별 상세 정보 매핑"""

# 5클래스 결함 정보 + 외관 결함
DEFECT_INFO = {
    # === CT 내부 결함 (5클래스) ===
    'cell_normal': {
        'title': '정상 셀 (Normal Cell)',
        'severity': 'SUCCESS',
        'severity_color': '#28A745',
        'description': '셀 내부 구조가 균일하며 결함이 발견되지 않았습니다.',
        'cause': '-',
        'action': '다음 공정으로 이동.',
        'icon': '✅',
        'badge_text': '정상',
    },
    'cell_porosity': {
        'title': '셀 내부 기공 결함 (Cell Porosity)',
        'severity': 'CRITICAL',
        'severity_color': '#DC3545',
        'description': '배터리 셀 내부 전극 사이에 공기 주머니(Bubble)나 빈 틈이 발견되었습니다.',
        'cause': '전해액 주입 공정 중 기포 발생 또는 전극 적층 불균형.',
        'action': '재검사 라인 이동 및 내부 밀도 정밀 분석 필요.',
        'icon': '🔴',
        'badge_text': '불량',
    },
    'module_normal': {
        'title': '정상 모듈 (Normal Module)',
        'severity': 'SUCCESS',
        'severity_color': '#28A745',
        'description': '모듈 조립 상태가 양호하며 결함이 발견되지 않았습니다.',
        'cause': '-',
        'action': '다음 공정으로 이동.',
        'icon': '✅',
        'badge_text': '정상',
    },
    'module_porosity': {
        'title': '모듈 내부 기공 결함 (Module Porosity)',
        'severity': 'CRITICAL',
        'severity_color': '#DC3545',
        'description': '모듈 내부에 기공(Porosity) 또는 빈 공간(Void)이 발견되었습니다.',
        'cause': '셀 적층 과정에서 공기 유입 또는 접합 불량.',
        'action': '재검사 라인 이동 및 모듈 분해 검사 필요.',
        'icon': '🔴',
        'badge_text': '불량',
    },
    'module_resin_overflow': {
        'title': '레진 오버플로우 (Resin Overflow)',
        'severity': 'WARNING',
        'severity_color': '#FFC107',
        'description': '모듈 고정용 수지(Resin)가 허용 범위를 벗어나 외부로 유출되었습니다.',
        'cause': '수지 도포량 과다 또는 경화 공정 중 압력 조절 실패.',
        'action': '외관 세척 후 수동 조립 간섭 여부 확인.',
        'icon': '🟠',
        'badge_text': '불량',
    },
    # === RGB 외관 결함 ===
    'external_defect': {
        'title': '외관 결함 (External Defect)',
        'severity': 'WARNING',
        'severity_color': '#FFC107',
        'description': '배터리 외관에서 오염, 손상, 스크래치 등의 이상이 발견되었습니다.',
        'cause': '제조/운반 과정에서의 물리적 충격 또는 이물질 부착.',
        'action': '외관 세척 및 손상 정도 검사 후 재판정.',
        'icon': '🟡',
        'badge_text': '외관불량',
    },
    'pollution': {
        'title': '오염 (Pollution)',
        'severity': 'WARNING',
        'severity_color': '#FFC107',
        'description': '배터리 외관에 이물질, 먼지, 기름 등의 오염이 발견되었습니다.',
        'cause': '제조 환경 오염 또는 취급 부주의.',
        'action': '외관 세척 후 재검사.',
        'icon': '🟡',
        'badge_text': '오염',
    },
    'damaged': {
        'title': '손상 (Damaged)',
        'severity': 'CRITICAL',
        'severity_color': '#DC3545',
        'description': '배터리 외관에 찍힘, 긁힘, 변형 등의 물리적 손상이 발견되었습니다.',
        'cause': '제조/운반 중 충격 또는 부적절한 취급.',
        'action': '손상 정도에 따라 폐기 또는 재가공 판정.',
        'icon': '🔴',
        'badge_text': '손상',
    },
    'scratch': {
        'title': '스크래치 (Scratch)',
        'severity': 'WARNING',
        'severity_color': '#FFC107',
        'description': '배터리 외관에 긁힘 자국이 발견되었습니다.',
        'cause': '취급 시 마찰 또는 부적절한 포장.',
        'action': '스크래치 깊이 확인 후 재판정.',
        'icon': '🟠',
        'badge_text': '스크래치',
    },
    # === 종합 판정용 ===
    'internal_defect': {
        'title': '내부 결함 (Internal Defect)',
        'severity': 'CRITICAL',
        'severity_color': '#DC3545',
        'description': '배터리 내부에서 결함이 발견되었습니다.',
        'cause': '제조 공정 중 내부 구조 이상 발생.',
        'action': '재검사 라인 이동 및 내부 정밀 분석 필요.',
        'icon': '🔬',
        'badge_text': '내부불량',
    },
    'external_defect': {
        'title': '외부 결함 (External Defect)',
        'severity': 'WARNING',
        'severity_color': '#FF6B35',
        'description': '배터리 외부에서 결함이 발견되었습니다.',
        'cause': '제조/운반 과정에서의 외부 손상 또는 오염.',
        'action': '외관 세척 및 손상 정도 검사 후 재판정.',
        'icon': '📷',
        'badge_text': '외부불량',
    },
    'complex_defect': {
        'title': '복합 결함 (Complex Defect)',
        'severity': 'CRITICAL',
        'severity_color': '#8B0000',
        'description': '배터리 내부와 외부 모두에서 결함이 발견되었습니다.',
        'cause': '제조 공정 전반의 품질 문제 또는 복합적 원인.',
        'action': '즉시 격리 및 정밀 분석 후 폐기 또는 재가공 판정.',
        'icon': '⚠️',
        'badge_text': '복합불량',
    },
}

# 심각도별 스타일
SEVERITY_STYLES = {
    'CRITICAL': {
        'color': '#DC3545',
        'bg_color': '#FFEBEE',
        'border_color': '#F44336',
        'label': '위험',
    },
    'WARNING': {
        'color': '#F57F17',
        'bg_color': '#FFF8E1',
        'border_color': '#FFC107',
        'label': '경고',
    },
    'SUCCESS': {
        'color': '#28A745',
        'bg_color': '#E8F5E9',
        'border_color': '#4CAF50',
        'label': '정상',
    },
}


def get_defect_info(class_name: str) -> dict:
    """
    클래스명으로 결함 정보 조회

    Args:
        class_name: 결함 클래스 이름 (다양한 형식 지원)

    Returns:
        결함 정보 딕셔너리
    """
    if not class_name:
        return DEFECT_INFO['module_porosity']

    # 정확히 일치하면 바로 반환
    if class_name in DEFECT_INFO:
        return DEFECT_INFO[class_name]

    # 소문자로 변환하여 매칭
    class_lower = class_name.lower()

    # 외관 결함 매핑 (다양한 표현 지원)
    external_keywords = {
        'external': 'external_defect',
        '외관': 'external_defect',
        '오염': 'pollution',
        'pollution': 'pollution',
        'contamination': 'pollution',
        '손상': 'damaged',
        'damaged': 'damaged',
        'damage': 'damaged',
        '스크래치': 'scratch',
        'scratch': 'scratch',
    }

    for keyword, defect_key in external_keywords.items():
        if keyword in class_lower:
            return DEFECT_INFO[defect_key]

    # CT 내부 결함 매핑
    internal_keywords = {
        'porosity': 'module_porosity',
        '기공': 'module_porosity',
        'void': 'module_porosity',
        'bubble': 'cell_porosity',
        'resin': 'module_resin_overflow',
        '레진': 'module_resin_overflow',
        'overflow': 'module_resin_overflow',
    }

    for keyword, defect_key in internal_keywords.items():
        if keyword in class_lower:
            return DEFECT_INFO[defect_key]

    # 기본값
    return DEFECT_INFO['module_porosity']


def get_severity_style(severity: str) -> dict:
    """
    심각도별 스타일 조회

    Args:
        severity: CRITICAL, WARNING, SUCCESS

    Returns:
        스타일 딕셔너리
    """
    return SEVERITY_STYLES.get(severity, SEVERITY_STYLES['WARNING'])


def is_normal(class_name: str) -> bool:
    """정상 클래스인지 확인"""
    return class_name in ['cell_normal', 'module_normal']


def is_defect(class_name: str) -> bool:
    """결함 클래스인지 확인"""
    return class_name in ['cell_porosity', 'module_porosity', 'module_resin_overflow']


def render_defect_card(class_name: str) -> str:
    """
    결함 정보 카드 HTML 렌더링

    Args:
        class_name: 5클래스 중 하나

    Returns:
        HTML 문자열
    """
    info = get_defect_info(class_name)
    style = get_severity_style(info['severity'])

    return f"""
    <div style="background: {style['bg_color']}; border-left: 4px solid {style['border_color']};
                border-radius: 8px; padding: 1rem; margin: 0.5rem 0;">
        <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
            <span style="font-size: 1.2rem;">{info['icon']}</span>
            <span style="font-weight: 700; color: {style['color']};">{info['title']}</span>
        </div>
        <div style="font-size: 0.9rem; color: #333; margin-bottom: 0.5rem;">
            {info['description']}
        </div>
        <div style="font-size: 0.85rem; color: #666;">
            <strong>원인:</strong> {info['cause']}
        </div>
        <div style="font-size: 0.85rem; color: #666;">
            <strong>조치:</strong> {info['action']}
        </div>
    </div>
    """


def render_severity_badge(class_name: str) -> str:
    """
    심각도 배지 HTML 렌더링

    Args:
        class_name: 5클래스 중 하나

    Returns:
        HTML 문자열
    """
    info = get_defect_info(class_name)
    style = get_severity_style(info['severity'])

    return f"""
    <span style="background: {style['color']}; color: white; padding: 0.25rem 0.75rem;
                 border-radius: 4px; font-size: 0.85rem; font-weight: 600;">
        {info['icon']} {info['badge_text']}
    </span>
    """
