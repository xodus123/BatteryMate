"""라이트 테마 커스텀 CSS - 참조 디자인 기반"""
import streamlit as st


def apply_custom_styles():
    """라이트 테마 스타일 적용"""
    st.markdown("""
    <style>
    /* 라이트 테마 배경 */
    .stApp {
        background-color: #F8F9FA;
    }

    /* 메인 컨테이너 */
    .main-container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 2rem;
    }

    /* 헤더 스타일 */
    .main-header {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 1.5rem;
    }

    /* 서브헤더 */
    .sub-header {
        font-size: 1.2rem;
        font-weight: 600;
        color: #333;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* 상태 배지 */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 4px;
        font-size: 0.85rem;
        font-weight: 600;
    }

    .badge-defect {
        background-color: #DC3545;
        color: white;
    }

    .badge-normal {
        background-color: #28A745;
        color: white;
    }

    /* 알림 박스 */
    .alert-box {
        padding: 1rem 1.5rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }

    .alert-info {
        background-color: #E3F2FD;
        border-left: 4px solid #2196F3;
        color: #1565C0;
    }

    .alert-success {
        background-color: #E8F5E9;
        border-left: 4px solid #4CAF50;
        color: #2E7D32;
    }

    .alert-warning {
        background-color: #FFF8E1;
        border-left: 4px solid #FFC107;
        color: #F57F17;
    }

    .alert-danger {
        background-color: #FFEBEE;
        border-left: 4px solid #F44336;
        color: #C62828;
    }

    /* 카드 스타일 */
    .card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        border: 1px solid #E9ECEF;
        margin-bottom: 1rem;
    }

    .card-header {
        font-size: 1rem;
        font-weight: 600;
        color: #333;
        margin-bottom: 0.75rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #E9ECEF;
    }

    .card-image {
        width: 100%;
        border-radius: 8px;
        margin-bottom: 0.75rem;
    }

    .card-label {
        font-size: 0.85rem;
        color: #666;
        text-align: center;
    }

    /* 결과 카드 (3-Way) */
    .result-card {
        background: white;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s, box-shadow 0.2s;
    }

    .result-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.15);
    }

    .result-card-image {
        width: 100%;
        height: 200px;
        object-fit: cover;
        background: #1a1a2e;
    }

    .result-card-body {
        padding: 1rem;
    }

    .result-card-title {
        font-size: 0.9rem;
        font-weight: 600;
        color: #333;
        margin-bottom: 0.25rem;
    }

    .result-card-subtitle {
        font-size: 0.8rem;
        color: #666;
    }

    /* 모델별 결과 박스 (CNN/VLM/VLG 구분) */
    .model-box {
        background: white;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.08);
        transition: transform 0.2s, box-shadow 0.2s;
        border: 2px solid transparent;
    }

    .model-box:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.12);
    }

    .model-box-header {
        padding: 0.75rem 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .model-box-header-icon {
        font-size: 1.2rem;
    }

    .model-box-header-title {
        font-size: 0.95rem;
        font-weight: 700;
        color: white;
    }

    .model-box-header-subtitle {
        font-size: 0.75rem;
        color: rgba(255, 255, 255, 0.8);
        margin-left: auto;
    }

    .model-box-content {
        padding: 0.75rem;
        background: #FAFAFA;
    }

    .model-box-footer {
        padding: 0.75rem 1rem;
        background: white;
        border-top: 1px solid #E9ECEF;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .model-box-result {
        font-size: 0.9rem;
        font-weight: 600;
    }

    .model-box-confidence {
        font-size: 0.85rem;
        color: #666;
    }

    /* Inspector 박스 (파랑) */
    .model-box-inspector {
        border-color: #2196F3;
    }

    .model-box-inspector .model-box-header {
        background: linear-gradient(135deg, #2196F3 0%, #1976D2 100%);
    }

    .model-box-inspector .model-box-result.defect {
        color: #DC3545;
    }

    .model-box-inspector .model-box-result.normal {
        color: #28A745;
    }

    /* VLM 박스 (보라) */
    .model-box-vlm {
        border-color: #9C27B0;
    }

    .model-box-vlm .model-box-header {
        background: linear-gradient(135deg, #9C27B0 0%, #7B1FA2 100%);
    }

    .model-box-vlm .model-box-result.defect {
        color: #DC3545;
    }

    .model-box-vlm .model-box-result.normal {
        color: #28A745;
    }

    /* VLG 박스 (주황) */
    .model-box-vlg {
        border-color: #FF9800;
    }

    .model-box-vlg .model-box-header {
        background: linear-gradient(135deg, #FF9800 0%, #F57C00 100%);
    }

    .model-box-vlg .model-box-result.defect {
        color: #DC3545;
    }

    .model-box-vlg .model-box-result.normal {
        color: #28A745;
    }

    /* 시스템 상세 카드 */
    .system-card {
        background: white;
        border-radius: 12px;
        padding: 1.25rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        border: 1px solid #E9ECEF;
        height: 100%;
    }

    .system-title {
        font-size: 1rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.25rem;
    }

    .system-subtitle {
        font-size: 0.8rem;
        color: #888;
        margin-bottom: 1rem;
    }

    /* 메트릭 표시 */
    .metric-large {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1a1a2e;
    }

    .metric-label {
        font-size: 0.85rem;
        color: #666;
        margin-top: 0.25rem;
    }

    .metric-badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        font-size: 0.8rem;
        background: #E3F2FD;
        color: #1565C0;
    }

    /* AI 설명 박스 */
    .ai-description {
        background: #F8F9FA;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.75rem 0;
    }

    .ai-description-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #666;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .ai-result {
        font-size: 1.5rem;
        font-weight: 700;
        color: #DC3545;
    }

    .ai-result-normal {
        color: #28A745;
    }

    /* 테이블 스타일 */
    .info-table {
        width: 100%;
        font-size: 0.85rem;
    }

    .info-table td {
        padding: 0.5rem 0;
        border-bottom: 1px solid #E9ECEF;
    }

    .info-table td:first-child {
        color: #666;
        width: 40%;
    }

    .info-table td:last-child {
        color: #333;
        font-weight: 500;
    }

    /* 검출 결과 테이블 */
    .detection-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.85rem;
        margin-top: 0.5rem;
    }

    .detection-table th {
        background: #F8F9FA;
        padding: 0.5rem;
        text-align: left;
        font-weight: 600;
        color: #333;
        border-bottom: 2px solid #E9ECEF;
    }

    .detection-table td {
        padding: 0.5rem;
        border-bottom: 1px solid #E9ECEF;
    }

    /* 최종 판정 표 */
    .verdict-section {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        margin-top: 2rem;
    }

    .verdict-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #DC3545;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .verdict-item {
        padding: 1rem;
        background: #F8F9FA;
        border-radius: 8px;
        margin-bottom: 0.75rem;
    }

    .verdict-item-title {
        font-size: 0.9rem;
        font-weight: 600;
        color: #333;
        margin-bottom: 0.5rem;
    }

    .verdict-item-content {
        font-size: 0.85rem;
        color: #666;
        line-height: 1.6;
    }

    /* 버튼 스타일 */
    .stButton > button {
        background: #1a1a2e;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: background 0.2s;
    }

    .stButton > button:hover {
        background: #2d2d44;
    }

    /* 프로그레스 바 */
    .stProgress > div > div {
        background-color: #4CAF50;
    }

    /* 사이드바 숨김 */
    [data-testid="stSidebar"] {
        display: none;
    }

    /* 이미지 컨테이너 */
    .image-container {
        background: #1a1a2e;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }

    .image-caption {
        color: #4ECDC4;
        font-size: 0.85rem;
        margin-top: 0.75rem;
    }

    /* 스크롤바 */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }

    ::-webkit-scrollbar-track {
        background: #F1F1F1;
    }

    ::-webkit-scrollbar-thumb {
        background: #C1C1C1;
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #A1A1A1;
    }

    /* Expander 스타일 */
    .streamlit-expanderHeader {
        background-color: #F8F9FA;
        border-radius: 8px;
    }

    /* 상세 정보 섹션 */
    .detail-section {
        margin-bottom: 1rem;
        padding-bottom: 0.75rem;
        border-bottom: 1px solid #E9ECEF;
    }

    .detail-label {
        font-size: 0.8rem;
        color: #888;
        margin-bottom: 0.25rem;
    }

    .detail-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1a1a2e;
    }

    .detail-value-small {
        font-size: 0.95rem;
        font-weight: 600;
        color: #333;
    }

    /* 상세 테이블 */
    .detail-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.85rem;
        margin-top: 0.75rem;
    }

    .detail-table th {
        background: #F8F9FA;
        padding: 0.6rem 0.75rem;
        text-align: left;
        font-weight: 600;
        color: #333;
        border-bottom: 2px solid #E9ECEF;
    }

    .detail-table td {
        padding: 0.6rem 0.75rem;
        border-bottom: 1px solid #E9ECEF;
        color: #555;
    }

    .detail-table tr:last-child td {
        border-bottom: none;
    }

    .detail-table tr:hover {
        background: #FAFAFA;
    }
    </style>
    """, unsafe_allow_html=True)


def render_alert(message: str, alert_type: str = "info", icon: str = None):
    """알림 박스 렌더링"""
    icons = {
        "info": "ℹ️",
        "success": "✨",
        "warning": "⚠️",
        "danger": "🚨"
    }
    icon = icon or icons.get(alert_type, "ℹ️")

    return f"""
    <div class="alert-box alert-{alert_type}">
        <span>{icon}</span>
        <span>{message}</span>
    </div>
    """


def render_status_badge(text: str, is_defect: bool = True):
    """상태 배지 렌더링"""
    badge_class = "badge-defect" if is_defect else "badge-normal"
    return f'<span class="status-badge {badge_class}">{text}</span>'


def render_system_card(title: str, subtitle: str, content: str):
    """시스템 상세 카드 렌더링"""
    return f"""
    <div class="system-card">
        <div class="system-title">{title}</div>
        <div class="system-subtitle">{subtitle}</div>
        {content}
    </div>
    """
