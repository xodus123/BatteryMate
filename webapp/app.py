"""배터리 결함 분석 시스템 - Streamlit 메인 앱"""
import streamlit as st
from pathlib import Path
import sys

# 프로젝트 루트 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from webapp.utils.session import init_session_state
from webapp.utils.styles import apply_custom_styles

# 페이지 설정
st.set_page_config(
    page_title="Battery Defect Multi-Analysis Dashboard",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# 커스텀 스타일 적용
apply_custom_styles()

# 세션 상태 초기화
init_session_state()


def main():
    """메인 앱 - 페이지 라우팅"""

    # 현재 페이지 상태에 따라 렌더링
    page = st.session_state.get('current_page', 'home')

    if page == 'home':
        from webapp.pages import home
        home.render()
    elif page == 'processing':
        from webapp.pages import processing
        processing.render()
    elif page == 'summary':
        from webapp.pages import summary
        summary.render()
    else:
        # 기본값: 홈
        from webapp.pages import home
        home.render()


if __name__ == "__main__":
    main()
