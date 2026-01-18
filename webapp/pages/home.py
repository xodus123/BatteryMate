"""Page 1: Home - 이미지 업로드 (CT + RGB 통합 검사 지원)"""
import streamlit as st
from PIL import Image
import io

from webapp.utils.session import set_uploaded_images, navigate_to


def render():
    """홈 페이지 렌더링"""

    # 헤더
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 2rem;">
        <span style="font-size: 1.5rem;">🔋</span>
        <span style="font-size: 1.5rem; font-weight: 600; color: #1a1a2e;">
            Battery Defect Multi-Analysis Dashboard
        </span>
    </div>
    """, unsafe_allow_html=True)

    # 안내 문구
    st.markdown("""
    <div class="card" style="padding: 1rem 1.5rem; margin-bottom: 1.5rem; background: #f8f9fa;">
        <div style="color: #333; font-size: 0.95rem;">
            <strong>📌 업로드 안내</strong><br>
            • <b>CT + RGB 둘 다</b>: 통합 검사 (내부 + 외부 결함 종합 판정)<br>
            • <b>CT만</b>: 내부 결함 분석 (기공, 레진 오버플로우)<br>
            • <b>RGB만</b>: 외부 결함 분석 (오염, 손상)
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 두 개의 업로드 영역
    col1, col2 = st.columns(2)

    ct_image = None
    rgb_image = None
    ct_filename = None
    rgb_filename = None

    # CT 이미지 업로드
    with col1:
        st.markdown("""
        <div class="card" style="text-align: center; padding: 1.5rem;">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">🔬</div>
            <div style="font-weight: 600; color: #1a1a2e; margin-bottom: 0.5rem;">CT 이미지</div>
            <div style="color: #666; font-size: 0.8rem; margin-bottom: 1rem;">내부 결함 검사 (X-ray)</div>
        </div>
        """, unsafe_allow_html=True)

        ct_file = st.file_uploader(
            "CT 이미지 선택",
            type=['png', 'jpg', 'jpeg', 'tif', 'tiff', 'bmp'],
            key="ct_uploader",
            label_visibility="collapsed"
        )

        if ct_file is not None:
            ct_image = Image.open(ct_file)
            ct_filename = ct_file.name
            st.image(ct_image, caption=f"CT: {ct_filename}", width="stretch")
            st.markdown(f"<div style='color:#666; font-size:0.8rem; text-align:center;'>{ct_image.size[0]}x{ct_image.size[1]} px</div>", unsafe_allow_html=True)

    # RGB 이미지 업로드
    with col2:
        st.markdown("""
        <div class="card" style="text-align: center; padding: 1.5rem;">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">📷</div>
            <div style="font-weight: 600; color: #1a1a2e; margin-bottom: 0.5rem;">RGB 이미지</div>
            <div style="color: #666; font-size: 0.8rem; margin-bottom: 1rem;">외부 결함 검사 (카메라)</div>
        </div>
        """, unsafe_allow_html=True)

        rgb_file = st.file_uploader(
            "RGB 이미지 선택",
            type=['png', 'jpg', 'jpeg', 'tif', 'tiff', 'bmp'],
            key="rgb_uploader",
            label_visibility="collapsed"
        )

        if rgb_file is not None:
            rgb_image = Image.open(rgb_file)
            rgb_filename = rgb_file.name
            st.image(rgb_image, caption=f"RGB: {rgb_filename}", width="stretch")
            st.markdown(f"<div style='color:#666; font-size:0.8rem; text-align:center;'>{rgb_image.size[0]}x{rgb_image.size[1]} px</div>", unsafe_allow_html=True)

    # 분석 모드 표시
    st.markdown("<br>", unsafe_allow_html=True)

    if ct_image is not None and rgb_image is not None:
        analysis_mode = "inspector"
        mode_text = "🔗 <b>통합 검사</b> - CT (내부) + RGB (외부) 종합 판정"
        mode_color = "#28a745"
    elif ct_image is not None:
        analysis_mode = "ct_only"
        mode_text = "🔬 <b>CT 분석</b> - 내부 결함 검사만 수행"
        mode_color = "#007bff"
    elif rgb_image is not None:
        analysis_mode = "rgb_only"
        mode_text = "📷 <b>RGB 분석</b> - 외부 결함 검사만 수행"
        mode_color = "#17a2b8"
    else:
        analysis_mode = None
        mode_text = "⬆️ 이미지를 업로드하세요"
        mode_color = "#6c757d"

    st.markdown(f"""
    <div style="text-align: center; padding: 1rem; background: {mode_color}15; border-radius: 8px; border: 1px solid {mode_color}30;">
        <span style="color: {mode_color}; font-size: 1rem;">{mode_text}</span>
    </div>
    """, unsafe_allow_html=True)

    # 고급 설정 (VLM/VLG 모델 선택)
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("⚙️ 고급 설정", expanded=False):
        col_setting1, col_setting2 = st.columns(2)

        # VLM 모델 선택
        with col_setting1:
            vlm_options = {
                'qwen2vl': '🧠 Qwen2-VL (로컬)',
                'gemini': '☁️ Gemini 2.0 Flash (API)'
            }
            current_vlm = st.session_state.get('vlm_model_type', 'qwen2vl')
            selected_vlm = st.selectbox(
                "VLM 모델 선택",
                options=list(vlm_options.keys()),
                format_func=lambda x: vlm_options[x],
                index=0 if current_vlm == 'qwen2vl' else 1,
                key="vlm_model_selector"
            )
            st.session_state.vlm_model_type = selected_vlm

        # VLG 모델 선택
        with col_setting2:
            vlg_options = {
                'groundingdino': '🎯 GroundingDINO (Swin-T)',
                'yoloworld': '🚀 YOLO-World (YOLOv8s)'
            }
            current_vlg = st.session_state.get('vlg_model_type', 'groundingdino')
            selected_vlg = st.selectbox(
                "VLG 모델 선택",
                options=list(vlg_options.keys()),
                format_func=lambda x: vlg_options[x],
                index=0 if current_vlg == 'groundingdino' else 1,
                key="vlg_model_selector"
            )
            st.session_state.vlg_model_type = selected_vlg

        st.markdown("""
        <div style="font-size: 0.85rem; color: #666; margin-top: 1rem;">
            <b>VLM</b>: Qwen2-VL (로컬 GPU) vs Gemini (Google API, 빠름)<br>
            <b>VLG</b>: GroundingDINO (정확) vs YOLO-World (빠름)
        </div>
        """, unsafe_allow_html=True)

    # 분석 시작 버튼
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        button_disabled = (ct_image is None and rgb_image is None)

        if st.button("🔍 분석 시작", width="stretch", disabled=button_disabled):
            # 이미지 데이터 저장
            ct_bytes = None
            rgb_bytes = None

            if ct_image is not None:
                ct_buf = io.BytesIO()
                ct_image.save(ct_buf, format='PNG')
                ct_buf.seek(0)
                ct_bytes = ct_buf.getvalue()

            if rgb_image is not None:
                rgb_buf = io.BytesIO()
                rgb_image.save(rgb_buf, format='PNG')
                rgb_buf.seek(0)
                rgb_bytes = rgb_buf.getvalue()

            set_uploaded_images(
                ct_image_data=ct_bytes,
                ct_filename=ct_filename,
                rgb_image_data=rgb_bytes,
                rgb_filename=rgb_filename,
                analysis_mode=analysis_mode
            )

            # Processing 페이지로 이동
            navigate_to('processing')

    # 데모 섹션
    st.markdown("<hr style='border: none; border-top: 1px solid #E9ECEF; margin: 2rem 0;'>", unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align: center; color: #666; margin: 1rem 0;">
        또는 데모 이미지로 바로 시작하기
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🔬 CT 데모", width="stretch"):
            _load_demo_image(mode='ct_only')
    with col2:
        if st.button("📷 RGB 데모", width="stretch"):
            _load_demo_image(mode='rgb_only')
    with col3:
        if st.button("🔗 통합 검사 데모", width="stretch"):
            _load_demo_image(mode='inspector')


def _load_demo_image(mode: str = 'ensemble'):
    """데모 이미지 로드"""
    import numpy as np

    ct_bytes = None
    rgb_bytes = None
    ct_filename = None
    rgb_filename = None

    # CT 데모 이미지 생성
    if mode in ['ct_only', 'inspector']:
        width, height = 512, 512
        img_array = np.ones((height, width, 3), dtype=np.uint8) * 40

        center_x, center_y = width // 2, height // 2
        radius = 150

        for y in range(height):
            for x in range(width):
                dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
                if dist < radius:
                    intensity = int(180 - (dist / radius) * 60)
                    img_array[y, x] = [intensity, intensity, intensity]

        # 기공 효과 추가 (작은 어두운 점들)
        np.random.seed(42)
        for _ in range(5):
            px = int(center_x + np.random.uniform(-100, 100))
            py = int(center_y + np.random.uniform(-100, 100))
            for dy in range(-8, 9):
                for dx in range(-8, 9):
                    if dx*dx + dy*dy < 64:
                        ny, nx = py + dy, px + dx
                        if 0 <= ny < height and 0 <= nx < width:
                            img_array[ny, nx] = [30, 30, 30]

        ct_image = Image.fromarray(img_array, 'RGB')
        ct_buf = io.BytesIO()
        ct_image.save(ct_buf, format='PNG')
        ct_buf.seek(0)
        ct_bytes = ct_buf.getvalue()
        ct_filename = "demo_battery_ct.png"

    # RGB 데모 이미지 생성
    if mode in ['rgb_only', 'inspector']:
        width, height = 512, 512
        img_array = np.ones((height, width, 3), dtype=np.uint8) * 200  # 밝은 배경

        # 배터리 외관 (사각형)
        img_array[100:400, 100:400] = [180, 180, 190]

        # 오염 효과 추가 (갈색 얼룩)
        np.random.seed(43)
        for _ in range(3):
            px = int(np.random.uniform(150, 350))
            py = int(np.random.uniform(150, 350))
            for dy in range(-20, 21):
                for dx in range(-20, 21):
                    if dx*dx + dy*dy < 400:
                        ny, nx = py + dy, px + dx
                        if 100 <= ny < 400 and 100 <= nx < 400:
                            img_array[ny, nx] = [139, 90, 43]  # 갈색

        rgb_image = Image.fromarray(img_array, 'RGB')
        rgb_buf = io.BytesIO()
        rgb_image.save(rgb_buf, format='PNG')
        rgb_buf.seek(0)
        rgb_bytes = rgb_buf.getvalue()
        rgb_filename = "demo_battery_rgb.png"

    set_uploaded_images(
        ct_image_data=ct_bytes,
        ct_filename=ct_filename,
        rgb_image_data=rgb_bytes,
        rgb_filename=rgb_filename,
        analysis_mode=mode
    )

    navigate_to('processing')
