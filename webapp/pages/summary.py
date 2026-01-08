"""Page 3: Summary - 3-Way Analysis Comparison (CT + RGB 앙상블 지원)"""
import streamlit as st
from PIL import Image, ImageDraw, ImageFilter
import io
import base64
import numpy as np

from webapp.utils.session import navigate_to, get_analysis_result
from webapp.utils.styles import render_alert, render_status_badge
from webapp.utils.defect_info import (
    get_defect_info, get_severity_style, is_normal, is_defect,
    render_defect_card, render_severity_badge, DEFECT_INFO
)


def _pil_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """PIL 이미지를 base64 문자열로 변환"""
    buffered = io.BytesIO()
    image.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue()).decode()


def render():
    """요약 페이지 렌더링 - 3-Way 비교"""

    # 분석 결과 없으면 홈으로
    if not st.session_state.analysis_complete:
        navigate_to('home')
        return

    # 이미지 가져오기
    ct_image = st.session_state.get('ct_image')
    rgb_image = st.session_state.get('rgb_image')
    analysis_mode = st.session_state.get('analysis_mode', 'ct_only')

    # 헤더
    st.markdown("""
    <div class="main-header">🔍 3-Way Analysis Comparison</div>
    """, unsafe_allow_html=True)

    # 결과 가져오기
    ensemble_result = get_analysis_result('ensemble')
    vlm_result = get_analysis_result('vlm')
    vlg_result = get_analysis_result('vlg')

    # 불량 여부 확인
    is_defect_flag = any([
        ensemble_result and ensemble_result.prediction not in ['normal', 'unknown', 'error'],
        vlm_result and vlm_result.prediction not in ['normal', 'unknown', 'error'],
        vlg_result and vlg_result.prediction not in ['normal', 'unknown', 'error'],
    ])

    # 상태 배지
    badge_text = "불량 부위 검출 결과" if is_defect_flag else "정상 판정 결과"
    st.markdown(f"""
    <div style="margin-bottom: 1.5rem;">
        {render_status_badge(badge_text, is_defect_flag)}
    </div>
    """, unsafe_allow_html=True)

    # PIL 이미지 준비
    ct_pil = Image.open(io.BytesIO(ct_image)) if ct_image else None
    rgb_pil = Image.open(io.BytesIO(rgb_image)) if rgb_image else None

    # 3-Way 결과 카드
    col1, col2, col3 = st.columns(3)

    with col1:
        _render_ensemble_card(ct_pil, rgb_pil, ensemble_result, analysis_mode)

    with col2:
        _render_vlm_card(ct_pil, rgb_pil, vlm_result, analysis_mode)

    with col3:
        _render_vlg_card(ct_pil, rgb_pil, vlg_result, analysis_mode)

    # 주의 메시지
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        render_alert(
            "이 시스템의 판단 결과만 맹신하지 마십시오. 이는 참고자료이며 최종 판단은 전문가의 판단에 따릅니다.",
            "warning", "⚠️"
        ),
        unsafe_allow_html=True
    )

    st.markdown("<hr style='border: none; border-top: 1px solid #E9ECEF; margin: 2rem 0;'>", unsafe_allow_html=True)

    # 3개 시스템 상세 결과
    st.markdown("""
    <div class="sub-header">📊 시스템별 상세 결과</div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        _render_ensemble_detail(ensemble_result, ct_pil, rgb_pil, analysis_mode)

    with col2:
        _render_vlm_detail(vlm_result, analysis_mode)

    with col3:
        _render_vlg_detail(vlg_result, analysis_mode)

    # 최종 판정 표
    _render_verdict_section(ensemble_result, vlm_result, vlg_result)

    # 하단 버튼
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        if st.button("🔄 새 이미지 분석", width="stretch"):
            from webapp.utils.session import reset_analysis
            reset_analysis()
            navigate_to('home')

    with col2:
        if st.button("📥 리포트 다운로드", width="stretch"):
            _download_report()


def _render_ensemble_card(ct_pil, rgb_pil, result, analysis_mode):
    """앙상블 결과 카드 (파란색 박스) - st.image() 사용"""
    # 에러 상태 확인
    is_error = result and result.prediction == 'error'

    if is_error:
        error_msg = result.details.get('error', '모델 로드 실패') if result.details else '모델 로드 실패'
        st.markdown(f"""
        <div class="model-box model-box-ensemble" style="border-color: #999;">
            <div class="model-box-header" style="background: linear-gradient(135deg, #999 0%, #777 100%);">
                <span class="model-box-header-icon">🔬</span>
                <span class="model-box-header-title">Ensemble (CNN+AE)</span>
                <span class="model-box-header-subtitle">오류</span>
            </div>
            <div class="model-box-content" style="padding: 2rem; text-align: center;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">⚠️</div>
                <div style="color: #DC3545; font-weight: 600;">모델 로드 실패</div>
                <div style="font-size: 0.8rem; color: #666; margin-top: 0.5rem; word-break: break-all;">{error_msg[:100]}...</div>
            </div>
            <div class="model-box-footer">
                <span class="model-box-result" style="color: #999;">❌ 에러</span>
                <span class="model-box-confidence">-</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    # 결과 정보 추출
    is_defect_flag = False
    confidence = 0.0
    verdict_text = "정상"
    visualizations = None

    if result:
        is_defect_flag = result.prediction not in ['normal', 'unknown', 'error']
        confidence = result.confidence
        if result.details:
            verdict_text = result.details.get('verdict', '정상')
            visualizations = result.details.get('visualizations')

    result_class = "defect" if is_defect_flag else "normal"
    result_icon = "🔴" if is_defect_flag else "✅"

    # 헤더
    st.markdown(f"""
    <div class="model-box model-box-ensemble">
        <div class="model-box-header">
            <span class="model-box-header-title">Ensemble (CNN+AE)</span>
            <span class="model-box-header-subtitle">내부+외부 통합</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 이미지 표시 (st.image 사용)
    if visualizations:
        if analysis_mode == 'ensemble':
            col_ct, col_rgb = st.columns(2)
            ct_overlay = visualizations.get('ct_gradcam_overlay')
            rgb_error = visualizations.get('rgb_error_map')

            with col_ct:
                if ct_overlay is not None:
                    ct_img = Image.fromarray((ct_overlay).astype(np.uint8))
                    st.image(ct_img, caption="CT Grad-CAM", width="stretch")
                elif ct_pil:
                    st.image(ct_pil, caption="CT 원본", width="stretch")

            with col_rgb:
                if rgb_error is not None:
                    import cv2
                    rgb_orig = visualizations.get('rgb_original')
                    if rgb_orig is not None:
                        rgb_orig_uint8 = (rgb_orig * 255).astype(np.uint8) if rgb_orig.max() <= 1.0 else rgb_orig.astype(np.uint8)
                        error_uint8 = (rgb_error * 255).astype(np.uint8)
                        error_colored = cv2.applyColorMap(error_uint8, cv2.COLORMAP_JET)
                        error_colored = cv2.cvtColor(error_colored, cv2.COLOR_BGR2RGB)
                        overlay_rgb = cv2.addWeighted(rgb_orig_uint8, 0.6, error_colored, 0.4, 0)
                        rgb_img = Image.fromarray(overlay_rgb)
                        st.image(rgb_img, caption="RGB Error Map", width="stretch")
                elif rgb_pil:
                    st.image(rgb_pil, caption="RGB 원본", width="stretch")

        elif analysis_mode == 'ct_only':
            ct_overlay = visualizations.get('ct_gradcam_overlay')
            if ct_overlay is not None:
                ct_img = Image.fromarray((ct_overlay).astype(np.uint8))
                st.image(ct_img, caption="CT Grad-CAM", width="stretch")
            elif ct_pil:
                st.image(ct_pil, caption="CT 원본", width="stretch")

        elif analysis_mode == 'rgb_only':
            rgb_error = visualizations.get('rgb_error_map')
            rgb_orig = visualizations.get('rgb_original')
            if rgb_error is not None and rgb_orig is not None:
                import cv2
                rgb_orig_uint8 = (rgb_orig * 255).astype(np.uint8) if rgb_orig.max() <= 1.0 else rgb_orig.astype(np.uint8)
                error_uint8 = (rgb_error * 255).astype(np.uint8)
                error_colored = cv2.applyColorMap(error_uint8, cv2.COLORMAP_JET)
                error_colored = cv2.cvtColor(error_colored, cv2.COLOR_BGR2RGB)
                overlay_rgb = cv2.addWeighted(rgb_orig_uint8, 0.6, error_colored, 0.4, 0)
                rgb_img = Image.fromarray(overlay_rgb)
                st.image(rgb_img, caption="RGB Error Map", width="stretch")
            elif rgb_pil:
                st.image(rgb_pil, caption="RGB 원본", width="stretch")
    else:
        # 시각화 데이터 없으면 원본 이미지 표시
        if ct_pil and rgb_pil:
            col_ct, col_rgb = st.columns(2)
            with col_ct:
                st.image(ct_pil, caption="CT 원본", width="stretch")
            with col_rgb:
                st.image(rgb_pil, caption="RGB 원본", width="stretch")
        elif ct_pil:
            st.image(ct_pil, caption="CT 원본", width="stretch")
        elif rgb_pil:
            st.image(rgb_pil, caption="RGB 원본", width="stretch")

    # 범례 (Grad-CAM)
    st.markdown("""
    <div style="display: flex; justify-content: center; align-items: center; margin: 8px 0; gap: 4px;">
        <span style="font-size: 0.7rem; color: #666;">낮음</span>
        <div style="width: 100px; height: 10px; background: linear-gradient(to right, #0000FF, #00FFFF, #00FF00, #FFFF00, #FF0000); border-radius: 3px;"></div>
        <span style="font-size: 0.7rem; color: #666;">높음</span>
    </div>
    """, unsafe_allow_html=True)

    # 푸터
    st.markdown(f"""
    <div class="model-box-footer" style="background: #f8f9fa; padding: 0.75rem; border-radius: 0 0 8px 8px; display: flex; justify-content: space-between; border-top: 1px solid #e9ecef;">
        <span class="model-box-result {result_class}" style="font-weight: 600; color: {'#DC3545' if is_defect_flag else '#28A745'};">{result_icon} {verdict_text}</span>
        <span class="model-box-confidence" style="color: #666;">신뢰도: {confidence:.1%}</span>
    </div>
    """, unsafe_allow_html=True)


def _render_vlm_card(ct_pil, rgb_pil, result, analysis_mode):
    """VLM 결과 카드 (보라색 박스) - st.image() 사용"""
    # VLM 모델명 동적 결정
    vlm_model = 'Qwen2-VL'
    if result and result.details:
        vlm_type = result.details.get('vlm_model', 'qwen2vl')
        vlm_model = 'Gemini' if vlm_type == 'gemini' else 'Qwen2-VL'

    # 에러 상태 확인
    is_error = result and result.prediction == 'error'

    if is_error:
        error_msg = result.details.get('error', '모델 로드 실패') if result.details else '모델 로드 실패'
        st.markdown(f"""
        <div class="model-box model-box-vlm" style="border-color: #999;">
            <div class="model-box-header" style="background: linear-gradient(135deg, #999 0%, #777 100%);">
                <span class="model-box-header-icon">🤖</span>
                <span class="model-box-header-title">VLM ({vlm_model})</span>
                <span class="model-box-header-subtitle">오류</span>
            </div>
            <div class="model-box-content" style="padding: 2rem; text-align: center;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">⚠️</div>
                <div style="color: #DC3545; font-weight: 600;">모델 로드 실패</div>
                <div style="font-size: 0.8rem; color: #666; margin-top: 0.5rem; word-break: break-all;">{error_msg[:100]}...</div>
            </div>
            <div class="model-box-footer">
                <span class="model-box-result" style="color: #999;">❌ 에러</span>
                <span class="model-box-confidence">-</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    # 결과 정보 추출
    is_defect_flag = False
    confidence = 0.0
    verdict_text = "정상"

    if result:
        is_defect_flag = result.prediction not in ['normal', 'unknown', 'error']
        confidence = result.confidence
        # 판정 텍스트 결정
        if result.prediction == 'internal_defect':
            verdict_text = "내부불량"
        elif result.prediction == 'external_defect':
            verdict_text = "외부불량"
        elif result.prediction == 'complex_defect':
            verdict_text = "복합불량"
        elif is_defect_flag:
            verdict_text = "불량"
        else:
            verdict_text = "정상"

    result_class = "defect" if is_defect_flag else "normal"
    result_icon = "🔴" if is_defect_flag else "✅"

    # 헤더
    subtitle = "Google AI API" if vlm_model == 'Gemini' else "AI 비전 분석"
    st.markdown(f"""
    <div class="model-box model-box-vlm">
        <div class="model-box-header">
            <span class="model-box-header-title">VLM ({vlm_model})</span>
            <span class="model-box-header-subtitle">{subtitle}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 이미지 표시 (st.image 사용)
    if analysis_mode == 'ensemble' and ct_pil and rgb_pil:
        col_ct, col_rgb = st.columns(2)
        with col_ct:
            ct_overlay = _generate_vlm_overlay(ct_pil, result, 'ct')
            st.image(ct_overlay, caption="CT VLM", width="stretch")
        with col_rgb:
            rgb_overlay = _generate_vlm_overlay(rgb_pil, result, 'rgb')
            st.image(rgb_overlay, caption="RGB VLM", width="stretch")
    elif ct_pil:
        overlay = _generate_vlm_overlay(ct_pil, result, 'ct')
        st.image(overlay, caption="CT VLM", width="stretch")
    elif rgb_pil:
        overlay = _generate_vlm_overlay(rgb_pil, result, 'rgb')
        st.image(overlay, caption="RGB VLM", width="stretch")

    # 푸터
    st.markdown(f"""
    <div class="model-box-footer" style="background: #f8f9fa; padding: 0.75rem; border-radius: 0 0 8px 8px; display: flex; justify-content: space-between; border-top: 1px solid #e9ecef;">
        <span class="model-box-result {result_class}" style="font-weight: 600; color: {'#DC3545' if is_defect_flag else '#28A745'};">{result_icon} {verdict_text}</span>
        <span class="model-box-confidence" style="color: #666;">신뢰도: {confidence:.1%}</span>
    </div>
    """, unsafe_allow_html=True)


def _render_vlg_card(ct_pil, rgb_pil, result, analysis_mode):
    """VLG 결과 카드 (주황색 박스) - st.image() 사용"""
    # VLG 모델 타입 확인
    vlg_model = 'GroundingDINO'
    if result and result.details:
        vlg_type = result.details.get('vlg_model', 'groundingdino')
        vlg_model = 'YOLO-World' if vlg_type == 'yoloworld' else 'GroundingDINO'

    # 에러 상태 확인
    is_error = result and result.prediction == 'error'

    if is_error:
        error_msg = result.details.get('error', '모델 로드 실패') if result.details else '모델 로드 실패'
        st.markdown(f"""
        <div class="model-box model-box-vlg" style="border-color: #999;">
            <div class="model-box-header" style="background: linear-gradient(135deg, #999 0%, #777 100%);">
                <span class="model-box-header-icon">🎯</span>
                <span class="model-box-header-title">VLG ({vlg_model})</span>
                <span class="model-box-header-subtitle">오류</span>
            </div>
            <div class="model-box-content" style="padding: 2rem; text-align: center;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">⚠️</div>
                <div style="color: #DC3545; font-weight: 600;">모델 로드 실패</div>
                <div style="font-size: 0.8rem; color: #666; margin-top: 0.5rem; word-break: break-all;">{error_msg[:100]}...</div>
            </div>
            <div class="model-box-footer">
                <span class="model-box-result" style="color: #999;">❌ 에러</span>
                <span class="model-box-confidence">-</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    # 결과 정보 추출
    is_defect_flag = False
    confidence = 0.0
    num_detections = 0

    if result:
        is_defect_flag = result.prediction not in ['normal', 'unknown', 'error']
        confidence = result.confidence
        if result.details:
            num_detections = result.details.get('num_detections', 0)

    result_class = "defect" if is_defect_flag else "normal"
    result_icon = "🔴" if is_defect_flag else "✅"

    # 판정 텍스트 결정
    if is_defect_flag:
        if result.prediction == 'internal_defect':
            verdict_text = f"내부불량 ({num_detections}개)"
        elif result.prediction == 'external_defect':
            verdict_text = f"외부불량 ({num_detections}개)"
        elif result.prediction == 'complex_defect':
            verdict_text = f"복합불량 ({num_detections}개)"
        else:
            verdict_text = f"불량 ({num_detections}개)"
    else:
        verdict_text = "정상"

    # 헤더
    st.markdown(f"""
    <div class="model-box model-box-vlg">
        <div class="model-box-header">
            <span class="model-box-header-title">VLG ({vlg_model})</span>
            <span class="model-box-header-subtitle">객체 탐지</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 이미지 표시 (st.image 사용)
    if analysis_mode == 'ensemble' and ct_pil and rgb_pil:
        col_ct, col_rgb = st.columns(2)
        with col_ct:
            ct_overlay = _generate_vlg_overlay(ct_pil, result, 'ct')
            st.image(ct_overlay, caption="CT Detection", width="stretch")
        with col_rgb:
            rgb_overlay = _generate_vlg_overlay(rgb_pil, result, 'rgb')
            st.image(rgb_overlay, caption="RGB Detection", width="stretch")
    elif ct_pil:
        overlay = _generate_vlg_overlay(ct_pil, result, 'ct')
        st.image(overlay, caption="CT Detection", width="stretch")
    elif rgb_pil:
        overlay = _generate_vlg_overlay(rgb_pil, result, 'rgb')
        st.image(overlay, caption="RGB Detection", width="stretch")

    # 푸터
    st.markdown(f"""
    <div class="model-box-footer" style="background: #f8f9fa; padding: 0.75rem; border-radius: 0 0 8px 8px; display: flex; justify-content: space-between; border-top: 1px solid #e9ecef;">
        <span class="model-box-result {result_class}" style="font-weight: 600; color: {'#DC3545' if is_defect_flag else '#28A745'};">{result_icon} {verdict_text}</span>
        <span class="model-box-confidence" style="color: #666;">신뢰도: {confidence:.1%}</span>
    </div>
    """, unsafe_allow_html=True)


def _render_ensemble_detail(result, ct_pil, rgb_pil, analysis_mode):
    """앙상블 시스템 상세"""
    # 디버깅
    print(f"[DEBUG] _render_ensemble_detail: result={result is not None}")
    if result:
        print(f"[DEBUG]   prediction={result.prediction}, confidence={result.confidence:.2%}, defect_type={result.defect_type}")
        print(f"[DEBUG]   details keys={list(result.details.keys()) if result.details else None}")
        if result.details:
            ct_r = result.details.get('ct_result')
            rgb_r = result.details.get('rgb_result')
            print(f"[DEBUG]   CT: class={ct_r.get('class_name') if ct_r else None}, is_defect={ct_r.get('is_defect') if ct_r else None}")
            if rgb_r:
                print(f"[DEBUG]   RGB: anomaly_score={rgb_r.get('anomaly_score'):.4f}, threshold={rgb_r.get('threshold'):.4f}, is_defect={rgb_r.get('is_defect')}")
                print(f"[DEBUG]   RGB 판정: score({rgb_r.get('anomaly_score'):.4f}) < threshold({rgb_r.get('threshold'):.4f}) = {rgb_r.get('anomaly_score') < rgb_r.get('threshold')} → is_defect={rgb_r.get('is_defect')}")

    st.markdown("""
    <div class="system-card">
        <div class="system-title">🔬 Ensemble System</div>
        <div class="system-subtitle">CNN + AutoEncoder 앙상블</div>
    """, unsafe_allow_html=True)

    if result:
        details = result.details
        verdict = details.get('verdict', '정상')
        verdict_en = details.get('verdict_en', 'normal')

        # 결함 정보 가져오기
        defect_info = get_defect_info(result.defect_type or 'module_normal')
        severity_style = get_severity_style(defect_info['severity'])

        # 판정 결과
        st.markdown(f"""
        <div class="detail-section">
            <div class="detail-label">최종 판정</div>
            <div class="detail-value" style="color: {severity_style['color']};">
                {defect_info['icon']} {verdict}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # 분석 모드
        mode_text = {'ensemble': 'CT + RGB 앙상블', 'ct_only': 'CT 분석', 'rgb_only': 'RGB 분석'}
        st.markdown(f"""
        <div class="detail-section">
            <div class="detail-label">분석 모드</div>
            <div class="detail-value-small">{mode_text.get(analysis_mode, analysis_mode)}</div>
        </div>
        """, unsafe_allow_html=True)

        # 신뢰도
        st.markdown(f"""
        <div class="detail-section">
            <div class="detail-label">신뢰도</div>
            <div class="detail-value">{result.confidence:.1%}</div>
        </div>
        """, unsafe_allow_html=True)

        # CT 결과 (있는 경우)
        ct_result = details.get('ct_result')
        if ct_result:
            st.markdown(f"""
            <table class="detail-table">
                <tr><th colspan="2">🔬 CT 분석 (내부)</th></tr>
                <tr><td>예측 클래스</td><td><strong>{ct_result.get('class_name', 'N/A')}</strong></td></tr>
                <tr><td>불량 확률</td><td><strong>{ct_result.get('defect_probability', 0):.1%}</strong></td></tr>
                <tr><td>결함 여부</td><td><strong>{'불량' if ct_result.get('is_defect') else '정상'}</strong></td></tr>
            </table>
            """, unsafe_allow_html=True)

        # RGB 결과 (있는 경우)
        rgb_result = details.get('rgb_result')
        if rgb_result:
            st.markdown(f"""
            <table class="detail-table">
                <tr><th colspan="2">📷 RGB 분석 (외부)</th></tr>
                <tr><td>이상 점수</td><td><strong>{rgb_result.get('anomaly_score', 0):.4f}</strong></td></tr>
                <tr><td>임계값</td><td><strong>{rgb_result.get('threshold', 0):.4f}</strong></td></tr>
                <tr><td>결함 여부</td><td><strong>{'불량' if rgb_result.get('is_defect') else '정상'}</strong></td></tr>
            </table>
            """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


def _render_vlm_detail(result, analysis_mode):
    """VLM 시스템 상세"""
    # VLM 모델명 동적 결정
    vlm_model = 'Qwen2-VL'
    if result and result.details:
        vlm_type = result.details.get('vlm_model', 'qwen2vl')
        vlm_model = 'Gemini 2.0 Flash' if vlm_type == 'gemini' else 'Qwen2-VL'

    # 디버깅
    print(f"[DEBUG] _render_vlm_detail: result={result is not None}")
    if result:
        print(f"[DEBUG]   prediction={result.prediction}, confidence={result.confidence:.2%}, defect_type={result.defect_type}")
        if result.details:
            ct_a = result.details.get('ct_analysis')
            rgb_a = result.details.get('rgb_analysis')
            print(f"[DEBUG]   ct_analysis exists: {ct_a is not None}")
            if ct_a:
                print(f"[DEBUG]     ct_analysis keys: {list(ct_a.keys())}")
                print(f"[DEBUG]     ct_analysis prediction: {ct_a.get('prediction')}, defect_type: {ct_a.get('defect_type')}")
            print(f"[DEBUG]   rgb_analysis exists: {rgb_a is not None}")
            if rgb_a:
                print(f"[DEBUG]     rgb_analysis keys: {list(rgb_a.keys())}")
                print(f"[DEBUG]     rgb_analysis prediction: {rgb_a.get('prediction')}, defect_type: {rgb_a.get('defect_type')}")
            print(f"[DEBUG]   explanation length: {len(result.details.get('explanation', ''))}")
            print(f"[DEBUG]   details keys: {list(result.details.keys())}")

    st.markdown(f"""
    <div class="system-card">
        <div class="system-title">🤖 VLM System</div>
        <div class="system-subtitle">Vision-Language 모델 ({vlm_model})</div>
    """, unsafe_allow_html=True)

    if result:
        details = result.details

        # 판정 결과 (내부/외부/복합불량 구분)
        prediction = result.prediction
        verdict = details.get('verdict', '정상')

        # 색상 및 아이콘 설정
        if prediction == 'normal':
            result_color = "#28A745"
            result_icon = "✅"
        elif prediction == 'internal_defect':
            result_color = "#DC3545"
            result_icon = "🔬"
        elif prediction == 'external_defect':
            result_color = "#FF6B35"
            result_icon = "📷"
        elif prediction == 'complex_defect':
            result_color = "#8B0000"
            result_icon = "⚠️"
        else:  # defect (이전 호환)
            result_color = "#DC3545" if prediction != 'normal' else "#28A745"
            result_icon = "🔴" if prediction != 'normal' else "✅"
            verdict = "불량" if prediction != 'normal' else "정상"

        st.markdown(f"""
        <div class="detail-section">
            <div class="detail-label">판정 결과</div>
            <div class="detail-value" style="color: {result_color};">
                {result_icon} {verdict}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # 결함 유형 (있는 경우)
        if result.defect_type:
            st.markdown(f"""
            <div class="detail-section">
                <div class="detail-label">결함 유형</div>
                <div class="detail-value-small">{result.defect_type}</div>
            </div>
            """, unsafe_allow_html=True)

        # 신뢰도
        st.markdown(f"""
        <div class="detail-section">
            <div class="detail-label">신뢰도</div>
            <div class="detail-value">{result.confidence:.1%}</div>
        </div>
        """, unsafe_allow_html=True)

        # CT 분석 (있는 경우)
        ct_analysis = details.get('ct_analysis')
        if ct_analysis:
            with st.expander("🔬 CT 이미지 분석", expanded=True):
                st.markdown(f"""
                <div style="font-size: 0.85rem; color: #333; line-height: 1.6;">
                    {ct_analysis.get('explanation', 'N/A')}
                </div>
                """, unsafe_allow_html=True)

        # RGB 분석 (있는 경우)
        rgb_analysis = details.get('rgb_analysis')
        if rgb_analysis:
            with st.expander("📷 RGB 이미지 분석", expanded=True):
                st.markdown(f"""
                <div style="font-size: 0.85rem; color: #333; line-height: 1.6;">
                    {rgb_analysis.get('explanation', 'N/A')}
                </div>
                """, unsafe_allow_html=True)

        # 종합 설명 (하나만 있는 경우)
        if not ct_analysis and not rgb_analysis:
            st.markdown(f"""
            <div class="ai-description">
                <div class="ai-description-title">💬 AI 분석 설명</div>
                <div style="font-size: 0.85rem; color: #333; line-height: 1.6;">
                    {details.get('explanation', 'N/A')}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # 상세 정보 테이블
        st.markdown(f"""
        <table class="detail-table">
            <tr><th>항목</th><th>값</th></tr>
            <tr><td>사용 모델</td><td><strong>{details.get('model_version', 'Qwen2-VL')}</strong></td></tr>
            <tr><td>추론 시간</td><td><strong>{result.inference_time:.2f}초</strong></td></tr>
        </table>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


def _render_vlg_detail(result, analysis_mode):
    """VLG 시스템 상세"""
    # 디버깅
    print(f"[DEBUG] _render_vlg_detail: result={result is not None}")
    if result:
        print(f"[DEBUG]   prediction={result.prediction}, confidence={result.confidence:.2%}, defect_type={result.defect_type}")
        if result.details:
            dets = result.details.get('detections', [])
            ct_d = result.details.get('ct_detections')
            rgb_d = result.details.get('rgb_detections')
            print(f"[DEBUG]   total detections: {len(dets)}")
            print(f"[DEBUG]   ct_detections: {ct_d.get('num_detections') if ct_d else 'None'}")
            print(f"[DEBUG]   rgb_detections: {rgb_d.get('num_detections') if rgb_d else 'None'}")

    st.markdown("""
    <div class="system-card">
        <div class="system-title">🎯 VLG System</div>
        <div class="system-subtitle">객체 탐지 (GroundingDINO)</div>
    """, unsafe_allow_html=True)

    if result:
        details = result.details
        all_detections = details.get('detections', [])
        ct_detections = details.get('ct_detections')
        rgb_detections = details.get('rgb_detections')

        # 판정 결과 (내부/외부/복합불량 구분)
        prediction = result.prediction
        verdict = details.get('verdict', '정상')

        # 색상 및 아이콘 설정
        if prediction == 'normal':
            result_color = "#28A745"
            result_icon = "✅"
        elif prediction == 'internal_defect':
            result_color = "#DC3545"
            result_icon = "🔬"
        elif prediction == 'external_defect':
            result_color = "#FF6B35"
            result_icon = "📷"
        elif prediction == 'complex_defect':
            result_color = "#8B0000"
            result_icon = "⚠️"
        else:  # defect (이전 호환)
            result_color = "#DC3545" if len(all_detections) > 0 else "#28A745"
            result_icon = "🔴" if len(all_detections) > 0 else "✅"
            verdict = "불량" if len(all_detections) > 0 else "정상"

        # 판정 결과
        st.markdown(f"""
        <div class="detail-section">
            <div class="detail-label">판정 결과</div>
            <div class="detail-value" style="color: {result_color};">
                {result_icon} {verdict}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # 결함 유형 (있는 경우)
        if result.defect_type:
            st.markdown(f"""
            <div class="detail-section">
                <div class="detail-label">결함 유형</div>
                <div class="detail-value-small">{result.defect_type}</div>
            </div>
            """, unsafe_allow_html=True)

        # 총 검출 개수
        st.markdown(f"""
        <div class="detail-section">
            <div class="detail-label">총 검출된 결함</div>
            <div class="detail-value">{len(all_detections)}개</div>
        </div>
        """, unsafe_allow_html=True)

        # CT 검출 결과 (있든 없든 표시)
        if ct_detections is not None:
            ct_count = ct_detections.get('num_detections', 0)
            with st.expander(f"🔬 CT 검출 결과 - 내부 검사 ({ct_count}개)", expanded=True):
                if ct_count > 0 and ct_detections.get('detections'):
                    st.markdown("""<table class="detail-table"><tr><th>결함 유형</th><th>신뢰도</th></tr>""", unsafe_allow_html=True)
                    for det in ct_detections['detections']:
                        det_info = get_defect_info(det['label'])
                        det_title = det_info['title'].split('(')[0].strip()
                        st.markdown(f"""<tr><td>{det_info['icon']} {det_title}</td><td><strong>{det['score']:.1%}</strong></td></tr>""", unsafe_allow_html=True)
                    st.markdown("</table>", unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style="text-align: center; color: #28A745; padding: 0.5rem;">
                        ✅ 내부 검사 결과: 결함 미검출
                    </div>
                    <div style="font-size: 0.8rem; color: #666; text-align: center;">
                        검사 항목: 기공(porosity), 공극(void), 크랙(crack), 레진 오버플로우(resin overflow)
                    </div>
                    """, unsafe_allow_html=True)

        # RGB 검출 결과 (있든 없든 표시)
        if rgb_detections is not None:
            rgb_count = rgb_detections.get('num_detections', 0)
            with st.expander(f"📷 RGB 검출 결과 - 외관 검사 ({rgb_count}개)", expanded=True):
                if rgb_count > 0 and rgb_detections.get('detections'):
                    st.markdown("""<table class="detail-table"><tr><th>결함 유형</th><th>신뢰도</th></tr>""", unsafe_allow_html=True)
                    for det in rgb_detections['detections']:
                        det_info = get_defect_info(det['label'])
                        det_title = det_info['title'].split('(')[0].strip()
                        st.markdown(f"""<tr><td>{det_info['icon']} {det_title}</td><td><strong>{det['score']:.1%}</strong></td></tr>""", unsafe_allow_html=True)
                    st.markdown("</table>", unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style="text-align: center; color: #28A745; padding: 0.5rem;">
                        ✅ 외관 검사 결과: 결함 미검출
                    </div>
                    <div style="font-size: 0.8rem; color: #666; text-align: center;">
                        검사 항목: 오염(pollution), 스크래치(scratch), 손상(damage), 얼룩(stain)
                    </div>
                    """, unsafe_allow_html=True)

        if not all_detections:
            st.markdown("""
            <div style="text-align: center; color: #28A745; padding: 1rem;">
                ✅ 검출된 결함 없음
            </div>
            """, unsafe_allow_html=True)

        # 상세 정보 테이블
        st.markdown(f"""
        <table class="detail-table" style="margin-top: 1rem;">
            <tr><th>항목</th><th>값</th></tr>
            <tr><td>추론 시간</td><td><strong>{result.inference_time:.2f}초</strong></td></tr>
        </table>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


def _render_verdict_section(ensemble_result, vlm_result, vlg_result):
    """최종 판정 섹션"""

    # 디버깅: 실제 결과 출력
    print(f"[DEBUG] _render_verdict_section:")
    print(f"[DEBUG]   ensemble: prediction={ensemble_result.prediction if ensemble_result else None}, defect_type={ensemble_result.defect_type if ensemble_result else None}")
    print(f"[DEBUG]   vlm: prediction={vlm_result.prediction if vlm_result else None}, defect_type={vlm_result.defect_type if vlm_result else None}")
    print(f"[DEBUG]   vlg: prediction={vlg_result.prediction if vlg_result else None}, defect_type={vlg_result.defect_type if vlg_result else None}")

    # 불량 유형별 카운트 (내부/외부/복합)
    defect_types = {'internal': 0, 'external': 0, 'complex': 0}

    def classify_prediction(pred):
        """prediction을 내부/외부/복합으로 분류"""
        if pred in ['internal_defect']:
            return 'internal'
        elif pred in ['external_defect']:
            return 'external'
        elif pred in ['complex_defect']:
            return 'complex'
        elif pred not in ['normal', 'error', 'unknown', None]:
            return 'defect'  # 이전 호환 (단순 defect)
        return None

    # 각 모델의 결과 집계
    valid_results = []
    if ensemble_result and ensemble_result.prediction != 'error':
        dtype = classify_prediction(ensemble_result.prediction)
        if dtype:
            if dtype == 'complex':
                defect_types['internal'] += 1
                defect_types['external'] += 1
            elif dtype in defect_types:
                defect_types[dtype] += 1
            valid_results.append(('ensemble', ensemble_result))

    if vlm_result and vlm_result.prediction != 'error':
        dtype = classify_prediction(vlm_result.prediction)
        if dtype:
            if dtype == 'complex':
                defect_types['internal'] += 1
                defect_types['external'] += 1
            elif dtype in defect_types:
                defect_types[dtype] += 1
            valid_results.append(('vlm', vlm_result))

    if vlg_result and vlg_result.prediction != 'error':
        dtype = classify_prediction(vlg_result.prediction)
        if dtype:
            if dtype == 'complex':
                defect_types['internal'] += 1
                defect_types['external'] += 1
            elif dtype in defect_types:
                defect_types[dtype] += 1
            valid_results.append(('vlg', vlg_result))

    # 최종 판정 결정
    has_internal = defect_types['internal'] > 0
    has_external = defect_types['external'] > 0
    has_defect = has_internal or has_external

    # verdict 결정
    if has_internal and has_external:
        verdict_kr = "복합불량"
        main_defect_class = 'complex_defect'
    elif has_internal:
        verdict_kr = "내부불량"
        main_defect_class = 'internal_defect'
    elif has_external:
        verdict_kr = "외부불량"
        main_defect_class = 'external_defect'
    else:
        verdict_kr = "정상"
        main_defect_class = 'cell_normal'

    # 상세 결함 유형 가져오기 (앙상블 > VLG > VLM 우선순위)
    detail_defect_type = None
    for model_name, result in valid_results:
        if result.defect_type:
            detail_defect_type = result.defect_type
            break

    defect_info = get_defect_info(detail_defect_type or main_defect_class)

    st.markdown(f"""
    <div class="verdict-section">
        <div class="verdict-title">
            {defect_info['icon']} 최종 진단 리포트
        </div>
    """, unsafe_allow_html=True)

    # 3개 모델 결과 요약 테이블
    def get_result_badge(result, model_type):
        if result is None:
            return '<span style="color: #999;">-</span>'
        if result.prediction == 'error':
            return '<span style="color: #999;">⚠️ 에러</span>'

        prediction = result.prediction
        verdict = result.details.get('verdict', '') if result.details else ''

        # 내부/외부/복합불량 구분
        if prediction == 'normal':
            return '<span style="color: #28A745; font-weight: 600;">✅ 정상</span>'
        elif prediction == 'internal_defect':
            defect_type = result.defect_type or '내부결함'
            return f'<span style="color: #DC3545; font-weight: 600;">🔬 내부불량 ({defect_type})</span>'
        elif prediction == 'external_defect':
            defect_type = result.defect_type or '외부결함'
            return f'<span style="color: #FF6B35; font-weight: 600;">📷 외부불량 ({defect_type})</span>'
        elif prediction == 'complex_defect':
            defect_type = result.defect_type or '복합결함'
            return f'<span style="color: #8B0000; font-weight: 600;">⚠️ 복합불량 ({defect_type})</span>'
        else:
            # 이전 호환 (defect 등)
            defect_type = result.defect_type or verdict or '불량'
            return f'<span style="color: #DC3545; font-weight: 600;">🔴 {defect_type}</span>'

    def get_confidence(result):
        if result is None or result.prediction == 'error':
            return '-'
        return f'{result.confidence:.1%}'

    ensemble_badge = get_result_badge(ensemble_result, 'ensemble')
    vlm_badge = get_result_badge(vlm_result, 'vlm')
    vlg_badge = get_result_badge(vlg_result, 'vlg')

    ensemble_conf = get_confidence(ensemble_result)
    vlm_conf = get_confidence(vlm_result)
    vlg_conf = get_confidence(vlg_result)

    # VLG 모델 이름 가져오기
    vlg_model_name = 'GroundingDINO'
    if vlg_result and vlg_result.details:
        vlg_type = vlg_result.details.get('vlg_model', 'groundingdino')
        vlg_model_name = 'YOLO-World' if vlg_type == 'yoloworld' else 'GroundingDINO'

    st.markdown(f"""
    <div class="verdict-item">
        <div class="verdict-item-title">🔬 3-Way 분석 결과</div>
        <table style="width: 100%; border-collapse: collapse; margin-top: 0.5rem;">
            <tr style="background: #f8f9fa;">
                <th style="padding: 8px; text-align: left; border: 1px solid #e9ecef;">모델</th>
                <th style="padding: 8px; text-align: center; border: 1px solid #e9ecef;">판정</th>
                <th style="padding: 8px; text-align: center; border: 1px solid #e9ecef;">신뢰도</th>
            </tr>
            <tr>
                <td style="padding: 8px; border: 1px solid #e9ecef;">🔬 Ensemble (CNN+AE)</td>
                <td style="padding: 8px; text-align: center; border: 1px solid #e9ecef;">{ensemble_badge}</td>
                <td style="padding: 8px; text-align: center; border: 1px solid #e9ecef;">{ensemble_conf}</td>
            </tr>
            <tr>
                <td style="padding: 8px; border: 1px solid #e9ecef;">🤖 VLM (Qwen2-VL)</td>
                <td style="padding: 8px; text-align: center; border: 1px solid #e9ecef;">{vlm_badge}</td>
                <td style="padding: 8px; text-align: center; border: 1px solid #e9ecef;">{vlm_conf}</td>
            </tr>
            <tr>
                <td style="padding: 8px; border: 1px solid #e9ecef;">🎯 VLG ({vlg_model_name})</td>
                <td style="padding: 8px; text-align: center; border: 1px solid #e9ecef;">{vlg_badge}</td>
                <td style="padding: 8px; text-align: center; border: 1px solid #e9ecef;">{vlg_conf}</td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

    # 종합 판정
    defect_count = len(valid_results)
    total_models = 3  # ensemble, vlm, vlg

    # verdict 색상 결정
    if verdict_kr == "정상":
        verdict_color = "#28A745"
    elif verdict_kr == "내부불량":
        verdict_color = "#DC3545"
    elif verdict_kr == "외부불량":
        verdict_color = "#FF6B35"
    else:  # 복합불량
        verdict_color = "#8B0000"

    if has_defect:
        # 결함 유형별 설명
        defect_desc = []
        if has_internal:
            defect_desc.append("내부 결함")
        if has_external:
            defect_desc.append("외부 결함")
        defect_desc_str = " 및 ".join(defect_desc)

        st.markdown(f"""
        <div class="verdict-item">
            <div class="verdict-item-title">📋 종합 판정</div>
            <div class="verdict-item-content">
                이 샘플은 <strong style="color: {verdict_color};">{verdict_kr}</strong>으로 판정되었습니다.<br>
                {defect_count}개 시스템에서 <strong>{defect_desc_str}</strong>을 감지하였습니다.
            </div>
        </div>
        """, unsafe_allow_html=True)

        # 결함 정보 카드
        st.markdown(render_defect_card(detail_defect_type or main_defect_class), unsafe_allow_html=True)

        st.markdown(f"""
        <div class="verdict-item">
            <div class="verdict-item-title">⚡ 권장 조치</div>
            <div class="verdict-item-content">
                <strong>추정 원인:</strong> {defect_info['cause']}<br>
                <strong>조치 사항:</strong> {defect_info['action']}
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="verdict-item">
            <div class="verdict-item-title">📋 종합 판정</div>
            <div class="verdict-item-content">
                이 샘플은 <strong style="color: #28A745;">정상(Normal)</strong>으로 판정되었습니다.<br>
                분석된 시스템에서 유의미한 결함이 감지되지 않았습니다.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="verdict-item">
            <div class="verdict-item-title">✅ 다음 단계</div>
            <div class="verdict-item-content">
                {defect_info['action']}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


def _generate_gradcam_overlay(image: Image.Image) -> Image.Image:
    """CT Grad-CAM 오버레이 생성"""
    img = image.copy().convert('RGB')
    w, h = img.size

    heatmap = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(heatmap)

    cx, cy = w // 2, h // 2
    max_r = min(w, h) // 4

    for r in range(max_r, 0, -3):
        ratio = 1 - (r / max_r)

        if ratio < 0.25:
            red, green, blue = 0, int(255 * (ratio / 0.25)), 255
        elif ratio < 0.5:
            red, green, blue = 0, 255, int(255 * (1 - (ratio - 0.25) / 0.25))
        elif ratio < 0.75:
            red, green, blue = int(255 * ((ratio - 0.5) / 0.25)), 255, 0
        else:
            red, green, blue = 255, int(255 * (1 - (ratio - 0.75) / 0.25)), 0

        alpha = int(120 * ratio + 30)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(red, green, blue, alpha))

    img = img.convert('RGBA')
    result = Image.alpha_composite(img, heatmap)
    result = result.convert('RGB')

    # 라벨
    draw = ImageDraw.Draw(result)
    draw.rectangle([5, 5, 100, 25], fill='#1a1a2e')
    draw.text((10, 8), "CT Grad-CAM", fill='white')

    return result


def _generate_ae_error_overlay(image: Image.Image) -> Image.Image:
    """RGB AE 에러맵 오버레이 생성"""
    img = image.copy().convert('RGB')
    w, h = img.size

    # 더미 에러맵 (실제 모델 연동 시 AE 재구성 오차 사용)
    heatmap = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(heatmap)

    # 임의의 오염 영역 표시
    import random
    random.seed(43)
    for _ in range(3):
        cx = random.randint(w // 4, 3 * w // 4)
        cy = random.randint(h // 4, 3 * h // 4)
        r = random.randint(20, 40)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(255, 100, 0, 100))

    img = img.convert('RGBA')
    result = Image.alpha_composite(img, heatmap)
    result = result.convert('RGB')

    # 라벨
    draw = ImageDraw.Draw(result)
    draw.rectangle([5, 5, 100, 25], fill='#FF6B35')
    draw.text((10, 8), "RGB Error", fill='white')

    return result


def _generate_vlm_overlay(image: Image.Image, result=None, modality='ct') -> Image.Image:
    """VLM Grounding 오버레이 생성 - 실제 VLM 결과 사용"""
    img = image.copy().convert('RGB')
    w, h = img.size
    draw = ImageDraw.Draw(img)

    # 색상 설정
    color = '#4CAF50' if modality == 'ct' else '#2196F3'

    # 결과에서 분석 데이터 추출
    analysis = None
    if result and result.details:
        if modality == 'ct':
            analysis = result.details.get('ct_analysis')
        else:
            analysis = result.details.get('rgb_analysis')

    # 라벨 먼저 그리기
    label_text = f"VLM {modality.upper()}"
    draw.rectangle([5, 5, 70, 25], fill=color)
    draw.text((10, 8), label_text, fill='white')

    # 분석 결과가 없거나 정상이면 박스 없이 반환
    if not analysis:
        return img

    prediction = analysis.get('prediction', 'normal')
    is_defect = prediction not in ['normal', 'unknown', 'error']
    if not is_defect:
        # 정상이면 체크마크 표시
        draw.rectangle([w - 90, 5, w - 5, 30], fill='#28A745')
        draw.text((w - 85, 8), "Normal ✓", fill='white')
        return img

    # 불량인 경우 - 위치 정보 기반 박스 생성
    location = analysis.get('location', '')
    defect_type = analysis.get('defect_type', 'Defect')
    confidence = analysis.get('confidence', 80)

    # 텍스트 위치를 대략적인 좌표로 변환
    bbox = _location_to_bbox(location, w, h)

    if bbox:
        x1, y1, x2, y2 = bbox
        # 결함 영역 박스 (굵은 선)
        draw.rectangle([x1, y1, x2, y2], outline='#FF4757', width=6)

        # 라벨
        label = f"{defect_type}: {confidence}%"
        text_width = len(label) * 7
        draw.rectangle([x1, y1 - 20, x1 + text_width + 6, y1], fill='#FF4757')
        draw.text((x1 + 3, y1 - 17), label, fill='white')

    return img


def _location_to_bbox(location: str, w: int, h: int) -> tuple:
    """
    텍스트 위치 설명을 대략적인 바운딩 박스로 변환

    Args:
        location: 위치 설명 텍스트 (예: "중앙", "상단 좌측")
        w, h: 이미지 크기

    Returns:
        (x1, y1, x2, y2) 또는 None
    """
    if not location:
        # 위치 정보 없으면 중앙 영역
        margin = 0.25
        return (int(w * margin), int(h * margin), int(w * (1 - margin)), int(h * (1 - margin)))

    location = location.lower()

    # 수직 위치 결정
    if '상단' in location or '위' in location or 'top' in location or 'upper' in location:
        y1_ratio, y2_ratio = 0.1, 0.45
    elif '하단' in location or '아래' in location or 'bottom' in location or 'lower' in location:
        y1_ratio, y2_ratio = 0.55, 0.9
    else:  # 중앙
        y1_ratio, y2_ratio = 0.3, 0.7

    # 수평 위치 결정
    if '좌측' in location or '왼쪽' in location or 'left' in location:
        x1_ratio, x2_ratio = 0.1, 0.45
    elif '우측' in location or '오른쪽' in location or 'right' in location:
        x1_ratio, x2_ratio = 0.55, 0.9
    else:  # 중앙
        x1_ratio, x2_ratio = 0.25, 0.75

    return (int(w * x1_ratio), int(h * y1_ratio), int(w * x2_ratio), int(h * y2_ratio))


def _generate_vlg_overlay(image: Image.Image, result, modality='ct') -> Image.Image:
    """VLG Detection 오버레이 생성"""
    img = image.copy().convert('RGB')
    w, h = img.size
    draw = ImageDraw.Draw(img)

    # 라벨
    label_text = f"VLG {modality.upper()}"
    draw.rectangle([5, 5, 70, 25], fill='#FF4757')
    draw.text((10, 8), label_text, fill='white')

    label_map = {
        'cell_normal': 'Normal', 'cell_porosity': 'Porosity',
        'module_normal': 'Normal', 'module_porosity': 'Porosity',
        'module_resin_overflow': 'Resin', 'pollution': 'Pollution',
        'scratch': 'Scratch', 'damage': 'Damage',
        'contamination': 'Contamination', 'stain': 'Stain',
        'porosity': 'Porosity', 'void': 'Void', 'bubble': 'Bubble',
        'crack': 'Crack', 'resin overflow': 'Resin',
    }

    # 해당 modality 검출 결과만 표시
    detections = []
    if result and result.details:
        if modality == 'ct' and result.details.get('ct_detections'):
            detections = result.details['ct_detections'].get('detections', [])
        elif modality == 'rgb' and result.details.get('rgb_detections'):
            detections = result.details['rgb_detections'].get('detections', [])

    # 검출 없으면 "검출 없음" 표시
    if not detections:
        draw.rectangle([w - 100, 5, w - 5, 30], fill='#28A745')
        draw.text((w - 95, 8), "No Defect", fill='white')
        return img

    colors = ['#FF4757', '#FFA502', '#2ED573']
    for i, det in enumerate(detections):
        bbox = det['bbox']

        # bbox 형식 처리: [cx, cy, w, h] 또는 [x1, y1, x2, y2]
        if len(bbox) == 4:
            # GroundingDINO는 [cx, cy, width, height] 형식 (정규화)
            # 또는 [x1, y1, x2, y2] 형식일 수 있음
            # x2 < x1이면 cxcywh 형식으로 간주
            if bbox[2] < bbox[0] or bbox[3] < bbox[1]:
                # [cx, cy, width, height] 형식
                cx, cy, bw, bh = bbox
                x1 = int((cx - bw / 2) * w)
                y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w)
                y2 = int((cy + bh / 2) * h)
            else:
                # [x1, y1, x2, y2] 형식
                x1, y1, x2, y2 = int(bbox[0] * w), int(bbox[1] * h), int(bbox[2] * w), int(bbox[3] * h)

            # 좌표 순서 보정 (x1 < x2, y1 < y2 보장)
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)

            # 경계 체크
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)

            # 최소 크기 보장
            if x2 - x1 < 10:
                x2 = x1 + 10
            if y2 - y1 < 10:
                y2 = y1 + 10

            color = colors[i % len(colors)]
            draw.rectangle([x1, y1, x2, y2], outline=color, width=6)

            eng_label = label_map.get(det['label'].lower(), det['label'])
            label = f"{eng_label}: {det['score']:.0%}"
            text_width = len(label) * 7

            # 라벨 위치 (이미지 경계 내)
            label_y = max(20, y1)
            draw.rectangle([x1, label_y - 20, x1 + text_width + 6, label_y], fill=color)
            draw.text((x1 + 3, label_y - 17), label, fill='white')

    return img


def _download_report():
    """분석 리포트 다운로드"""
    import json
    from datetime import datetime

    report = {
        'timestamp': datetime.now().isoformat(),
        'analysis_mode': st.session_state.get('analysis_mode'),
        'ct_filename': st.session_state.get('ct_filename'),
        'rgb_filename': st.session_state.get('rgb_filename'),
        'results': {}
    }

    for model_id in ['ensemble', 'vlm', 'vlg']:
        result = get_analysis_result(model_id)
        if result:
            report['results'][model_id] = {
                'model_name': result.model_name,
                'prediction': result.prediction,
                'confidence': result.confidence,
                'defect_type': result.defect_type,
                'inference_time': result.inference_time,
                'details': result.details,
            }

    report_json = json.dumps(report, indent=2, ensure_ascii=False, default=str)

    st.download_button(
        label="📥 JSON 리포트 다운로드",
        data=report_json,
        file_name=f"battery_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
    )
