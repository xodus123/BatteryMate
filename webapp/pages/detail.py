"""Page 4: Detail Dashboard - TensorBoard 스타일 상세 분석"""
import streamlit as st
from PIL import Image, ImageDraw
import io
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

from webapp.utils.session import navigate_to, get_analysis_result


def render():
    """상세 대시보드 렌더링"""

    # 선택된 모델 없으면 요약으로
    if st.session_state.selected_model is None:
        navigate_to('summary')
        return

    model_id = st.session_state.selected_model
    result = get_analysis_result(model_id)

    if result is None:
        navigate_to('summary')
        return

    # 헤더와 네비게이션
    col1, col2 = st.columns([3, 1])

    with col1:
        model_icons = {'ensemble': '🛡️', 'vlm': '🤖', 'vlg': '🎯'}
        icon = model_icons.get(model_id, '📊')
        st.markdown(f"""
        <div class="main-header">
            {icon} {result.model_name} 상세 분석
        </div>
        """, unsafe_allow_html=True)

    with col2:
        if st.button("← 요약으로 돌아가기"):
            navigate_to('summary')

    # 모델별 상세 페이지 렌더링
    if model_id == 'ensemble':
        _render_ensemble_detail(result)
    elif model_id == 'vlm':
        _render_vlm_detail(result)
    elif model_id == 'vlg':
        _render_vlg_detail(result)


def _render_ensemble_detail(result):
    """앙상블 모델 상세 (Scalars & Images Style)"""

    # 탭 구성
    tab_images, tab_scalars, tab_dist = st.tabs([
        "📷 Images", "📈 Scalars", "📊 Distributions"
    ])

    with tab_images:
        st.markdown("""
        <div class="sub-header">원본 이미지 & Grad-CAM 히트맵</div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**원본 이미지**")
            image = Image.open(io.BytesIO(st.session_state.uploaded_image))
            st.image(image, width="stretch")

        with col2:
            st.markdown("**Grad-CAM 히트맵**")
            # 더미 히트맵 생성 (실제로는 Grad-CAM 결과)
            heatmap = _generate_dummy_heatmap(image)
            st.image(heatmap, width="stretch")

        st.markdown("""
        <div style="color: #B0B0B0; font-size: 0.9rem; margin-top: 1rem;">
            💡 Grad-CAM은 모델이 주목한 영역을 시각화합니다. 빨간색 영역이 결함 판단에 중요한 부분입니다.
        </div>
        """, unsafe_allow_html=True)

    with tab_scalars:
        st.markdown("""
        <div class="sub-header">클래스별 확률 분포</div>
        """, unsafe_allow_html=True)

        # 클래스별 확률 바 차트
        class_probs = result.details.get('class_probs', {})
        if class_probs:
            fig = go.Figure()

            classes = list(class_probs.keys())
            probs = list(class_probs.values())

            # 확률 정규화
            total = sum(probs)
            probs = [p / total for p in probs]

            colors = ['#00D084' if 'normal' in c else '#FF4757' for c in classes]

            fig.add_trace(go.Bar(
                x=classes,
                y=probs,
                marker_color=colors,
                text=[f'{p:.1%}' for p in probs],
                textposition='outside',
            ))

            fig.update_layout(
                template='plotly_dark',
                paper_bgcolor='#0E1117',
                plot_bgcolor='#1A1D24',
                height=400,
                yaxis_title='확률',
                xaxis_title='클래스',
                showlegend=False,
            )

            st.plotly_chart(fig, width="stretch")

        # 메트릭 카드
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "CNN 예측",
                result.details.get('cnn_prediction', '-'),
                f"{result.details.get('cnn_confidence', 0):.1%}"
            )

        with col2:
            st.metric(
                "AE 이상 점수",
                f"{result.details.get('ae_anomaly_score', 0):.3f}",
                "정상 범위" if result.details.get('ae_anomaly_score', 0) < 0.5 else "이상 감지"
            )

        with col3:
            st.metric(
                "추론 시간",
                f"{result.inference_time:.2f}초",
            )

    with tab_dist:
        st.markdown("""
        <div class="sub-header">AutoEncoder 이상 점수 분포</div>
        """, unsafe_allow_html=True)

        # 정상 분포와 현재 샘플 비교 (더미 데이터)
        normal_scores = np.random.normal(0.3, 0.1, 1000)
        current_score = result.details.get('ae_anomaly_score', 0.5)

        fig = go.Figure()

        # 정상 분포 히스토그램
        fig.add_trace(go.Histogram(
            x=normal_scores,
            nbinsx=50,
            name='정상 샘플 분포',
            marker_color='#4ECDC4',
            opacity=0.7,
        ))

        # 현재 샘플 표시
        fig.add_vline(
            x=current_score,
            line_dash="dash",
            line_color="#FF6B35",
            annotation_text=f"현재 샘플: {current_score:.3f}",
        )

        # 임계값 표시
        fig.add_vline(
            x=0.5,
            line_dash="dot",
            line_color="#FF4757",
            annotation_text="임계값: 0.5",
        )

        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='#0E1117',
            plot_bgcolor='#1A1D24',
            height=400,
            xaxis_title='이상 점수',
            yaxis_title='빈도',
            showlegend=True,
        )

        st.plotly_chart(fig, width="stretch")


def _render_vlm_detail(result):
    """VLM 상세 (Reasoning & Text Style)"""

    # 탭 구성
    tab_text, tab_grounding, tab_logs = st.tabs([
        "📝 Text (AI 소견)", "🎯 Grounding", "📋 Logs"
    ])

    with tab_text:
        st.markdown("""
        <div class="sub-header">AI 분석 소견서</div>
        """, unsafe_allow_html=True)

        # 판정 결과 헤더
        is_defect = result.prediction not in ['normal', 'unknown', 'error']
        pred_color = "#FF4757" if is_defect else "#00D084"
        # 판정 텍스트 결정
        if result.prediction == 'internal_defect':
            pred_text = "내부불량"
        elif result.prediction == 'external_defect':
            pred_text = "외부불량"
        elif result.prediction == 'complex_defect':
            pred_text = "복합불량"
        elif is_defect:
            pred_text = "불량"
        else:
            pred_text = "정상"

        st.markdown(f"""
        <div style="background: #1A1D24; border-radius: 8px; padding: 1rem; margin-bottom: 1rem;
                    border-left: 4px solid {pred_color};">
            <span style="color: {pred_color}; font-size: 1.5rem; font-weight: 700;">
                판정: {pred_text}
            </span>
            <span style="color: #B0B0B0; margin-left: 1rem;">
                신뢰도: {result.confidence:.1%}
            </span>
        </div>
        """, unsafe_allow_html=True)

        # AI 소견서 (chat message 스타일)
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(result.details.get('explanation', '분석 결과 없음'))

    with tab_grounding:
        st.markdown("""
        <div class="sub-header">텍스트-이미지 연결 (Grounding)</div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**원본 이미지**")
            image = Image.open(io.BytesIO(st.session_state.uploaded_image))
            st.image(image, width="stretch")

        with col2:
            st.markdown("**소견 키워드 하이라이트**")

            # 소견서에서 키워드 추출 및 시각화 (더미)
            explanation = result.details.get('explanation', '')
            keywords = ['기공', 'porosity', '결함', '중앙', '불규칙']

            highlighted = explanation
            for kw in keywords:
                if kw in highlighted:
                    highlighted = highlighted.replace(
                        kw,
                        f'<span style="background: #FF6B35; padding: 0 4px; border-radius: 4px;">{kw}</span>'
                    )

            st.markdown(f"""
            <div style="background: #1A1D24; border-radius: 8px; padding: 1rem; line-height: 1.8;">
                {highlighted}
            </div>
            """, unsafe_allow_html=True)

    with tab_logs:
        st.markdown("""
        <div class="sub-header">추론 파라미터 및 로그</div>
        """, unsafe_allow_html=True)

        # 프롬프트 정보
        with st.expander("사용된 프롬프트", expanded=True):
            prompt_type = result.details.get('prompt_used', 'CT_ANALYSIS')
            st.code(f"프롬프트 유형: {prompt_type}", language='text')

        # 추론 정보
        st.markdown("""
        | 파라미터 | 값 |
        |---------|-----|
        | 모델 | Qwen2-VL-7B-Instruct |
        | 최대 토큰 | 512 |
        | Temperature | 0.0 (Deterministic) |
        | 추론 시간 | {:.2f}초 |
        """.format(result.inference_time))


def _render_vlg_detail(result):
    """VLG 상세 (Detection & PR Curve Style)"""

    # 탭 구성
    tab_detect, tab_metrics, tab_threshold = st.tabs([
        "🎯 Detection", "📊 Metrics", "⚙️ Thresholding"
    ])

    with tab_detect:
        st.markdown("""
        <div class="sub-header">결함 검출 결과</div>
        """, unsafe_allow_html=True)

        detections = result.details.get('detections', [])

        col1, col2 = st.columns([2, 1])

        with col1:
            # 바운딩 박스가 그려진 이미지
            image = Image.open(io.BytesIO(st.session_state.uploaded_image))
            annotated = _draw_bboxes(image, detections)
            st.image(annotated, width="stretch")

        with col2:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value">{len(detections)}</div>
                <div class="metric-label">검출된 결함 수</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # 검출 목록
            if detections:
                st.markdown("**검출 목록**")
                for i, det in enumerate(detections):
                    st.markdown(f"""
                    <div style="background: #1A1D24; border-radius: 8px; padding: 0.5rem; margin: 0.3rem 0;
                                border-left: 3px solid #FF6B35;">
                        <strong>#{i+1}</strong> {det['label']}<br>
                        <span style="color: #B0B0B0;">신뢰도: {det['score']:.1%}</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("검출된 결함이 없습니다.")

    with tab_metrics:
        st.markdown("""
        <div class="sub-header">검출 신뢰도 분포</div>
        """, unsafe_allow_html=True)

        detections = result.details.get('detections', [])

        if detections:
            scores = [d['score'] for d in detections]
            labels = [d['label'] for d in detections]

            fig = go.Figure()

            fig.add_trace(go.Bar(
                x=[f"#{i+1} {l}" for i, l in enumerate(labels)],
                y=scores,
                marker_color='#FF6B35',
                text=[f'{s:.1%}' for s in scores],
                textposition='outside',
            ))

            fig.update_layout(
                template='plotly_dark',
                paper_bgcolor='#0E1117',
                plot_bgcolor='#1A1D24',
                height=300,
                yaxis_title='신뢰도',
                yaxis_range=[0, 1],
                showlegend=False,
            )

            st.plotly_chart(fig, width="stretch")

            # 통계
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("최고 신뢰도", f"{max(scores):.1%}")
            with col2:
                st.metric("평균 신뢰도", f"{np.mean(scores):.1%}")
            with col3:
                st.metric("최저 신뢰도", f"{min(scores):.1%}")
        else:
            st.info("검출된 결함이 없어 메트릭을 표시할 수 없습니다.")

    with tab_threshold:
        st.markdown("""
        <div class="sub-header">임계값 조절</div>
        """, unsafe_allow_html=True)

        # 임계값 슬라이더
        threshold = st.slider(
            "신뢰도 임계값",
            min_value=0.0,
            max_value=1.0,
            value=0.35,
            step=0.05,
            help="임계값 이상의 검출만 표시합니다."
        )

        detections = result.details.get('detections', [])
        filtered = [d for d in detections if d['score'] >= threshold]

        st.markdown(f"""
        <div style="color: #B0B0B0; margin: 1rem 0;">
            임계값 {threshold:.0%} 적용: {len(detections)}개 → {len(filtered)}개 검출
        </div>
        """, unsafe_allow_html=True)

        # 필터링된 결과로 이미지 업데이트
        image = Image.open(io.BytesIO(st.session_state.uploaded_image))
        annotated = _draw_bboxes(image, filtered)
        st.image(annotated, width="stretch")


def _generate_dummy_heatmap(image: Image.Image) -> Image.Image:
    """더미 Grad-CAM 히트맵 생성"""
    import numpy as np

    # 이미지를 numpy 배열로 변환
    img_array = np.array(image)

    # 더미 히트맵 생성 (가우시안)
    h, w = img_array.shape[:2]
    x = np.linspace(0, w, w)
    y = np.linspace(0, h, h)
    X, Y = np.meshgrid(x, y)

    # 랜덤 중심점
    cx, cy = w * 0.4 + np.random.rand() * w * 0.2, h * 0.4 + np.random.rand() * h * 0.2
    heatmap = np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * (w/4)**2))

    # 히트맵을 컬러맵으로 변환
    heatmap_colored = np.zeros((h, w, 3), dtype=np.uint8)
    heatmap_colored[:, :, 0] = (heatmap * 255).astype(np.uint8)  # Red channel

    # 원본과 블렌딩
    alpha = 0.4
    blended = (img_array * (1 - alpha) + heatmap_colored * alpha).astype(np.uint8)

    return Image.fromarray(blended)


def _draw_bboxes(image: Image.Image, detections: list) -> Image.Image:
    """바운딩 박스 그리기"""
    img = image.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size

    colors = {
        'porosity': '#FF4757',
        'void': '#FF6B35',
        'bubble': '#FFA502',
        'default': '#FF4757',
    }

    for det in detections:
        bbox = det['bbox']
        label = det['label']
        score = det['score']

        # 정규화 좌표를 픽셀 좌표로 변환
        x1 = int(bbox[0] * w)
        y1 = int(bbox[1] * h)
        x2 = int(bbox[2] * w)
        y2 = int(bbox[3] * h)

        color = colors.get(label, colors['default'])

        # 박스 그리기 (굵은 선)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=6)

        # 라벨 그리기
        label_text = f"{label}: {score:.0%}"
        draw.rectangle([x1, y1 - 20, x1 + len(label_text) * 8, y1], fill=color)
        draw.text((x1 + 2, y1 - 18), label_text, fill='white')

    return img
