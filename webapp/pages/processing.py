"""Page 2: Processing - 추론 진행 중 (실제 모델 연동)"""
import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
_project_root = Path(__file__).parent.parent.parent.absolute()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import streamlit as st
import time
from PIL import Image
import io
import tempfile
import os

# 설정 모듈 로드
from config import settings

from webapp.utils.session import (
    navigate_to, set_analysis_result, AnalysisResult
)
from webapp.utils.styles import render_alert


def log(msg: str):
    """터미널 로그 출력"""
    print(f"[WEBAPP] {time.strftime('%H:%M:%S')} - {msg}", flush=True)


# 모델 싱글톤 (앱 시작 시 한 번만 로드)
@st.cache_resource
def load_ensemble_model():
    """앙상블 모델 로드 (캐싱)"""
    log("🔵 Ensemble 모델 로드 시작...")
    from models.ensemble.ensemble import create_ensemble
    try:
        ensemble = create_ensemble()
        log("✅ Ensemble 모델 로드 성공!")
        return ensemble, None
    except Exception as e:
        log(f"❌ Ensemble 모델 로드 실패: {e}")
        return None, str(e)


@st.cache_resource
def load_vlm_model(model_type: str = None):
    """VLM 모델 로드 (캐싱) - Qwen2-VL 또는 Gemini"""
    # 모델 타입이 지정되지 않으면 설정에서 기본값 사용
    if model_type is None:
        model_type = settings.VLM_DEFAULT_MODEL

    log(f"🟣 VLM 모델 로드 시작... (모델: {model_type})")
    try:
        if model_type == 'gemini':
            from models.vlm.inference_gemini import GeminiVLMInference
            # API 키는 config에서 로드
            vlm = GeminiVLMInference(
                api_key=settings.GEMINI_API_KEY,
                model_name=settings.GEMINI_MODEL_NAME
            )
            log("✅ Gemini VLM 모델 로드 성공!")
        else:
            from models.vlm.inference import VLMInference
            vlm = VLMInference(model_size=settings.VLM_MODEL_SIZE)
            log("✅ Qwen2-VL 모델 로드 성공!")
        return vlm, None
    except Exception as e:
        import traceback
        log(f"❌ VLM 모델 로드 실패: {e}")
        return None, f"{e}\n{traceback.format_exc()}"


@st.cache_resource
def load_vlg_model(model_type: str = 'groundingdino'):
    """VLG 모델 로드 (캐싱)"""
    log(f"🟠 VLG 모델 로드 시작... (모델: {model_type})")
    try:
        if model_type == 'yoloworld':
            from models.vlg.inference_yoloworld import YOLOWorldInference
            vlg = YOLOWorldInference()
            log("✅ YOLO-World 모델 로드 성공!")
        else:
            from models.vlg.inference import VLGInference
            vlg = VLGInference()
            log("✅ GroundingDINO 모델 로드 성공!")
        return vlg, None
    except Exception as e:
        log(f"❌ VLG 모델 로드 실패: {e}")
        return None, str(e)


def render():
    """프로세싱 페이지 렌더링"""

    # 이미지 확인 (CT 또는 RGB 중 하나라도 있어야 함)
    ct_image = st.session_state.get('ct_image')
    rgb_image = st.session_state.get('rgb_image')
    analysis_mode = st.session_state.get('analysis_mode', 'ct_only')

    if ct_image is None and rgb_image is None:
        navigate_to('home')
        return

    # 이미지 표시 영역 구성
    _render_images(ct_image, rgb_image, analysis_mode)

    st.markdown("<br>", unsafe_allow_html=True)

    # 분석 진행
    _run_analysis(ct_image, rgb_image, analysis_mode)


def _render_images(ct_image, rgb_image, analysis_mode):
    """업로드된 이미지 표시"""

    if analysis_mode == 'ensemble':
        # 2컬럼: CT | RGB
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            <div class="card" style="text-align: center; padding: 0.5rem;">
                <div style="font-size: 0.9rem; color: #666; margin-bottom: 0.5rem;">🔬 CT Image (내부)</div>
            </div>
            """, unsafe_allow_html=True)
            if ct_image:
                image = Image.open(io.BytesIO(ct_image))
                st.image(image, width="stretch")

        with col2:
            st.markdown("""
            <div class="card" style="text-align: center; padding: 0.5rem;">
                <div style="font-size: 0.9rem; color: #666; margin-bottom: 0.5rem;">📷 RGB Image (외부)</div>
            </div>
            """, unsafe_allow_html=True)
            if rgb_image:
                image = Image.open(io.BytesIO(rgb_image))
                st.image(image, width="stretch")

    elif analysis_mode == 'ct_only' and ct_image:
        # CT만
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            <div class="card" style="text-align: center; padding: 0.5rem;">
                <div style="font-size: 0.9rem; color: #666; margin-bottom: 0.5rem;">🔬 CT Image (내부 검사)</div>
            </div>
            """, unsafe_allow_html=True)
            image = Image.open(io.BytesIO(ct_image))
            st.image(image, width="stretch")

    elif analysis_mode == 'rgb_only' and rgb_image:
        # RGB만
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            <div class="card" style="text-align: center; padding: 0.5rem;">
                <div style="font-size: 0.9rem; color: #666; margin-bottom: 0.5rem;">📷 RGB Image (외부 검사)</div>
            </div>
            """, unsafe_allow_html=True)
            image = Image.open(io.BytesIO(rgb_image))
            st.image(image, width="stretch")


def _save_temp_image(image_bytes: bytes, prefix: str = "temp") -> str:
    """이미지 바이트를 임시 파일로 저장"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png", prefix=prefix) as f:
        image = Image.open(io.BytesIO(image_bytes))
        image.save(f.name)
        return f.name


def _run_analysis(ct_image, rgb_image, analysis_mode):
    """분석 실행"""

    # 상태 메시지 컨테이너
    status_container = st.container()

    with status_container:
        # 모드별 안내 메시지
        if analysis_mode == 'ensemble':
            mode_msg = "🔗 앙상블 분석 - CT (내부) + RGB (외부) 종합 판정"
        elif analysis_mode == 'ct_only':
            mode_msg = "🔬 CT 분석 - 내부 결함 검사"
        else:
            mode_msg = "📷 RGB 분석 - 외부 결함 검사"

        st.markdown(
            render_alert(f"이미지 업로드 완료. {mode_msg}", "info", "✅"),
            unsafe_allow_html=True
        )

        # 임시 파일로 이미지 저장
        ct_path = None
        rgb_path = None

        if ct_image:
            ct_path = _save_temp_image(ct_image, "ct_")
        if rgb_image:
            rgb_path = _save_temp_image(rgb_image, "rgb_")

        try:
            # 분석 진행
            progress_placeholder = st.empty()

            # 3개 모델 분석
            models = [
                ('ensemble', 'Ensemble System', '앙상블 (CNN + AE)'),
                ('vlm', 'VLM System', 'VLM (Qwen2-VL)'),
                ('vlg', 'VLG System', 'VLG (GroundingDINO)'),
            ]

            for i, (model_id, model_name, model_desc) in enumerate(models):
                with progress_placeholder.container():
                    st.markdown(f"""
                    <div class="alert-box alert-info" style="background: #FFF3E0; border-left-color: #FF9800; color: #E65100;">
                        <span>⏳</span>
                        <span>{model_desc} 분석 중... ({i+1}/3)</span>
                    </div>
                    """, unsafe_allow_html=True)

                # 분석 실행
                result = _run_inference(model_id, ct_path, rgb_path, analysis_mode)
                set_analysis_result(model_id, result)

            # 완료 메시지
            progress_placeholder.empty()

            st.markdown(
                render_alert("모든 분석이 완료되었습니다!", "success", "✨"),
                unsafe_allow_html=True
            )

        finally:
            # 임시 파일 정리
            if ct_path and os.path.exists(ct_path):
                os.remove(ct_path)
            if rgb_path and os.path.exists(rgb_path):
                os.remove(rgb_path)

        st.markdown("<br>", unsafe_allow_html=True)

        # 결과 보기 버튼
        if st.button("⏱ 비교 대시보드 결과 보기", width="stretch"):
            st.session_state.analysis_complete = True
            navigate_to('summary')


def _run_inference(model_id: str, ct_path: str, rgb_path: str, analysis_mode: str) -> AnalysisResult:
    """
    모델 추론 실행

    Args:
        model_id: 모델 ID (ensemble, vlm, vlg)
        ct_path: CT 이미지 임시 파일 경로
        rgb_path: RGB 이미지 임시 파일 경로
        analysis_mode: 분석 모드 (ensemble, ct_only, rgb_only)

    Returns:
        AnalysisResult
    """
    if model_id == 'ensemble':
        return _run_ensemble_inference(ct_path, rgb_path, analysis_mode)

    elif model_id == 'vlm':
        return _run_vlm_inference(ct_path, rgb_path, analysis_mode)

    elif model_id == 'vlg':
        return _run_vlg_inference(ct_path, rgb_path, analysis_mode)

    return AnalysisResult(
        model_name='Unknown',
        prediction='unknown',
        confidence=0.0,
    )


def _run_ensemble_inference(ct_path: str, rgb_path: str, analysis_mode: str) -> AnalysisResult:
    """
    앙상블 추론 (CT CNN + RGB AE) - 실제 모델 사용
    """
    import time
    start_time = time.time()

    log("🔵 Ensemble 추론 시작...")

    # 모델 로드
    ensemble, error = load_ensemble_model()

    if error:
        # 모델 로드 실패 시 에러 결과 반환
        log(f"❌ Ensemble 모델 사용 불가: {error}")
        return AnalysisResult(
            model_name='Ensemble System',
            prediction='error',
            confidence=0.0,
            defect_type=None,
            details={'error': error, 'mode': analysis_mode},
            inference_time=0.0,
        )

    try:
        # 분석 모드에 따른 추론 (시각화 포함)
        visualizations = None

        if analysis_mode == 'ensemble' and ct_path and rgb_path:
            # 앙상블: Grad-CAM + Error Map 포함
            result = ensemble.predict_with_visualization(ct_path, rgb_path)
            visualizations = result.get('visualizations')
        elif analysis_mode == 'ct_only' and ct_path:
            # CT only: Grad-CAM 포함
            ct_result_with_gradcam = ensemble.ct_predictor.predict_with_gradcam(ct_path)
            result = ensemble.predict_ct_only(ct_path)
            result['ct_result'] = ct_result_with_gradcam
            visualizations = {
                'ct_gradcam_overlay': ct_result_with_gradcam['gradcam']['overlay'],
                'ct_gradcam_heatmap': ct_result_with_gradcam['gradcam']['heatmap_colored'],
                'ct_original': ct_result_with_gradcam['gradcam']['original'],
            }
        elif analysis_mode == 'rgb_only' and rgb_path:
            # RGB only: Error Map 포함
            result = ensemble.predict_rgb_only(rgb_path)
            rgb_original, rgb_reconstructed, rgb_error_map = ensemble.get_rgb_reconstruction(rgb_path)
            visualizations = {
                'rgb_original': rgb_original,
                'rgb_reconstructed': rgb_reconstructed,
                'rgb_error_map': rgb_error_map,
            }
        else:
            # 이미지가 없는 경우
            return AnalysisResult(
                model_name='Ensemble System',
                prediction='error',
                confidence=0.0,
                details={'error': '이미지가 없습니다.', 'mode': analysis_mode},
                inference_time=0.0,
            )

        inference_time = time.time() - start_time

        # 결과 변환
        verdict = result.get('verdict', '알 수 없음')
        verdict_en = result.get('verdict_en', 'unknown')
        confidence = result.get('confidence', 0.0)

        # 결함 유형 추출
        defect_type = None
        if result.get('ct_result') and result['ct_result'].get('is_defect'):
            defect_type = result['ct_result'].get('class_name')
        if result.get('rgb_result') and result['rgb_result'].get('is_defect'):
            if defect_type:
                defect_type += " + 외관이상"
            else:
                defect_type = "외관이상 (오염/손상)"

        log(f"✅ Ensemble 추론 완료: {verdict} (신뢰도: {confidence:.1%})")
        return AnalysisResult(
            model_name='Ensemble System',
            prediction=verdict_en,
            confidence=confidence,
            defect_type=defect_type,
            details={
                'verdict': verdict,
                'verdict_en': verdict_en,
                'mode': analysis_mode,
                'ct_result': result.get('ct_result'),
                'rgb_result': result.get('rgb_result'),
                'visualizations': visualizations,  # 실제 Grad-CAM/Error Map
            },
            inference_time=inference_time,
        )

    except Exception as e:
        log(f"❌ Ensemble 추론 오류: {e}")
        return AnalysisResult(
            model_name='Ensemble System',
            prediction='error',
            confidence=0.0,
            details={'error': str(e), 'mode': analysis_mode},
            inference_time=0.0,
        )


def _run_vlm_inference(ct_path: str, rgb_path: str, analysis_mode: str) -> AnalysisResult:
    """
    VLM 추론 (Qwen2-VL 또는 Gemini) - 실제 모델 사용
    """
    import time
    start_time = time.time()

    # 선택된 VLM 모델 타입 가져오기
    vlm_model_type = st.session_state.get('vlm_model_type', 'qwen2vl')
    model_display_name = 'Gemini 2.0 Flash' if vlm_model_type == 'gemini' else 'Qwen2-VL-2B'

    log(f"🟣 VLM 추론 시작... (모델: {model_display_name})")

    # 모델 로드 시도
    vlm, error = load_vlm_model(vlm_model_type)

    if error or vlm is None:
        # 모델 로드 실패 시 에러 반환 (더미 결과 X)
        log(f"❌ VLM 모델 사용 불가: {error}")
        return AnalysisResult(
            model_name=f'VLM System ({model_display_name})',
            prediction='error',
            confidence=0.0,
            defect_type=None,
            details={
                'error': error or 'VLM 모델 로드 실패',
                'analysis_mode': analysis_mode,
                'model_version': f'{model_display_name} (미로드)',
                'vlm_model': vlm_model_type,
            },
            inference_time=time.time() - start_time,
        )

    # 실제 VLM 추론
    try:
        ct_analysis = None
        rgb_analysis = None

        # CT 이미지 분석
        if ct_path:
            ct_result = vlm.analyze_image(ct_path, modality='ct')
            ct_analysis = {
                'prediction': 'defect' if not ct_result.get('is_normal', True) else 'normal',
                'confidence': ct_result.get('confidence', 80),
                'defect_type': ct_result.get('defect_type'),
                'explanation': ct_result.get('raw_response', '분석 완료'),
                'location': ct_result.get('location'),
                'modality': 'ct',
            }

        # RGB 이미지 분석
        if rgb_path:
            rgb_result = vlm.analyze_image(rgb_path, modality='rgb')
            rgb_analysis = {
                'prediction': 'defect' if not rgb_result.get('is_normal', True) else 'normal',
                'confidence': rgb_result.get('confidence', 80),
                'defect_type': rgb_result.get('defect_type'),
                'explanation': rgb_result.get('raw_response', '분석 완료'),
                'location': rgb_result.get('location'),
                'modality': 'rgb',
            }

        inference_time = time.time() - start_time

        # 종합 판정 (내부/외부/복합불량 구분)
        ct_is_defect = ct_analysis and ct_analysis['prediction'] == 'defect'
        rgb_is_defect = rgb_analysis and rgb_analysis['prediction'] == 'defect'

        explanation = ""
        defect_type = None
        confidence = 0.0

        # prediction 결정: normal, internal_defect, external_defect, complex_defect
        if ct_is_defect and rgb_is_defect:
            prediction = 'complex_defect'
            verdict = '복합불량'
            defect_type = f"{ct_analysis['defect_type'] or '내부결함'} + {rgb_analysis['defect_type'] or '외부결함'}"
        elif ct_is_defect:
            prediction = 'internal_defect'
            verdict = '내부불량'
            defect_type = ct_analysis['defect_type']
        elif rgb_is_defect:
            prediction = 'external_defect'
            verdict = '외부불량'
            defect_type = rgb_analysis['defect_type'] or '외관이상'
        else:
            prediction = 'normal'
            verdict = '정상'

        # explanation & confidence
        if ct_analysis and rgb_analysis:
            explanation = f"[CT 분석]\n{ct_analysis['explanation']}\n\n[RGB 분석]\n{rgb_analysis['explanation']}"
            confidence = ((ct_analysis.get('confidence') or 80) + (rgb_analysis.get('confidence') or 80)) / 200.0
        elif ct_analysis:
            explanation = ct_analysis['explanation']
            confidence = (ct_analysis.get('confidence') or 80) / 100.0
        elif rgb_analysis:
            explanation = rgb_analysis['explanation']
            confidence = (rgb_analysis.get('confidence') or 80) / 100.0

        log(f"✅ VLM 추론 완료 ({model_display_name}): {verdict} (신뢰도: {confidence:.1%})")
        return AnalysisResult(
            model_name=f'VLM System ({model_display_name})',
            prediction=prediction,
            confidence=confidence,
            defect_type=defect_type,
            details={
                'verdict': verdict,
                'explanation': explanation,
                'ct_analysis': ct_analysis,
                'rgb_analysis': rgb_analysis,
                'analysis_mode': analysis_mode,
                'model_version': model_display_name,
                'vlm_model': vlm_model_type,
            },
            inference_time=inference_time,
        )

    except Exception as e:
        import traceback
        log(f"❌ VLM 추론 오류: {e}")
        return AnalysisResult(
            model_name=f'VLM System ({model_display_name})',
            prediction='error',
            confidence=0.0,
            details={'error': str(e), 'traceback': traceback.format_exc(), 'vlm_model': vlm_model_type},
            inference_time=time.time() - start_time,
        )


def _run_vlg_inference(ct_path: str, rgb_path: str, analysis_mode: str) -> AnalysisResult:
    """
    VLG 추론 (GroundingDINO 또는 YOLO-World) - 실제 모델 사용
    """
    import time
    start_time = time.time()

    # 선택된 VLG 모델 타입 가져오기
    vlg_model_type = st.session_state.get('vlg_model_type', 'groundingdino')
    model_display_name = 'YOLO-World' if vlg_model_type == 'yoloworld' else 'GroundingDINO'

    log(f"🟠 VLG 추론 시작... (모델: {model_display_name})")

    # 모델 로드 시도
    vlg, error = load_vlg_model(vlg_model_type)

    if error or vlg is None:
        # 모델 로드 실패 시 에러 반환 (더미 결과 X)
        log(f"❌ VLG 모델 사용 불가: {error}")
        return AnalysisResult(
            model_name=f'VLG System ({model_display_name})',
            prediction='error',
            confidence=0.0,
            defect_type=None,
            details={
                'error': error or 'VLG 모델 로드 실패',
                'num_detections': 0,
                'detections': [],
                'analysis_mode': analysis_mode,
                'vlg_model': vlg_model_type,
            },
            inference_time=time.time() - start_time,
        )

    # 실제 VLG 추론
    try:
        all_detections = []
        ct_detections = None
        rgb_detections = None

        # CT 결함 탐지
        if ct_path:
            ct_result = vlg.detect(
                ct_path,
                text_prompt="porosity . void . bubble . crack . resin overflow",
                modality='ct',
            )
            # DetectionResult 데이터클래스에서 속성 접근
            ct_detections = {
                'num_detections': len(ct_result.boxes),
                'detections': [
                    {
                        'label': ct_result.labels[i] if i < len(ct_result.labels) else 'defect',
                        'score': float(ct_result.scores[i]) if i < len(ct_result.scores) else 0.5,
                        'bbox': ct_result.boxes[i],
                    }
                    for i in range(len(ct_result.boxes))
                ],
                'modality': 'ct',
            }
            for det in ct_detections['detections']:
                all_detections.append({**det, 'source': 'ct'})

        # RGB 결함 탐지
        if rgb_path:
            rgb_result = vlg.detect(
                rgb_path,
                text_prompt="pollution . contamination . scratch . damage . stain",
                modality='rgb',
            )
            # DetectionResult 데이터클래스에서 속성 접근
            rgb_detections = {
                'num_detections': len(rgb_result.boxes),
                'detections': [
                    {
                        'label': rgb_result.labels[i] if i < len(rgb_result.labels) else 'defect',
                        'score': float(rgb_result.scores[i]) if i < len(rgb_result.scores) else 0.5,
                        'bbox': rgb_result.boxes[i],
                    }
                    for i in range(len(rgb_result.boxes))
                ],
                'modality': 'rgb',
            }
            for det in rgb_detections['detections']:
                all_detections.append({**det, 'source': 'rgb'})

        inference_time = time.time() - start_time

        total = len(all_detections)
        max_score = max([d['score'] for d in all_detections], default=0.0)

        # CT/RGB 검출 여부 확인
        ct_has_defect = ct_detections and ct_detections.get('num_detections', 0) > 0
        rgb_has_defect = rgb_detections and rgb_detections.get('num_detections', 0) > 0

        # prediction 결정: normal, internal_defect, external_defect, complex_defect
        if ct_has_defect and rgb_has_defect:
            prediction = 'complex_defect'
            verdict = '복합불량'
            ct_labels = [d['label'] for d in ct_detections.get('detections', [])]
            rgb_labels = [d['label'] for d in rgb_detections.get('detections', [])]
            defect_type = f"{ct_labels[0] if ct_labels else '내부결함'} + {rgb_labels[0] if rgb_labels else '외부결함'}"
        elif ct_has_defect:
            prediction = 'internal_defect'
            verdict = '내부불량'
            ct_labels = [d['label'] for d in ct_detections.get('detections', [])]
            defect_type = ct_labels[0] if ct_labels else '내부결함'
        elif rgb_has_defect:
            prediction = 'external_defect'
            verdict = '외부불량'
            rgb_labels = [d['label'] for d in rgb_detections.get('detections', [])]
            defect_type = rgb_labels[0] if rgb_labels else '외부결함'
        else:
            prediction = 'normal'
            verdict = '정상'
            defect_type = None

        log(f"✅ VLG 추론 완료 ({model_display_name}): {verdict} - {total}개 검출 (최대 신뢰도: {max_score:.1%})")
        return AnalysisResult(
            model_name=f'VLG System ({model_display_name})',
            prediction=prediction,
            confidence=max_score,
            defect_type=defect_type,
            details={
                'verdict': verdict,
                'num_detections': total,
                'detections': all_detections,
                'ct_detections': ct_detections,
                'rgb_detections': rgb_detections,
                'analysis_mode': analysis_mode,
                'text_prompt': 'porosity . void . pollution . scratch . damage',
                'vlg_model': vlg_model_type,
            },
            inference_time=inference_time,
        )

    except Exception as e:
        log(f"❌ VLG 추론 오류: {e}")
        return AnalysisResult(
            model_name=f'VLG System ({model_display_name})',
            prediction='error',
            confidence=0.0,
            details={'error': str(e), 'vlg_model': vlg_model_type},
            inference_time=time.time() - start_time,
        )
