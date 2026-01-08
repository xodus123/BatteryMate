"""세션 상태 관리"""
import streamlit as st
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class AnalysisResult:
    """분석 결과 데이터 클래스"""
    model_name: str
    prediction: str  # 'normal' or 'defect'
    confidence: float
    defect_type: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    inference_time: float = 0.0


def init_session_state():
    """세션 상태 초기화"""
    defaults = {
        # 페이지 네비게이션
        'current_page': 'home',

        # 업로드된 이미지 (기존 호환)
        'uploaded_image': None,
        'image_path': None,
        'image_modality': 'ct',  # 'ct' or 'rgb'

        # CT/RGB 분리 업로드 (앙상블용)
        'ct_image': None,
        'ct_filename': None,
        'rgb_image': None,
        'rgb_filename': None,
        'analysis_mode': None,  # 'ensemble', 'ct_only', 'rgb_only'

        # 모델 설정
        'vlg_model_type': 'groundingdino',  # 'groundingdino' or 'yoloworld'
        'vlm_model_type': 'qwen2vl',  # 'qwen2vl' or 'gemini'

        # 분석 결과
        'analysis_results': {
            'ensemble': None,
            'vlm': None,
            'vlg': None,
        },

        # 상세 페이지용
        'selected_model': None,

        # 분석 완료 여부
        'analysis_complete': False,

        # 분석 시작 시간
        'analysis_start_time': None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def navigate_to(page: str):
    """페이지 이동"""
    st.session_state.current_page = page
    st.rerun()


def set_uploaded_image(image_data, filename: str, modality: str = 'ct'):
    """업로드된 이미지 설정 (기존 호환)"""
    st.session_state.uploaded_image = image_data
    st.session_state.image_path = filename
    st.session_state.image_modality = modality
    st.session_state.analysis_complete = False
    st.session_state.analysis_results = {
        'ensemble': None,
        'vlm': None,
        'vlg': None,
    }


def set_uploaded_images(
    ct_image_data=None,
    ct_filename: str = None,
    rgb_image_data=None,
    rgb_filename: str = None,
    analysis_mode: str = None
):
    """CT/RGB 이미지 분리 업로드 (앙상블용)"""
    st.session_state.ct_image = ct_image_data
    st.session_state.ct_filename = ct_filename
    st.session_state.rgb_image = rgb_image_data
    st.session_state.rgb_filename = rgb_filename
    st.session_state.analysis_mode = analysis_mode
    st.session_state.analysis_complete = False
    st.session_state.analysis_results = {
        'ensemble': None,
        'vlm': None,
        'vlg': None,
    }

    # 기존 호환: 메인 이미지 설정 (CT 우선)
    if ct_image_data is not None:
        st.session_state.uploaded_image = ct_image_data
        st.session_state.image_path = ct_filename
        st.session_state.image_modality = 'ct'
    elif rgb_image_data is not None:
        st.session_state.uploaded_image = rgb_image_data
        st.session_state.image_path = rgb_filename
        st.session_state.image_modality = 'rgb'


def set_analysis_result(model_name: str, result: AnalysisResult):
    """분석 결과 저장"""
    st.session_state.analysis_results[model_name] = result


def get_analysis_result(model_name: str) -> Optional[AnalysisResult]:
    """분석 결과 조회"""
    return st.session_state.analysis_results.get(model_name)


def is_analysis_complete() -> bool:
    """모든 분석이 완료되었는지 확인"""
    results = st.session_state.analysis_results
    return all(r is not None for r in results.values())


def reset_analysis():
    """분석 상태 초기화"""
    # 기존 호환
    st.session_state.uploaded_image = None
    st.session_state.image_path = None
    st.session_state.image_modality = 'ct'

    # CT/RGB 분리 업로드
    st.session_state.ct_image = None
    st.session_state.ct_filename = None
    st.session_state.rgb_image = None
    st.session_state.rgb_filename = None
    st.session_state.analysis_mode = None

    # 분석 결과
    st.session_state.analysis_complete = False
    st.session_state.analysis_results = {
        'ensemble': None,
        'vlm': None,
        'vlg': None,
    }
    st.session_state.current_page = 'home'
