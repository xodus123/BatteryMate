# 배터리 검사 프로젝트 구현 구조 (Web 기반 + 통합 검사)

> **작성일**: 2025-12-28 (수정: 2026-01-05)
> **현재 Phase**: Phase 3 - CT CNN 학습 중 + VLM/VLG 구현 완료 + Webapp 구현 완료
> **기반 문서**: vision_pipeline_design.md + config_and_evaluation_design.md + inspector_design.md
> **핵심**: CT 통합 CNN + RGB AE 통합 검사 시스템 vs VLM/VLG 성능 비교

---

## 📋 설계 문서 핵심 분석

### 설계 핵심 구조

| 항목 | 설명 |
|------|------|
| **CT 통합 CNN** | Cell + Module 통합 5클래스 분류 (내부 결함 탐지) |
| **RGB AE** | AutoEncoder 기반 외부 결함 이상탐지 (정상 vs 불량) |
| **통합 검사 시스템** | CT CNN + RGB AE → 내부불량/외부불량 종합 판정 |
| **비교 대상** | VLM (Qwen3-VL), VLG (GroundingDINO) |
| **통합 검사 대상** | CT ∩ RGB 겹치는 74개 배터리 |
| **Config 관리** | YAML 파일 (통합 검사 가중치 포함) |

### 데이터 분할 구조 (2026-01-03 확정)

| 데이터셋 | 클래스 수 | 배터리 수 | Train | Val | Test |
|----------|-----------|-----------|-------|-----|------|
| **CT 통합** | 5 | 134 | 138,316 | 26,662 | 36,424 |
| **RGB** | 2 (이상탐지) | 300 (샘플) | 35,919 | 11,625 | 11,719 |
| **통합 검사** | - | 74 | 51/11/12 배터리 | - | - |

**CT 5클래스**: cell_normal, cell_porosity, module_normal, module_porosity, module_resin_overflow
**RGB 이상탐지**: normal vs defect (AutoEncoder 기반)

### 핵심 설계 철학

1. **"CT Cell + Module을 통합 CNN으로, RGB는 AE로 학습하고 통합 검사로 내부/외부 불량을 종합 판정한다."**
   - CT CNN: 내부 결함 분류 (Cell/Module 통합)
   - RGB AE: 외부 결함 탐지 (오염/손상)
   - 통합 검사: 두 결과 종합 → "내부불량" / "외부불량" / "복합불량" 판정

2. **"코드는 고정하고, 실험은 설정으로 바꾼다"**
   - 통합 검사 가중치, Threshold, metric은 **YAML 설정**
   - 동일 config → 동일 결과 재현 가능

3. **"실험은 설정으로, 판단은 지표로, 결과는 로그로 남긴다"**

---

## 🏗️ 전체 파이프라인 구조

```
[배터리 이미지 입력: CT + RGB]
           ↓
┌──────────────────────────────────────────────────────────┐
│  System 1: CT CNN + RGB AE 통합 검사                        │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  │
│                                                          │
│  [CT 이미지] ─────────────────→ [RGB 이미지]             │
│       ↓                              ↓                   │
│  ┌────────────────────┐    ┌─────────────────────┐      │
│  │  CT 통합 CNN        │    │  RGB AutoEncoder   │      │
│  │  (ResNet18)         │    │  (CAE)             │      │
│  │                     │    │                    │      │
│  │  5클래스 분류:      │    │  이상탐지 (Binary):│      │
│  │  - cell_normal      │    │  - normal          │      │
│  │  - cell_porosity    │    │  - defect          │      │
│  │  - module_normal    │    │                    │      │
│  │  - module_porosity  │    │  (Reconstruction   │      │
│  │  - module_resin_overflow │   Error 기반)     │      │
│  └────────────────────┘    └─────────────────────┘      │
│       ↓                              ↓                   │
│  ┌────────────────────┐    ┌─────────────────────┐      │
│  │  Grad-CAM          │    │  Anomaly Heatmap   │      │
│  │  → 내부 결함 위치   │    │  → 외부 결함 위치  │      │
│  └────────────────────┘    └─────────────────────┘      │
│       ↓                              ↓                   │
│  ┌────────────────────────────────────────────────┐     │
│  │           통합 검사 종합 판정 레이어              │     │
│  │  ─────────────────────────────────────────     │     │
│  │  CT 결과 + RGB 결과 → 최종 판정                │     │
│  │                                                │     │
│  │  출력 예시:                                    │     │
│  │  - "내부불량 (cell_porosity)" + Grad-CAM       │     │
│  │  - "외부불량" + Anomaly Heatmap                │     │
│  │  - "복합불량" + 양쪽 시각화                    │     │
│  └────────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────────┘
                         VS (비교)
┌──────────────────────────────────────────────────────────┐
│  System 2: VLM (Qwen2-VL)                                │
│  → Zero-shot 판정 + 불량 원인 설명                       │
│  → Grounding: 불량 위치 BBox 출력                        │
└──────────────────────────────────────────────────────────┘
                         VS
┌──────────────────────────────────────────────────────────┐
│  System 3: VLG (GroundingDINO)                           │
│  → 불량 유형별 BBox 검출                                 │
│  → Query: "porosity", "resin overflow", "pollution"      │
└──────────────────────────────────────────────────────────┘
           ↓
┌──────────────────────────────────────────────────────────┐
│  Web UI: 3개 시스템 결과 비교 시각화                     │
│  ─────────────────────────────────────────               │
│  [CT Grad-CAM] + [RGB Heatmap] = 종합 판정 뷰            │
└──────────────────────────────────────────────────────────┘
```

---

## 🗂️ 폴더 구조

### 현재 구현된 구조 (Phase 3)

```
battery-inspection/
├── CLAUDE.md                               # Claude 개발 가이드
├── TASK.md                                 # 작업 현황 (우선 참조!)
├── README.md
│
├── data -> /home/ubuntu/battery-data       # 심볼릭 링크 (원본 데이터)
│
├── docs/                                   # 문서
│   ├── implementation_structure.md         # 본 문서 (전체 설계)
│   ├── inspector_design.md                  # 통합 검사 설계
│   ├── MODEL_ARCHITECTURE.md               # 모델 아키텍처
│   └── TENSORBOARD_GUIDE.md                # TensorBoard 사용 가이드
│
├── models/                                 # ⭐ 모델 학습/추론
│   ├── ct_cnn/                            # CT 통합 CNN (5클래스)
│   │   ├── train.py                       # 학습 스크립트
│   │   ├── test.py                        # 평가 스크립트
│   │   ├── model.py                       # ResNet18 모델 정의
│   │   ├── checkpoints/                   # 체크포인트 저장
│   │   │   ├── ct_unified_best_*.pt      # Best 모델
│   │   │   └── ct_unified_last_*.pt      # 최신 모델
│   │   └── logs/                          # TensorBoard 로그
│   │
│   ├── rgb_ae/                            # RGB AutoEncoder (이상탐지)
│   │   └── checkpoints/                   # 체크포인트 저장
│   │
│   ├── vlm/                               # ✅ VLM (Qwen2-VL) - 구현 완료
│   │   ├── inference.py                   # VLM 추론 모듈
│   │   └── prompts.py                     # 프롬프트 템플릿 (5클래스)
│   │
│   └── vlg/                               # ✅ VLG (GroundingDINO) - 구현 완료
│       ├── inference.py                   # VLG 추론 모듈
│       ├── prompts.py                     # 프롬프트 템플릿 (5클래스 매핑)
│       └── weights/                       # 사전학습 가중치
│           └── groundingdino_swint_ogc.pth  # swinT 가중치 (662MB)
│
├── webapp/                                # ✅ Streamlit 웹앱 - 구현 완료
│   ├── app.py                             # 메인 앱 (페이지 라우팅)
│   ├── pages/                             # 페이지 컴포넌트
│   │   ├── home.py                        # 홈 (이미지 업로드)
│   │   ├── processing.py                  # 분석 진행 페이지
│   │   └── summary.py                     # 3-Way 비교 결과 페이지
│   └── utils/                             # 유틸리티
│       ├── session.py                     # 세션 상태 관리
│       ├── styles.py                      # CSS 스타일 (라이트 테마)
│       └── defect_info.py                 # 결함 정보 매핑 (5클래스)
│
├── scripts/                               # 유틸리티 스크립트
│   ├── create_splits_final.py             # 데이터 분할 생성
│   ├── check_data_leakage.py              # Data Leakage 검증
│   └── check_label_consistency.py         # 라벨 일관성 검증
│
├── training/                              # ⭐ 학습 관련 모듈
│   ├── configs/                           # YAML 설정 파일
│   │   ├── cnn_ct_unified.yaml           # CT CNN 학습 설정
│   │   ├── autoencoder_rgb.yaml          # RGB AE 학습 설정
│   │   ├── inspector.yaml                 # 통합 검사 설정
│   │   └── config_loader.py              # Config 로더
│   │
│   ├── data/                             # 데이터 처리
│   │   ├── dataset.py                    # BatteryDataset 클래스
│   │   ├── dataloader.py                 # DataLoader 팩토리
│   │   └── splits/                       # Train/Val/Test 분할 파일
│   │       ├── ct/                       # CT 데이터 분할
│   │       │   ├── train.txt             # 138,316개
│   │       │   ├── val.txt               # 26,662개
│   │       │   └── test.txt              # 36,424개
│   │       └── rgb/                      # RGB 데이터 분할
│   │           ├── train.txt
│   │           ├── val.txt
│   │           └── test.txt
│   │
│   ├── evaluation/                       # 평가 모듈
│   │   └── metrics.py                    # 메트릭 계산
│   │
│   └── visualization/                    # 시각화
│       └── tensorboard_logger.py         # TensorBoard 로거
│
└── .envrc                                # direnv 설정
```

### 향후 구현 예정 구조 (Phase 4~5)

```
battery-inspection/
├── (현재 구조...)
│
├── models/
│   └── inspector/                        # ⏳ Phase 4: 통합 검사 추론
│       ├── inference.py                 # CT CNN + RGB AE 통합 검사
│       └── gradcam.py                   # Grad-CAM 시각화
│
└── webapp/
    └── pages/
        └── processing.py                # ⏳ Phase 5: 실제 모델 연동
            # 현재: 더미 데이터
            # 목표: VLM/VLG/Ensemble 실제 호출
```

### ✅ 구현 완료된 항목 (기존 Phase 4~6)

| 기존 계획 | 상태 | 구현 위치 |
|-----------|------|-----------|
| VLM (Qwen2-VL) | ✅ 완료 | `models/vlm/inference.py` |
| VLG (GroundingDINO) | ✅ 완료 | `models/vlg/inference.py` |
| Streamlit UI | ✅ 완료 | `webapp/` |
| 5클래스 통일 | ✅ 완료 | `prompts.py`, `defect_info.py` |
| 결함 정보 매핑 | ✅ 완료 | `webapp/utils/defect_info.py` |

---

## 🤖 VLM/VLG 모델 상세

### VLM (Vision-Language Model)

| 항목 | 값 |
|------|-----|
| **모델** | Qwen2-VL (HuggingFace) |
| **지원 크기** | 2B, 7B, 72B |
| **기본값** | 7B (~16GB VRAM) |
| **출력** | 자연어 설명 + 분류 결과 |
| **프롬프트** | CT/RGB 분석용 한국어 프롬프트 |

```python
# 사용 예시
from models.vlm.inference import VLMInference
vlm = VLMInference(model_size='7b', device='cuda')
result = vlm.analyze_image('image.jpg', modality='ct')
```

### VLG (Vision-Language Grounding)

| 항목 | 값 |
|------|-----|
| **모델** | GroundingDINO |
| **지원 백본** | swinT (662MB), swinB (1GB) |
| **기본값** | swinT (~4GB VRAM) |
| **출력** | 바운딩 박스 + 라벨 + 신뢰도 |
| **가중치 경로** | `models/vlg/weights/groundingdino_swint_ogc.pth` |

```python
# 사용 예시
from models.vlg.inference import VLGInference
vlg = VLGInference(model_type='swinT', device='cuda')
result = vlg.detect('image.jpg', modality='ct')
```

### 5클래스 통일 체계

| 클래스 | 설명 | 심각도 |
|--------|------|--------|
| `cell_normal` | 정상 셀 | SUCCESS |
| `cell_porosity` | 셀 내부 기공 결함 | CRITICAL |
| `module_normal` | 정상 모듈 | SUCCESS |
| `module_porosity` | 모듈 내부 기공 결함 | CRITICAL |
| `module_resin_overflow` | 레진 오버플로우 | WARNING |

---

## 🌐 Webapp 구조

### 페이지 흐름

```
[Home] → [Processing] → [Summary]
  │          │             │
  │          │             └── 3-Way 비교 결과
  │          │                 - Ensemble 상세
  │          │                 - VLM 상세
  │          │                 - VLG 상세
  │          │                 - 최종 판정
  │          │
  │          └── 3개 모델 순차 분석
  │              - Ensemble (CNN+AE)
  │              - VLM (Qwen2-VL)
  │              - VLG (GroundingDINO)
  │
  └── 이미지 업로드 또는 Demo
```

### 실행 방법

```bash
# Webapp 실행
streamlit run webapp/app.py --server.port 8501

# 접속
http://localhost:8501
```

---

## 🎯 핵심 설계 원칙

### 1. Backend-Frontend 분리

**Backend (FastAPI)**
- 모델 추론만 담당
- RESTful API로 결과 제공
- 독립적으로 실행 및 테스트 가능

**Frontend (Streamlit)**
- 사용자 인터랙션
- API 호출 및 결과 시각화
- Backend 독립적 개발 가능

### 2. Training-Inference 분리

**Training 폴더**
- 모델 학습 스크립트
- TensorBoard 로깅
- 한 번 학습 후 체크포인트 저장

**Backend 폴더**
- 학습된 체크포인트 로드
- 추론만 수행
- 빠른 응답 시간 최적화

### 3. 통합 검사 통합 실행

```python
# backend/app/core/pipeline.py
class InferencePipeline:
    """통합 검사 시스템 + VLM/VLG 비교 실행"""

    def __init__(self):
        # 통합 검사 시스템 (CNN + AE + Grad-CAM)
        self.ensemble = EnsemblePredictor(config="training/configs/inspector.yaml")

        # 비교 대상
        self.vlm = VLMInference()
        self.vlg = VLGInference()

    async def run_all(self, ct_image: str, rgb_image: str) -> dict:
        """
        통합 검사 + VLM/VLG 비교 실행

        Args:
            ct_image: CT 이미지 경로
            rgb_image: RGB 이미지 경로

        Returns:
            {
                "ensemble": {...},  # CNN+AE+Grad-CAM 통합 결과
                "vlm": {...},       # VLM 독립 결과
                "vlg": {...}        # VLG 독립 결과
            }
        """
        results = await asyncio.gather(
            self.ensemble.predict(ct_image, rgb_image),  # 통합 검사
            self.vlm.predict(rgb_image),                 # VLM
            self.vlg.predict(rgb_image),                 # VLG
            return_exceptions=True
        )

        return {
            "ensemble": results[0],
            "vlm": results[1],
            "vlg": results[2]
        }
```

---

## 📊 JSON Schema 정의

### Request Schema

```python
# backend/app/schemas/request.py
from pydantic import BaseModel
from typing import List, Optional

class InferenceRequest(BaseModel):
    """단일/배치 추론 요청"""
    ct_image_path: str      # CT 이미지 경로
    rgb_image_path: str     # RGB 이미지 경로
    systems: Optional[List[str]] = ["ensemble", "vlm", "vlg"]  # 실행할 시스템 선택

class InferenceRequestUpload(BaseModel):
    """파일 업로드 요청"""
    ct_file: bytes
    rgb_file: bytes
```

### Response Schema

```python
# backend/app/schemas/response.py
from pydantic import BaseModel
from typing import List, Optional, Dict

class BoundingBox(BaseModel):
    x: float
    y: float
    w: float
    h: float

class EnsembleResult(BaseModel):
    """통합 검사 시스템 결과 (CNN + AE + Grad-CAM)"""
    # 최종 판정
    prediction: str  # "normal" or "defect"
    defect_type: Optional[str] = None  # "porosity", "resin_overflow", "pollution", "damaged"
    confidence: float  # 0~1

    # 개별 모델 기여도
    cnn_prob: float  # CNN 예측 확률
    cnn_defect_type: Optional[str] = None  # CNN이 예측한 불량 유형
    ae_score: float  # AE 이상 점수 (정규화됨)

    # 통합 검사 정보
    method: str  # "weighted_average", "voting", "rule_based"
    weights: Optional[Dict[str, float]] = None  # {"w_cnn": 0.6, "w_ae": 0.4}

    # Grad-CAM 위치 정보
    gradcam_heatmap: Optional[str] = None  # 히트맵 이미지 경로
    gradcam_bbox: Optional[List[BoundingBox]] = None  # 추출된 BBox

class VLMResult(BaseModel):
    """VLM 결과"""
    prediction: str  # "normal" or "defect"
    defect_type: Optional[str] = None  # 불량 유형 (VLM이 분석한)
    explanation: str  # 불량 원인 설명
    confidence: Optional[float] = None
    bbox: Optional[List[BoundingBox]] = None  # Grounding 위치 정보

class VLGResult(BaseModel):
    """VLG 결과"""
    bboxes: List[BoundingBox]
    scores: List[float]
    defect_types: List[str]  # 각 bbox별 불량 유형 ("porosity", "pollution", etc.)

class InferenceResponse(BaseModel):
    """3개 시스템 비교 결과"""
    image_id: str

    # System 1: 통합 검사
    ensemble: Optional[EnsembleResult] = None

    # System 2: VLM
    vlm: Optional[VLMResult] = None

    # System 3: VLG
    vlg: Optional[VLGResult] = None

    class Config:
        schema_extra = {
            "example": {
                "image_id": "battery_001",
                "ensemble": {
                    "prediction": "defect",
                    "defect_type": "porosity",
                    "confidence": 0.78,
                    "cnn_prob": 0.85,
                    "cnn_defect_type": "porosity",
                    "ae_score": 0.72,
                    "method": "weighted_average",
                    "weights": {"w_cnn": 0.6, "w_ae": 0.4},
                    "gradcam_heatmap": "/runs/job_xxx/heatmap.jpg",
                    "gradcam_bbox": [{"x": 120, "y": 80, "w": 200, "h": 150}]
                },
                "vlm": {
                    "prediction": "defect",
                    "defect_type": "porosity",
                    "explanation": "배터리 내부에 기공(porosity) 결함 발견. 전극 층 사이에 공극이 형성되어 있음.",
                    "confidence": 0.82,
                    "bbox": [{"x": 118, "y": 78, "w": 198, "h": 148}]
                },
                "vlg": {
                    "bboxes": [{"x": 115, "y": 85, "w": 205, "h": 145}],
                    "scores": [0.87],
                    "defect_types": ["porosity"]
                }
            }
        }
```

---

## 🌐 FastAPI 엔드포인트 설계

```python
# backend/app/api/inference.py
from fastapi import APIRouter, UploadFile, File
from app.schemas.request import InferenceRequest
from app.schemas.response import InferenceResponse
from app.core.pipeline import InferencePipeline

router = APIRouter()
pipeline = InferencePipeline()

@router.post("/infer", response_model=InferenceResponse)
async def infer_single(request: InferenceRequest):
    """단일 이미지 추론"""
    result = await pipeline.run_all(
        image_path=request.image_paths[0],
        modality=request.modality,
        models=request.models
    )
    return result

@router.post("/infer/batch", response_model=List[InferenceResponse])
async def infer_batch(request: InferenceRequest):
    """배치 이미지 추론"""
    results = []
    for img_path in request.image_paths:
        result = await pipeline.run_all(img_path, request.modality, request.models)
        results.append(result)
    return results

@router.post("/upload")
async def upload_and_infer(files: List[UploadFile] = File(...), modality: str = "ct"):
    """파일 업로드 + 추론"""
    # 파일 저장 후 추론
    pass
```

```python
# backend/app/api/model_info.py
@router.get("/models")
async def get_model_info():
    """모델 정보 조회"""
    return {
        "cnn": {
            "name": "ResNet50",
            "pretrained": "ImageNet-1K",
            "available_for": ["ct"]
        },
        "autoencoder": {
            "name": "ConvAutoencoder",
            "available_for": ["rgb", "ct"]
        },
        "vlm": {
            "name": "Qwen3-VL-8B-Instruct",
            "available_for": ["rgb", "ct"]
        },
        "vlg": {
            "name": "GroundingDINO",
            "available_for": ["rgb", "ct"]
        }
    }
```

---

## 🎨 Streamlit UI 구조

```python
# frontend/app.py
import streamlit as st
from components.uploader import ImageUploader
from components.result_viewer import ResultViewer
from utils.api_client import APIClient

st.set_page_config(page_title="배터리 불량 검사", layout="wide")

# 사이드바: 모달리티 선택
modality = st.sidebar.selectbox("데이터 타입", ["RGB", "CT"])

# 메인: 이미지 업로드
uploader = ImageUploader()
uploaded_files = uploader.render()

if uploaded_files:
    # API 호출
    client = APIClient()
    results = client.infer(uploaded_files, modality.lower())

    # 결과 표시
    viewer = ResultViewer()
    viewer.render(results)
```

```python
# frontend/components/result_viewer.py
import streamlit as st

class ResultViewer:
    def render(self, results):
        """모델별 결과 탭 표시"""
        tab1, tab2, tab3, tab4 = st.tabs(["CNN", "AutoEncoder", "VLM", "VLG"])

        with tab1:
            self._render_cnn(results["cnn"])

        with tab2:
            self._render_ae(results["autoencoder"])

        with tab3:
            self._render_vlm(results["vlm"])

        with tab4:
            self._render_vlg(results["vlg"])

    def _render_cnn(self, cnn_result):
        st.subheader("CNN 분류 결과")
        st.metric("판정", cnn_result["pred"])
        st.progress(cnn_result["confidence"])

    def _render_ae(self, ae_result):
        st.subheader("AutoEncoder 이상 감지")
        col1, col2 = st.columns(2)
        col1.metric("Anomaly Score", f"{ae_result['score']:.4f}")
        col2.metric("Threshold", f"{ae_result['threshold']:.4f}")

        # Score 분포 히스토그램 (배치 업로드 시)
        # ...

    def _render_vlm(self, vlm_result):
        st.subheader("VLM 설명")
        st.write(f"**판정**: {vlm_result['judgement']}")
        st.write(f"**이유**: {vlm_result['reason']}")

    def _render_vlg(self, vlg_result):
        st.subheader("VLG Bounding Box")
        # 이미지 위에 bbox 오버레이
        # ...
```

---

## 🔄 실행 흐름

### 1. 모델 학습 (1회)

```bash
# 1. CT CNN 학습
cd training
python scripts/train_cnn_ct.py

# 2. RGB AutoEncoder 학습
python scripts/train_ae_rgb.py

# 3. CT AutoEncoder 학습
python scripts/train_ae_ct.py

# 4. TensorBoard로 학습 과정 확인
tensorboard --logdir ../experiments/runs
```

### 2. Backend 서버 실행

```bash
cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Frontend 실행

```bash
cd frontend
streamlit run app.py
```

### 4. 사용자 워크플로우

1. 웹 브라우저에서 Streamlit 접속 (http://localhost:8501)
2. 모달리티 선택 (RGB/CT)
3. 이미지 업로드 (단일/배치)
4. 실행할 모델 선택 (CNN, AE, VLM, VLG)
5. 결과 탭별로 확인
   - CNN: 분류 결과 + Confidence
   - AE: Anomaly Score + Threshold
   - VLM: 텍스트 설명
   - VLG: Bounding Box 오버레이
6. 배치 업로드 시: Anomaly Score 히스토그램

---

## 📦 주요 의존성

### Backend (backend/requirements.txt)

```txt
# FastAPI
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
pydantic>=2.4.0
python-multipart>=0.0.6

# Deep Learning
torch>=2.0.0
torchvision>=0.15.0
timm>=0.9.0

# VLM/VLG
transformers>=4.35.0
# groundingdino (설치 방법 별도)

# 이미지 처리
pillow>=10.0.0
opencv-python>=4.8.0

# 유틸리티
python-dotenv>=1.0.0
```

### Frontend (frontend/requirements.txt)

```txt
streamlit>=1.28.0
requests>=2.31.0
pillow>=10.0.0
matplotlib>=3.7.0
plotly>=5.17.0  # 인터랙티브 그래프
```

### Training (training/requirements.txt)

```txt
torch>=2.0.0
torchvision>=0.15.0
timm>=0.9.0
tensorboard>=2.13.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
```

---

## 🚀 구현 우선순위

### Phase 1: Backend 기본 구조 (2-3일)
1. ✅ 폴더 구조 생성
2. ✅ Config 파일 작성
3. ✅ Pydantic Schema 정의
4. ✅ FastAPI 기본 엔드포인트 구현 (mock 응답)
5. ✅ Health check API

### Phase 2: Training (3-4일)
6. ✅ Dataset/DataLoader 구현
7. ✅ CNN (CT) 학습
8. ✅ AutoEncoder (RGB) 학습
9. ✅ Threshold 자동 계산
10. ✅ 체크포인트 저장

### Phase 3: Backend 모델 연동 + Grad-CAM (3-4일)
11. ✅ CNN Predictor 구현
12. ⭐ Grad-CAM 모듈 구현 (pytorch-grad-cam 라이브러리)
13. ⭐ Heatmap → BBox 추출 함수 구현
14. ⭐ 시각화 함수 구현 (heatmap overlay, bbox overlay)
15. ⭐ CNN Predictor에 Grad-CAM 통합
16. ✅ AE Predictor 구현
17. ✅ InferencePipeline 구현
18. ✅ 실제 추론 API 연결

### Phase 4: Frontend (2-3일)
19. ✅ Streamlit 기본 UI
20. ✅ 이미지 업로드 컴포넌트
21. ✅ API 클라이언트
22. ✅ 모델별 결과 탭
23. ⭐ CNN Grad-CAM 시각화 (heatmap + bbox overlay)
24. ✅ VLM/VLG Bounding Box 오버레이

### Phase 5: VLM/VLG (선택적, 3-4일)
25. ⏸️ VLM (Qwen3-VL) 로컬 추론
26. ⏸️ VLG (GroundingDINO) 연동
27. ⏸️ Frontend에 결과 연동

### Phase 6: 배치 처리 & 고도화 (2-3일)
28. ⏸️ 배치 추론 최적화
29. ⏸️ Anomaly Score 히스토그램
30. ⏸️ 결과 저장/로드 기능

---

## 🎯 핵심 차별점

### 1. Web 기반 실시간 비교
- TensorBoard: 학습 과정만
- Streamlit: 추론 결과 실시간 비교

### 2. 사용자 친화적
- CLI 대신 웹 UI
- 이미지 업로드만으로 모든 모델 결과 확인

### 3. 확장 가능한 아키텍처
- Backend-Frontend 분리
- Training-Inference 분리
- 새 모델 추가 용이

### 4. 결정적(Deterministic) 파이프라인
- 모든 모델 독립 실행
- 재현 가능한 결과

---

## 📊 TensorBoard vs Web UI 역할 분리

### 핵심 원칙
> **"학습 과정은 TensorBoard로, 최종 결과는 로그 기반 Web UI로 본다."**

- TensorBoard ≠ 서비스 UI
- Web UI ≠ 학습 모니터링 도구
- **두 시스템은 같은 로그를 공유하지 않는다**

### 역할 분담표

| 항목 | TensorBoard | Web UI |
|------|-------------|--------|
| **학습 모니터링** | ⭕ 주 용도 | ❌ |
| **실험 비교** | ⭕ 학습 과정 비교 | ⭕ 최종 결과 비교 |
| **이미지별 결과** | ❌ | ⭕ 주 용도 |
| **사용자 입력** | ❌ | ⭕ 이미지 업로드 |
| **서비스 확장** | ❌ | ⭕ 배치 처리, API |

### TensorBoard 기록 대상 (학습용)

```python
# training/visualization/tensorboard_logger.py
class TensorBoardLogger:
    def log_training_metrics(self, epoch):
        """학습 과정 모니터링"""
        self.writer.add_scalar('train/loss', train_loss, epoch)
        self.writer.add_scalar('val/loss', val_loss, epoch)
        self.writer.add_scalar('val/f1', val_f1, epoch)
        self.writer.add_scalar('lr', current_lr, epoch)

        # AutoEncoder: 재구성 이미지
        if epoch % 5 == 0:
            self.writer.add_image('reconstruction', recon_img, epoch)

        # 재구성 오류 분포 (Threshold 결정용)
        self.writer.add_histogram('recon_error', errors, epoch)
```

**기록 목적**:
- 수렴 여부 확인
- 과적합 판단
- 실험 간 비교
- Threshold 결정 (히스토그램)

### Web UI가 읽는 데이터 (추론 결과용)

| 파일 | 경로 | 용도 |
|------|------|------|
| `result.json` | `experiments/results/job_xxx/` | 이미지별 모델 결과 |
| `summary.csv` | `experiments/results/job_xxx/` | Job 단위 metric 요약 |
| `images/` | `experiments/results/job_xxx/images/` | bbox overlay, 시각화 |

**result.json 예시**:
```json
{
  "job_id": "job_20250101_001",
  "timestamp": "2025-01-01T10:30:00",
  "total_images": 50,
  "images": [
    {
      "image_id": "img_001.jpg",
      "cnn": {
        "pred": "defect",
        "confidence": 0.92
      },
      "autoencoder": {
        "score": 0.034,
        "threshold": 0.028,
        "is_anomaly": true
      },
      "vlm": {
        "judgement": "defect",
        "reason": "표면 크랙 발견"
      },
      "vlg": {
        "bboxes": [{"x": 120, "y": 30, "w": 240, "h": 160}],
        "scores": [0.87]
      }
    }
  ]
}
```

**summary.csv 예시**:
```csv
job_id,total_images,defect_count,normal_count,avg_confidence,processing_time
job_20250101_001,50,12,38,0.89,45.2
```

### 실험(run) 이름 규칙

**TensorBoard run 이름 형식**:
```text
<model>_<key_param>_<value>_<key_param>_<value>
```

**예시**:
- `resnet18_lr1e-4_bs32` - CNN 실험
- `resnet50_lr5e-5_bs16_wd1e-4` - CNN 실험 (weight decay 추가)
- `ae_rgb_k2.5` - RGB AutoEncoder (k=2.5)
- `ae_ct_k2.0_latent128` - CT AutoEncoder (k=2.0, latent_dim=128)

**장점**:
- Run 이름만 봐도 실험 조건 파악 가능
- TensorBoard에서 실험 비교 시 직관적

---

## 🔄 비동기 Job 처리 흐름 (run_id 기반)

### 핵심 원칙
> **"업로드 → 처리 → 결과 보기" 완전 비동기 구조**

- 업로드와 처리 분리 → 사용자 대기 시간 ❌
- 백그라운드에서 추론 수행
- 상태 조회로 진행 상황 확인
- **DB 없이 파일 기반으로 가능**

### 전체 흐름

```text
[Frontend] 이미지 업로드
   ↓
[Backend] POST /upload
   ↓
run_id 생성 (timestamp)
run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
   ↓
experiments/runs/{run_id}/ 생성
   ├─ status.json (pending)
   └─ input_images/ (이미지 저장)
   ↓
즉시 run_id 반환 → Frontend
   ↓
[Backend] BackgroundTasks로 추론 실행
   ├─ status.json → "processing" 업데이트
   ├─ CNN, AE, VLM, VLG 병렬 실행
   ├─ results/ 에 모델별 결과 저장
   ├─ summary.json 생성
   └─ status.json → "completed" 업데이트
   ↓
[Frontend] Polling으로 상태 확인
GET /jobs/{run_id}/status (2초마다)
   ↓
status = "completed" 감지
   ↓
[Frontend] 결과 조회
GET /jobs/{run_id}/results
   ↓
시각화 (모델별 탭, 이미지별 결과)
```

### Job 상태 관리 (status.json)

**파일 경로**: `experiments/runs/{run_id}/status.json`

```json
{
  "run_id": "20250101_143210",
  "status": "processing",
  "created_at": "2025-01-01T14:32:10",
  "updated_at": "2025-01-01T14:32:15",
  "total_images": 10,
  "processed_images": 3,
  "progress_percent": 30,
  "error_message": null
}
```

**상태 값**:
- `pending`: 업로드 완료, 처리 대기
- `processing`: 추론 진행 중
- `completed`: 모든 처리 완료
- `failed`: 에러 발생

### Backend API 엔드포인트

```python
# backend/app/api/jobs.py
from fastapi import APIRouter, UploadFile, File, BackgroundTasks
from typing import List
import json
from datetime import datetime
from pathlib import Path

router = APIRouter(prefix="/jobs", tags=["jobs"])

@router.post("/upload")
async def upload_images(
    files: List[UploadFile] = File(...),
    modality: str = "ct",
    background_tasks: BackgroundTasks = None
):
    """
    이미지 업로드 + 비동기 처리 시작
    - run_id 즉시 반환
    - 백그라운드에서 추론 실행
    """
    # 1. run_id 생성
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = Path(f"experiments/runs/{run_id}")
    run_dir.mkdir(parents=True, exist_ok=True)

    # 2. 이미지 저장
    input_dir = run_dir / "input_images"
    input_dir.mkdir(exist_ok=True)

    image_paths = []
    for file in files:
        file_path = input_dir / file.filename
        with open(file_path, "wb") as f:
            f.write(await file.read())
        image_paths.append(str(file_path))

    # 3. 초기 상태 저장
    status_data = {
        "run_id": run_id,
        "status": "pending",
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "total_images": len(files),
        "processed_images": 0,
        "progress_percent": 0,
        "error_message": None
    }

    with open(run_dir / "status.json", "w") as f:
        json.dump(status_data, f, indent=2)

    # 4. 백그라운드 작업 등록
    background_tasks.add_task(
        process_inference,
        run_id=run_id,
        image_paths=image_paths,
        modality=modality
    )

    # 5. 즉시 run_id 반환
    return {
        "run_id": run_id,
        "status": "pending",
        "message": "추론이 백그라운드에서 진행됩니다."
    }


@router.get("/{run_id}/status")
async def get_job_status(run_id: str):
    """Job 상태 조회 (Frontend polling용)"""
    status_path = Path(f"experiments/runs/{run_id}/status.json")

    if not status_path.exists():
        return {"error": "Run not found"}, 404

    with open(status_path) as f:
        status = json.load(f)

    return status


@router.get("/{run_id}/results")
async def get_job_results(run_id: str):
    """완료된 Job의 결과 조회"""
    run_dir = Path(f"experiments/runs/{run_id}")

    # 1. 상태 확인
    with open(run_dir / "status.json") as f:
        status = json.load(f)

    if status["status"] != "completed":
        return {
            "error": f"Job not completed yet. Current status: {status['status']}"
        }, 400

    # 2. summary.json 로드
    with open(run_dir / "summary.json") as f:
        summary = json.load(f)

    # 3. 모델별 결과 로드 (선택적)
    results_dir = run_dir / "results"
    model_results = {}

    for result_file in results_dir.glob("*_result.json"):
        model_name = result_file.stem.replace("_result", "")
        with open(result_file) as f:
            model_results[model_name] = json.load(f)

    return {
        "run_id": run_id,
        "status": status,
        "summary": summary,
        "results": model_results
    }


async def process_inference(run_id: str, image_paths: List[str], modality: str):
    """
    백그라운드 추론 작업
    - 상태 업데이트
    - 모델별 추론
    - 결과 저장
    """
    run_dir = Path(f"experiments/runs/{run_id}")

    try:
        # 1. 상태 → processing
        update_status(run_id, "processing", processed=0)

        # 2. 모델 로드
        pipeline = InferencePipeline(modality=modality)

        # 3. 이미지별 추론
        all_results = []
        for idx, img_path in enumerate(image_paths):
            result = await pipeline.run_all(img_path, modality)
            all_results.append(result)

            # 진행 상황 업데이트
            progress = int((idx + 1) / len(image_paths) * 100)
            update_status(run_id, "processing", processed=idx+1, progress=progress)

        # 4. 모델별 결과 저장
        results_dir = run_dir / "results"
        results_dir.mkdir(exist_ok=True)

        # CNN 결과
        cnn_results = [r.cnn for r in all_results if r.cnn]
        with open(results_dir / "cnn_result.json", "w") as f:
            json.dump([r.dict() for r in cnn_results], f, indent=2)

        # AE 결과
        ae_results = [r.autoencoder for r in all_results if r.autoencoder]
        with open(results_dir / "ae_result.json", "w") as f:
            json.dump([r.dict() for r in ae_results], f, indent=2)

        # 5. summary.json 생성
        defect_count = sum(1 for r in all_results if r.cnn and r.cnn.pred == "defect")
        summary = {
            "run_id": run_id,
            "total_images": len(image_paths),
            "defect_count": defect_count,
            "normal_count": len(image_paths) - defect_count,
            "completed_at": datetime.now().isoformat()
        }

        with open(run_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        # 6. 상태 → completed
        update_status(run_id, "completed", processed=len(image_paths), progress=100)

    except Exception as e:
        # 에러 시 상태 업데이트
        update_status(run_id, "failed", error=str(e))


def update_status(run_id: str, status: str, processed: int = None, progress: int = None, error: str = None):
    """상태 업데이트 헬퍼"""
    status_path = Path(f"experiments/runs/{run_id}/status.json")

    with open(status_path) as f:
        data = json.load(f)

    data["status"] = status
    data["updated_at"] = datetime.now().isoformat()

    if processed is not None:
        data["processed_images"] = processed
    if progress is not None:
        data["progress_percent"] = progress
    if error is not None:
        data["error_message"] = error

    with open(status_path, "w") as f:
        json.dump(data, f, indent=2)
```

### Frontend Polling 로직 (Streamlit)

```python
# frontend/app.py
import streamlit as st
import requests
import time
from pathlib import Path

API_BASE = "http://localhost:8000"

st.set_page_config(page_title="배터리 불량 검사", layout="wide")

# 사이드바: 이전 run 선택
st.sidebar.subheader("이전 결과 불러오기")
run_dirs = list(Path("experiments/runs").glob("*"))
run_ids = [d.name for d in sorted(run_dirs, reverse=True)]

selected_run = st.sidebar.selectbox("Run 선택", ["새로운 추론"] + run_ids)

if selected_run == "새로운 추론":
    # 1. 이미지 업로드
    st.header("이미지 업로드")
    uploaded_files = st.file_uploader("이미지 선택", accept_multiple_files=True, type=["jpg", "jpeg", "png"])
    modality = st.selectbox("데이터 타입", ["RGB", "CT"])

    if uploaded_files and st.button("추론 시작"):
        # 2. Backend에 업로드
        files = [("files", (f.name, f, "image/jpeg")) for f in uploaded_files]
        response = requests.post(
            f"{API_BASE}/jobs/upload",
            files=files,
            params={"modality": modality.lower()}
        )

        if response.status_code == 200:
            data = response.json()
            run_id = data["run_id"]

            st.success(f"✅ Run ID: {run_id}")
            st.info("📊 추론이 백그라운드에서 진행 중입니다...")

            # 3. Polling으로 상태 확인
            progress_bar = st.progress(0)
            status_text = st.empty()

            while True:
                # 상태 조회
                status_response = requests.get(f"{API_BASE}/jobs/{run_id}/status")
                status_data = status_response.json()

                # 진행률 업데이트
                progress = status_data.get("progress_percent", 0)
                progress_bar.progress(progress / 100)
                status_text.text(f"처리 중... {status_data['processed_images']}/{status_data['total_images']} ({progress}%)")

                # 완료 확인
                if status_data["status"] == "completed":
                    st.success("✅ 추론 완료!")
                    break
                elif status_data["status"] == "failed":
                    st.error(f"❌ 에러 발생: {status_data.get('error_message')}")
                    break

                # 2초 대기
                time.sleep(2)

            # 4. 결과 로드
            if status_data["status"] == "completed":
                results_response = requests.get(f"{API_BASE}/jobs/{run_id}/results")
                results = results_response.json()

                # 결과 표시
                display_results(results)

else:
    # 기존 결과 로드
    st.header(f"Run: {selected_run}")

    # API를 통해 결과 조회
    results_response = requests.get(f"{API_BASE}/jobs/{selected_run}/results")

    if results_response.status_code == 200:
        results = results_response.json()
        display_results(results)
    else:
        st.error("결과를 불러올 수 없습니다.")


def display_results(results):
    """결과 시각화"""
    st.subheader("📊 요약")

    col1, col2, col3 = st.columns(3)
    summary = results["summary"]
    col1.metric("전체 이미지", summary["total_images"])
    col2.metric("불량", summary["defect_count"])
    col3.metric("정상", summary["normal_count"])

    # 모델별 결과 탭
    tabs = st.tabs(["CNN", "AutoEncoder", "VLM", "VLG"])

    with tabs[0]:
        st.subheader("CNN 분류 결과")
        # CNN 결과 표시...

    with tabs[1]:
        st.subheader("AutoEncoder 이상 감지")
        # AE 결과 표시...

    # ...
```

---

## 🔧 모델별 Inference 로직 (Threshold Config 로딩)

### 핵심 원칙
> **"Inference에서 Threshold를 직접 쓰면 안 되고 반드시 config 파일에서 로드"**

- 하드코딩 ❌
- 학습 시 저장된 `threshold.json` 로드 ✅
- 재현 가능성 확보

### AutoEncoder Predictor (Threshold 로딩)

```python
# backend/app/models/autoencoder/predictor.py
import torch
import json
from pathlib import Path
from typing import Dict

class AEPredictor:
    """AutoEncoder 추론기 - Threshold Config 로딩"""

    def __init__(self, modality: str):
        """
        Args:
            modality: "rgb" or "ct"
        """
        self.modality = modality
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1. 모델 로드
        checkpoint_path = Path(f"experiments/checkpoints/autoencoder/ae_{modality}.pt")
        self.model = self._load_model(checkpoint_path)
        self.model.eval()

        # 2. ⭐ Threshold 로드 (config 파일에서!)
        threshold_path = Path(f"experiments/checkpoints/autoencoder/ae_{modality}_threshold.json")
        self.threshold_config = self._load_threshold(threshold_path)

        print(f"✅ AE ({modality}) 로드 완료")
        print(f"  - Threshold: {self.threshold_config['threshold']:.4f}")
        print(f"  - Method: {self.threshold_config['method']}")

    def _load_model(self, path: Path):
        """체크포인트 로드"""
        checkpoint = torch.load(path, map_location=self.device)
        model = AutoEncoderModel()  # 모델 아키텍처 정의
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        return model

    def _load_threshold(self, path: Path) -> Dict:
        """⭐ Threshold Config 로드 (필수)"""
        if not path.exists():
            raise FileNotFoundError(
                f"Threshold 파일이 없습니다: {path}\n"
                f"학습 후 반드시 threshold.json을 생성해야 합니다."
            )

        with open(path) as f:
            config = json.load(f)

        # 필수 필드 검증
        required_fields = ["threshold", "method", "mean_error", "std_error"]
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Threshold config에 {field} 필드가 없습니다.")

        return config

    async def predict(self, image_path: str) -> Dict:
        """추론 실행"""
        # 1. 이미지 로드 + 전처리
        image_tensor = self._preprocess(image_path)

        # 2. 재구성
        with torch.no_grad():
            reconstructed = self.model(image_tensor)

        # 3. 재구성 오류 계산
        error = torch.nn.functional.mse_loss(image_tensor, reconstructed).item()

        # 4. ⭐ Threshold와 비교 (config에서 로드한 값 사용!)
        threshold = self.threshold_config["threshold"]
        is_anomaly = error > threshold

        return {
            "score": float(error),
            "threshold": float(threshold),
            "is_anomaly": bool(is_anomaly),
            "method": self.threshold_config["method"],  # 정보 제공
            "k": self.threshold_config.get("k")  # mean_std 방식일 경우
        }
```

### CNN Predictor (Grad-CAM 통합)

```python
# backend/app/models/cnn/predictor.py
import torch
from pathlib import Path
from typing import Dict, Optional
from .gradcam import GradCAMGenerator
from .bbox_extractor import extract_bboxes_from_heatmap
from .visualizer import visualize_gradcam

class CNNPredictor:
    """CNN 추론기 + Grad-CAM 위치 정보"""

    def __init__(self, modality: str = "ct", enable_gradcam: bool = True):
        """
        CNN은 CT 데이터만 지원

        Args:
            modality: "ct"만 지원
            enable_gradcam: Grad-CAM 활성화 여부
        """
        if modality != "ct":
            raise ValueError("CNN은 CT 데이터만 지원합니다.")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.enable_gradcam = enable_gradcam

        # 1. 모델 로드
        checkpoint_path = Path("experiments/checkpoints/cnn/resnet18_best.pt")
        self.model = self._load_model(checkpoint_path)
        self.model.eval()

        # 2. ⭐ Grad-CAM 초기화
        if self.enable_gradcam:
            self.gradcam_generator = GradCAMGenerator(
                model=self.model,
                target_layer=self.model.layer4[-1]  # ResNet 마지막 Conv 레이어
            )

        print("✅ CNN (CT) 로드 완료")
        if self.enable_gradcam:
            print("  - Grad-CAM 활성화")

    def _load_model(self, path: Path):
        """체크포인트 로드"""
        checkpoint = torch.load(path, map_location=self.device)
        model = ResNet18(num_classes=1)  # BCEWithLogitsLoss
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        return model

    async def predict(self, image_path: str, save_visualization: bool = True) -> Dict:
        """
        추론 실행 + Grad-CAM 위치 정보

        Args:
            image_path: 이미지 경로
            save_visualization: 시각화 이미지 저장 여부

        Returns:
            {
                "pred": "defect" or "normal",
                "confidence": 0.95,
                "bboxes": [{"x": 120, "y": 340, "w": 50, "h": 80, "score": 0.92}],
                "visualization": {"heatmap": "path/to/heatmap.jpg", ...}
            }
        """
        # 1. 이미지 전처리
        image_tensor = self._preprocess(image_path)
        original_image = self._load_original_image(image_path)

        # 2. 추론
        with torch.no_grad():
            logit = self.model(image_tensor)
            prob = torch.sigmoid(logit)
            confidence = prob.item()

        # 3. 결과 판정
        pred_label = "defect" if confidence > 0.4 else "normal"

        result = {
            "pred": pred_label,
            "confidence": float(confidence)
        }

        # 4. ⭐ Grad-CAM 생성 (불량일 경우에만)
        if self.enable_gradcam and pred_label == "defect":
            # Heatmap 생성
            heatmap = self.gradcam_generator.generate(
                image_tensor,
                target_class=1  # defect class
            )

            # BBox 추출
            bboxes = extract_bboxes_from_heatmap(
                heatmap=heatmap,
                threshold=0.5,  # Heatmap threshold
                min_area=100
            )

            result["bboxes"] = bboxes

            # 5. 시각화 저장 (선택적)
            if save_visualization and bboxes:
                vis_paths = visualize_gradcam(
                    original_image=original_image,
                    heatmap=heatmap,
                    bboxes=bboxes,
                    image_name=Path(image_path).stem,
                    save_dir=Path("experiments/runs/current/visualizations")
                )
                result["visualization"] = vis_paths

        return result
```

### InferencePipeline (모델 독립 실행)

```python
# backend/app/core/pipeline.py
import asyncio
from app.models.cnn.predictor import CNNPredictor
from app.models.autoencoder.predictor import AEPredictor
from app.models.vlm.inference import VLMInference
from app.models.vlg.inference import VLGInference

class InferencePipeline:
    """모델별 독립 실행 파이프라인"""

    def __init__(self, modality: str):
        """
        Args:
            modality: "rgb" or "ct"
        """
        self.modality = modality

        # ⭐ 모델 초기화 (각자 독립적으로 설정 로드)
        if modality == "ct":
            self.cnn = CNNPredictor(modality="ct")
        else:
            self.cnn = None  # RGB는 CNN 미지원

        self.ae = AEPredictor(modality=modality)  # Threshold 자동 로드
        self.vlm = VLMInference()  # 선택적
        self.vlg = VLGInference()  # 선택적

    async def run_all(self, image_path: str, modality: str):
        """모든 모델 병렬 실행"""
        tasks = []

        # 1. CNN (CT만)
        if self.cnn is not None:
            tasks.append(self.cnn.predict(image_path))
        else:
            tasks.append(self._return_none())  # 플레이스홀더

        # 2. AutoEncoder (항상)
        tasks.append(self.ae.predict(image_path))

        # 3. VLM/VLG (선택적)
        if self.vlm:
            tasks.append(self.vlm.predict(image_path))
        else:
            tasks.append(self._return_none())

        if self.vlg:
            tasks.append(self.vlg.predict(image_path))
        else:
            tasks.append(self._return_none())

        # 병렬 실행
        cnn_result, ae_result, vlm_result, vlg_result = await asyncio.gather(
            *tasks,
            return_exceptions=True
        )

        # 결과 통합
        return {
            "image_id": Path(image_path).name,
            "cnn": cnn_result if not isinstance(cnn_result, Exception) else None,
            "autoencoder": ae_result if not isinstance(ae_result, Exception) else None,
            "vlm": vlm_result if not isinstance(vlm_result, Exception) else None,
            "vlg": vlg_result if not isinstance(vlg_result, Exception) else None,
        }

    async def _return_none(self):
        """플레이스홀더"""
        return None
```

### Threshold 관리 흐름 정리

```text
[학습 시]
1. AutoEncoder 학습 완료
   ↓
2. Validation 데이터로 재구성 오류 계산
   ↓
3. Threshold 계산 (mean + k * std)
   ↓
4. threshold.json 저장
   {
     "threshold": 0.0285,
     "method": "mean_std",
     "k": 2.5,
     "mean_error": 0.0198,
     "std_error": 0.0035
   }
   ↓
5. 모델 (.pt) + threshold.json 함께 저장

[추론 시]
1. AEPredictor 초기화
   ↓
2. ⭐ threshold.json 로드 (자동)
   ↓
3. 추론 시 로드된 threshold 사용
   ↓
4. 하드코딩 ❌, 재현 가능 ✅
```

---

## 🎯 왜 이 구조가 좋은가?

### 1. 명확한 역할 분리
- **실험 단계**: TensorBoard로 학습 과정 추적
- **서비스 단계**: Web UI로 최종 결과 시각화

### 2. 확장 가능성
```text
Phase 1 (현재): 로그 기반 Web UI
  - result.json 읽어서 표시
  - DB 없이 동작

Phase 2 (확장): DB 기반 Web UI
  - result.json → PostgreSQL 저장
  - Web UI는 동일 인터페이스 유지
  - 검색, 필터링, 통계 기능 추가
```

### 3. TensorBoard 로그가 지저분해지지 않음
- 학습용 로그와 추론용 로그 완전 분리
- 실험 조건이 run 이름에 명시적으로 포함

### 4. DB 없이도 웹 시각화 가능
- 파일 기반 시스템으로 간단하게 시작
- 추후 DB 마이그레이션 용이

---

## ⚙️ Config 파일 설계 (YAML)

### 1. CNN 학습 Config (training/configs/cnn.yaml)

```yaml
model:
  name: resnet18  # resnet18, resnet50, convnext_tiny
  pretrained: true
  num_classes: 2

training:
  optimizer: Adam
  lr: 0.0001
  batch_size: 32
  epochs: 30
  weight_decay: 0.0001
  device: cuda

criteria:
  loss: CrossEntropy
  early_stopping:
    monitor: val_loss
    patience: 7
    min_delta: 0.001

checkpoint:
  save_best_by: val_f1  # val_f1, val_accuracy
  save_dir: experiments/checkpoints/cnn
```

**설계 의도**:
- Early stopping과 best model 기준 분리
- Baseline → 확장 실험 모두 동일 구조 사용
- F1을 기준으로 최고 모델 저장

---

### 2. AutoEncoder Config (training/configs/autoencoder.yaml)

```yaml
model:
  type: convolutional_autoencoder
  input_channels: 3
  latent_dim: 128  # Bottleneck size

training:
  optimizer: Adam
  lr: 0.001
  batch_size: 32
  epochs: 50
  device: cuda

criteria:
  loss: MSE

threshold:
  mode: fixed  # fixed, adaptive
  method: mean_std  # mean_std, percentile, f1_max
  k: 2.5  # mean + k * std (method가 mean_std일 때)
  percentile: 95  # method가 percentile일 때

checkpoint:
  save_dir: experiments/checkpoints/autoencoder
  save_threshold: true  # Threshold도 함께 저장
```

**설계 의도**:
- 초기 구현은 **fixed threshold** (mean + k * std)
- Threshold 계산은 학습 종료 후 1회 수행
- Phase 2에서 adaptive (f1_max) 방식으로 확장 가능

---

### 3. Evaluation Config (training/configs/evaluation.yaml)

```yaml
metrics:
  primary: f1  # 모든 의사결정의 기준
  secondary:
    - accuracy
    - precision
    - recall
    - roc_auc

cnn:
  decision_metric: f1

autoencoder:
  threshold_metric: f1  # Threshold 결정 시 최적화할 지표

reporting:
  save_confusion_matrix: true
  save_roc_curve: true
  save_dir: experiments/results
```

**설계 의도**:
- **F1-score**를 모든 의사결정의 기준으로 사용
- Accuracy는 참고 지표로만 활용
- 클래스 불균형 대응

---

### 4. Logging Config (training/configs/logging.yaml)

```yaml
logging:
  save_train_log: true
  save_inference_log: false  # 추론 로그는 Backend에서 관리
  log_level: INFO

paths:
  train_logs: experiments/logs/train_logs/
  inference_logs: experiments/logs/inference_logs/  # Backend에서 사용

format:
  train: csv  # epoch, train_loss, val_loss, val_f1, val_accuracy
  inference: json  # 전체 추론 결과

tensorboard:
  enabled: true
  log_dir: experiments/logs/tensorboard/  # ⭐ TensorBoard 전용 경로
  run_name_format: "{model}_{key_params}"  # 예: resnet18_lr1e-4_bs32
  log_scalars: true  # Loss, F1, Accuracy
  log_images: true   # AutoEncoder 재구성 결과
  log_histograms: true  # 재구성 오류 분포 (Threshold 결정용)
  log_embeddings: false  # Phase 2

results:
  save_dir: experiments/results/  # ⭐ Web UI용 결과 (job_id 기반)
  save_visualization: true  # bbox overlay 등 이미지 저장
```

**설계 의도**:
- **학습 로그**: CSV (간단, 가독성) → `experiments/logs/train_logs/`
- **추론 로그**: JSON (구조화, API 호환) → `experiments/logs/inference_logs/`
- **TensorBoard**: 학습 시에만 사용 → `experiments/logs/tensorboard/`
- **Web 결과**: job_id 기반 → `experiments/results/job_xxx/`

---

## 📊 Evaluation 기준

### 1. 평가 지표 우선순위

| 지표 | 역할 | 사용처 |
|-----|-----|--------|
| **F1 Score** | 주 평가 지표 | 모델 저장, Threshold 결정, 최종 평가 |
| Accuracy | 참고 지표 | 리포트용 |
| Precision | 참고 지표 | 불량 검출 정확도 분석 |
| **Recall** | 중요 지표 | 불량 미탐 최소화 확인 |
| ROC-AUC | 참고 지표 | 모델 간 비교 |

### 2. 평가 흐름

```text
학습
 ↓
val_loss로 Early Stopping
 ↓
val_F1 최고 모델 저장
 ↓
고정 Threshold 적용 (AutoEncoder)
 ↓
테스트셋에서 Accuracy / F1 / Recall / ROC-AUC 리포트
```

### 3. CNN 평가 기준

```python
# training/evaluation/metrics.py
def evaluate_cnn(model, test_loader):
    """
    CNN 평가
    - 주 지표: F1 Score
    - Early Stopping: val_loss
    - Best Model: val_f1
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_proba)
    }
    return metrics
```

### 4. AutoEncoder 평가 기준

```python
# training/evaluation/threshold_finder.py
def find_threshold(ae_model, val_loader, method='mean_std', k=2.5):
    """
    Threshold 계산
    - Phase 1: mean + k * std (고정)
    - Phase 2: F1 최대화 (적응형)
    """
    reconstruction_errors = []
    for img in val_loader:
        reconstructed = ae_model(img)
        error = mse(img, reconstructed)
        reconstruction_errors.append(error)

    if method == 'mean_std':
        threshold = np.mean(reconstruction_errors) + k * np.std(reconstruction_errors)
    elif method == 'f1_max':
        # Grid search로 F1 최대화하는 threshold 찾기
        threshold = find_optimal_threshold_by_f1(reconstruction_errors, labels)

    return threshold
```

---

## 📝 Logging 전략

### 1. 학습 로그 (CSV)

**형식**: `experiments/logs/train/cnn_ct_train.csv`

```csv
epoch,train_loss,val_loss,val_f1,val_accuracy,val_recall
1,0.523,0.412,0.78,0.82,0.75
2,0.401,0.389,0.81,0.84,0.79
3,0.356,0.375,0.83,0.86,0.82
```

**활용**:
- 학습 과정 추적
- 그래프 생성 (Loss curve, Metric curve)
- Early stopping 판단 근거

### 2. 추론 로그 (JSON)

**형식**: `experiments/logs/inference/batch_results_20250101.json`

```json
[
  {
    "image_id": "img_001.jpg",
    "timestamp": "2025-01-01T10:30:00",
    "cnn": {
      "pred": "defect",
      "confidence": 0.91
    },
    "autoencoder": {
      "score": 0.034,
      "threshold": 0.028,
      "is_anomaly": true
    },
    "vlm": {
      "judgement": "defect",
      "reason": "표면에 크랙 발견"
    },
    "vlg": {
      "bboxes": [{"x": 10, "y": 20, "w": 50, "h": 60}],
      "scores": [0.87]
    }
  }
]
```

**활용**:
- 웹 추론 결과 저장
- 배치 처리 이력 관리
- 재현성 확보

### 3. TensorBoard (학습용)

**로깅 항목**:
- Scalars: Loss, F1, Accuracy, Recall
- Images: AutoEncoder 재구성 결과 (원본 vs 재구성)
- Histograms: 재구성 오류 분포 (Threshold 결정용)
- Confusion Matrix: CNN 분류 결과

**TensorBoard Logger 초기화 (run 이름 자동 생성)**:
```python
# training/visualization/tensorboard_logger.py
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

class TensorBoardLogger:
    def __init__(self, config):
        """
        TensorBoard Logger 초기화
        - run 이름: <model>_<key_params>
        """
        # Run 이름 생성
        run_name = self._generate_run_name(config)

        # TensorBoard Writer 생성
        log_dir = Path(config['tensorboard']['log_dir']) / config['model']['name'] / run_name
        self.writer = SummaryWriter(log_dir=str(log_dir))

    def _generate_run_name(self, config):
        """
        실험 조건 기반 run 이름 생성
        예: resnet18_lr1e-4_bs32
        """
        model_name = config['model']['name']
        lr = config['training']['lr']
        bs = config['training']['batch_size']

        run_name = f"{model_name}_lr{lr}_bs{bs}"

        # 추가 파라미터 (선택적)
        if 'weight_decay' in config['training'] and config['training']['weight_decay'] > 0:
            wd = config['training']['weight_decay']
            run_name += f"_wd{wd}"

        # AutoEncoder: k 값 추가
        if 'threshold' in config and 'k' in config['threshold']:
            k = config['threshold']['k']
            run_name += f"_k{k}"

        return run_name

    def log_scalars(self, epoch, train_loss, val_loss, val_f1):
        self.writer.add_scalar('train/loss', train_loss, epoch)
        self.writer.add_scalar('val/loss', val_loss, epoch)
        self.writer.add_scalar('val/f1', val_f1, epoch)

    def close(self):
        self.writer.close()
```

**실행 예시**:
```bash
# 학습 시작 → TensorBoard 자동 로깅
python training/scripts/train_cnn_ct.py
# → experiments/logs/tensorboard/resnet18/resnet18_lr0.0001_bs32/

# TensorBoard 실행
tensorboard --logdir experiments/logs/tensorboard
# 브라우저에서 http://localhost:6006 접속
# → "resnet18_lr0.0001_bs32" run 이름으로 표시
```

---

## 🎚️ Threshold 관리 전략

### Phase 1: 고정 Threshold (현재)

**방식**: `mean + k * std`

```python
# training/evaluation/threshold_finder.py
def compute_fixed_threshold(val_errors, k=2.5):
    """
    고정 Threshold 계산
    - 검증 데이터의 재구성 오류 분포 기반
    - mean + k * std
    """
    threshold = np.mean(val_errors) + k * np.std(val_errors)
    return threshold
```

**저장 형식**: `experiments/checkpoints/autoencoder/ae_rgb_threshold.json`

```json
{
  "threshold": 0.0285,
  "method": "mean_std",
  "k": 2.5,
  "computed_from": "validation_set",
  "num_samples": 520,
  "mean_error": 0.0198,
  "std_error": 0.0035
}
```

**특징**:
- 재현성과 비교 실험에 최적
- Config로 k 값 조절 가능
- 학습 후 1회만 계산

### Phase 2: 적응형 Threshold (확장)

**방식**: F1 최대화

```yaml
# training/configs/autoencoder.yaml (Phase 2)
threshold:
  mode: adaptive
  method: f1_max
```

```python
def find_adaptive_threshold(val_errors, val_labels):
    """
    F1 최대화하는 Threshold 찾기
    - Grid search
    """
    best_f1 = 0
    best_threshold = 0

    for threshold in np.linspace(min(val_errors), max(val_errors), 100):
        predictions = (val_errors > threshold).astype(int)
        f1 = f1_score(val_labels, predictions)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold
```

**적용 시점**: 실험 결과 충분히 쌓인 이후

---

## 🔧 Config 로딩 예시

```python
# training/config/config_loader.py
import yaml
from pathlib import Path

class ConfigLoader:
    """YAML Config 로더"""

    @staticmethod
    def load(config_name: str):
        """
        Config 파일 로드
        Args:
            config_name: 'cnn', 'autoencoder', 'evaluation', 'logging'
        """
        config_path = Path(__file__).parent.parent / 'configs' / f'{config_name}.yaml'

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        return config

# 사용 예시
# training/scripts/train_cnn_ct.py
from config.config_loader import ConfigLoader

config = ConfigLoader.load('cnn')

model = ResNet(
    name=config['model']['name'],
    pretrained=config['model']['pretrained'],
    num_classes=config['model']['num_classes']
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=config['training']['lr'],
    weight_decay=config['training']['weight_decay']
)
```

---

## 📋 핵심 요약

### Config 관리
- ✅ 모든 실험 설정은 YAML 파일로 관리
- ✅ 하드코딩 금지 → 재현성 확보
- ✅ Baseline → 확장 실험 동일 구조

### Evaluation 기준
- ✅ **F1 Score = 주 평가 지표**
- ✅ Accuracy = 참고 지표
- ✅ Recall = 불량 미탐 최소화 확인

### Logging 전략
- ✅ 학습 로그: CSV (간단, 가독성)
- ✅ 추론 로그: JSON (구조화, API 호환)
- ✅ TensorBoard: 학습 과정 시각화

### Threshold 관리
- ✅ Phase 1: 고정 (mean + k * std)
- ✅ Phase 2: 적응형 (F1 최대화)
- ✅ **모델과 함께 저장 필수**

---

## 🎨 Data Transform 설계 (CT vs RGB 공통화)

### 문제점
CT 데이터와 RGB 데이터는 전처리 방식이 완전히 다름:

| 항목 | CT 데이터 | RGB 데이터 |
|------|----------|-----------|
| **채널** | 그레이스케일 (1채널) | 컬러 (3채널) |
| **정규화** | CT 특화 정규화 (HU 값 등) | ImageNet 정규화 |
| **전처리** | 윈도잉, 클리핑 | 일반적인 이미지 전처리 |
| **Data Augmentation** | 회전, Flip만 | 컬러 jitter, 밝기 조정 등 |

→ **해결책**: Factory 패턴 + Config 기반으로 modality별 Transform을 자동 선택

---

### 설계 원칙

1. **공통 인터페이스**: `get_transforms(modality, mode)` 함수로 통일
2. **Config 기반**: YAML 파일에서 augmentation 설정 로드
3. **모듈화**: Train/Val/Test 별로 다른 transform 적용
4. **재사용성**: Dataset에서 modality만 전달하면 자동 선택

---

### 구현 코드

#### 1. Transform Factory (`training/data/transforms.py`)

```python
# training/data/transforms.py
import torch
from torchvision import transforms
from typing import Literal

class CTTransforms:
    """CT 데이터 전용 Transform"""

    @staticmethod
    def get_train_transforms(config):
        """CT 학습용 Transform"""
        return transforms.Compose([
            transforms.Resize((config['data']['image_size'], config['data']['image_size'])),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            # CT는 컬러 augmentation 제외
            transforms.ToTensor(),
            # CT 전용 정규화 (평균 0.5, 표준편차 0.5)
            transforms.Normalize(mean=[0.5], std=[0.5])  # 단일 채널
        ])

    @staticmethod
    def get_val_transforms(config):
        """CT 검증/테스트용 Transform (augmentation 제외)"""
        return transforms.Compose([
            transforms.Resize((config['data']['image_size'], config['data']['image_size'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    @staticmethod
    def get_test_transforms(config):
        """CT 테스트용 (Val과 동일)"""
        return CTTransforms.get_val_transforms(config)


class RGBTransforms:
    """RGB 데이터 전용 Transform"""

    @staticmethod
    def get_train_transforms(config):
        """RGB 학습용 Transform"""
        return transforms.Compose([
            transforms.Resize((config['data']['image_size'], config['data']['image_size'])),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            # RGB 전용 augmentation
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            # ImageNet 정규화 (3채널)
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    @staticmethod
    def get_val_transforms(config):
        """RGB 검증/테스트용 Transform"""
        return transforms.Compose([
            transforms.Resize((config['data']['image_size'], config['data']['image_size'])),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    @staticmethod
    def get_test_transforms(config):
        """RGB 테스트용 (Val과 동일)"""
        return RGBTransforms.get_val_transforms(config)


# ⭐ Factory 함수 (핵심)
def get_transforms(
    modality: Literal['ct', 'rgb'],
    mode: Literal['train', 'val', 'test'],
    config: dict
):
    """
    Modality와 Mode에 따라 적절한 Transform 반환

    Args:
        modality: 'ct' 또는 'rgb'
        mode: 'train', 'val', 'test'
        config: YAML config 딕셔너리

    Returns:
        transforms.Compose 객체

    Example:
        >>> config = ConfigLoader.load('cnn')
        >>> train_transform = get_transforms('ct', 'train', config)
        >>> val_transform = get_transforms('ct', 'val', config)
    """
    if modality == 'ct':
        transform_class = CTTransforms
    elif modality == 'rgb':
        transform_class = RGBTransforms
    else:
        raise ValueError(f"Unknown modality: {modality}. Must be 'ct' or 'rgb'.")

    if mode == 'train':
        return transform_class.get_train_transforms(config)
    elif mode == 'val':
        return transform_class.get_val_transforms(config)
    elif mode == 'test':
        return transform_class.get_test_transforms(config)
    else:
        raise ValueError(f"Unknown mode: {mode}. Must be 'train', 'val', or 'test'.")
```

---

#### 2. Dataset에서 사용 (`training/data/dataset.py`)

```python
# training/data/dataset.py
from torch.utils.data import Dataset
from PIL import Image
from .transforms import get_transforms

class BatteryDataset(Dataset):
    """배터리 데이터셋 (CT/RGB 공통)"""

    def __init__(self, csv_path, modality, mode, config):
        """
        Args:
            csv_path: 데이터 CSV 경로
            modality: 'ct' or 'rgb'
            mode: 'train', 'val', 'test'
            config: YAML config
        """
        self.data = self._load_csv(csv_path)
        self.modality = modality
        self.mode = mode

        # ⭐ Transform 자동 선택
        self.transform = get_transforms(modality, mode, config)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]['image_path']
        label = self.data.iloc[idx]['label']

        # 이미지 로드
        image = Image.open(img_path)

        # CT는 L 모드 (그레이스케일), RGB는 RGB 모드
        if self.modality == 'ct':
            image = image.convert('L')  # 그레이스케일
        else:
            image = image.convert('RGB')

        # Transform 적용 (자동으로 modality별 처리)
        if self.transform:
            image = self.transform(image)

        return image, label
```

---

#### 3. DataLoader 팩토리 (`training/data/dataloader.py`)

```python
# training/data/dataloader.py
from torch.utils.data import DataLoader
from .dataset import BatteryDataset

def get_dataloader(csv_path, modality, mode, config, shuffle=True):
    """
    DataLoader 생성 (modality별 자동 처리)

    Args:
        csv_path: CSV 경로
        modality: 'ct' or 'rgb'
        mode: 'train', 'val', 'test'
        config: YAML config
        shuffle: 셔플 여부

    Returns:
        DataLoader 객체
    """
    dataset = BatteryDataset(
        csv_path=csv_path,
        modality=modality,
        mode=mode,
        config=config
    )

    return DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=shuffle,
        num_workers=config['training'].get('num_workers', 4),
        pin_memory=True
    )
```

---

#### 4. 학습 스크립트에서 사용

```python
# models/ct_cnn/train.py
from training.config.config_loader import ConfigLoader
from training.data.dataloader import get_dataloader

# Config 로드
config = ConfigLoader.load('cnn')

# ⭐ DataLoader 생성 (modality와 mode만 전달)
train_loader = get_dataloader(
    csv_path='training/data/splits/ct_cnn/train.txt',
    modality='ct',  # CT 데이터
    mode='train',   # 학습 모드 (augmentation 적용)
    config=config,
    shuffle=True
)

val_loader = get_dataloader(
    csv_path='training/data/splits/ct_cnn/val.txt',
    modality='ct',
    mode='val',     # 검증 모드 (augmentation 제외)
    config=config,
    shuffle=False
)

test_loader = get_dataloader(
    csv_path='training/data/splits/ct_cnn/test.txt',
    modality='ct',
    mode='test',    # 테스트 모드
    config=config,
    shuffle=False
)
```

---

### Config 파일 설정

#### CNN Config (`training/configs/cnn.yaml`)
```yaml
data:
  image_size: 512
  modality: ct  # ⭐ modality 명시

training:
  batch_size: 32
  num_workers: 4
```

#### AutoEncoder Config (`training/configs/autoencoder.yaml`)
```yaml
data:
  image_size: 256
  modality: rgb  # ⭐ RGB AutoEncoder

training:
  batch_size: 32
  num_workers: 4
```

---

### 핵심 장점

#### 1. **단일 인터페이스**
```python
# CT든 RGB든 동일한 방식
transform = get_transforms(modality='ct', mode='train', config)
```

#### 2. **자동 처리**
- Dataset이 modality만 받으면 자동으로 적절한 transform 적용
- 개발자가 일일이 transform 선택할 필요 없음

#### 3. **확장 가능**
```python
# 새 modality 추가 시
class XRayTransforms:
    @staticmethod
    def get_train_transforms(config):
        # X-Ray 전용 전처리
        pass

# Factory 함수에 추가만 하면 됨
if modality == 'xray':
    transform_class = XRayTransforms
```

#### 4. **Config 기반**
- `image_size`, `batch_size` 등 config에서 관리
- 코드 수정 없이 실험 가능

#### 5. **재현성**
- Train/Val/Test 분리 명확
- Val/Test는 augmentation 제외 (동일 결과)

---

### 주의사항

#### 1. **채널 수 불일치 방지**
```python
# CT: 1채널 → 3채널 복제 (Pretrained 모델 사용 시)
if self.modality == 'ct' and self.use_pretrained:
    image = image.convert('RGB')  # L → RGB 변환
```

#### 2. **정규화 값 검증**
```python
# CT 데이터 정규화 후 범위 확인
assert image.min() >= -1.0 and image.max() <= 1.0
```

#### 3. **Augmentation 강도 조절**
```yaml
# training/configs/cnn.yaml
augmentation:
  rotation_degrees: 15
  flip_prob: 0.5
  color_jitter: false  # CT는 비활성화
```

---

## 🎯 Grad-CAM 통합 계획 (모델 간 공정 비교)

### 배경 및 목적
현재 프로젝트는 **CNN/AutoEncoder vs VLM/VLG** 비교를 목표로 하지만, 출력 형식이 다름:
- **CNN/AE**: 분류만 (위치 정보 ❌)
- **VLM/VLG**: 분류 + 위치 정보 (텍스트 또는 BBox ✅)

→ **해결책**: CNN에 Grad-CAM을 추가하여 모든 모델이 **위치 정보**를 제공하도록 통일

### 구현 전략

#### 1. Grad-CAM 모듈 (`models/ct_cnn/gradcam.py`)
```python
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget

class GradCAMGenerator:
    """Grad-CAM 히트맵 생성기"""

    def __init__(self, model, target_layer):
        self.model = model
        self.cam = GradCAM(model=model, target_layers=[target_layer])

    def generate(self, image_tensor, target_class=1):
        """
        Grad-CAM 히트맵 생성

        Args:
            image_tensor: 입력 이미지 텐서 (1, C, H, W)
            target_class: 타겟 클래스 (0: normal, 1: defect)

        Returns:
            heatmap: numpy array (H, W) 범위 [0, 1]
        """
        targets = [BinaryClassifierOutputTarget(target_class)]
        heatmap = self.cam(input_tensor=image_tensor, targets=targets)
        return heatmap[0]  # Batch dimension 제거
```

#### 2. BBox 추출 (`models/ct_cnn/bbox_extractor.py`)
```python
import cv2
import numpy as np

def extract_bboxes_from_heatmap(heatmap, threshold=0.5, min_area=100):
    """
    Grad-CAM 히트맵에서 여러 개의 Bounding Box 추출

    Args:
        heatmap: Grad-CAM 히트맵 (H, W) 범위 [0, 1]
        threshold: 이진화 임계값 (0.5 = 히트맵 상위 50%)
        min_area: 최소 영역 크기 (픽셀)

    Returns:
        bboxes: [{"x": int, "y": int, "w": int, "h": int, "score": float}, ...]
    """
    # 1. 이진화
    binary_mask = (heatmap > threshold).astype(np.uint8)

    # 2. Connected Components 분석
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_mask, connectivity=8
    )

    bboxes = []
    for i in range(1, num_labels):  # 0은 배경
        x, y, w, h, area = stats[i]

        # 작은 영역 필터링
        if area < min_area:
            continue

        # 해당 영역의 평균 히트맵 값 = confidence score
        region_mask = (labels == i)
        score = float(heatmap[region_mask].mean())

        bboxes.append({
            "x": int(x),
            "y": int(y),
            "w": int(w),
            "h": int(h),
            "score": score
        })

    # Confidence 기준 내림차순 정렬
    bboxes.sort(key=lambda b: b["score"], reverse=True)
    return bboxes
```

#### 3. 시각화 (`models/ct_cnn/visualizer.py`)
```python
import cv2
import numpy as np
from pathlib import Path
from PIL import Image

def visualize_gradcam(original_image, heatmap, bboxes, image_name, save_dir):
    """
    Grad-CAM 시각화 (3가지 이미지 생성)

    Args:
        original_image: 원본 이미지 (H, W, C) numpy array
        heatmap: Grad-CAM 히트맵 (H, W) 범위 [0, 1]
        bboxes: BBox 리스트
        image_name: 이미지 파일명 (확장자 제외)
        save_dir: 저장 디렉토리

    Returns:
        {
            "heatmap": "path/to/heatmap.jpg",
            "overlay": "path/to/overlay.jpg",
            "heatmap_overlay": "path/to/heatmap_overlay.jpg"
        }
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1. 히트맵 컬러맵 적용 (JET)
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # 2. 히트맵 오버레이 (원본 60% + 히트맵 40%)
    heatmap_overlay = cv2.addWeighted(
        original_image, 0.6,
        heatmap_colored, 0.4,
        0
    )

    # 3. BBox 오버레이
    bbox_overlay = original_image.copy()
    for bbox in bboxes:
        x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
        score = bbox["score"]

        # 녹색 박스 그리기
        cv2.rectangle(
            bbox_overlay,
            (x, y), (x + w, y + h),
            color=(0, 255, 0),  # 녹색
            thickness=3
        )

        # Confidence 표시
        cv2.putText(
            bbox_overlay,
            f"{score:.2f}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    # 4. 저장
    paths = {}

    heatmap_path = save_dir / f"{image_name}_heatmap.jpg"
    cv2.imwrite(str(heatmap_path), heatmap_colored)
    paths["heatmap"] = str(heatmap_path)

    overlay_path = save_dir / f"{image_name}_overlay.jpg"
    cv2.imwrite(str(overlay_path), bbox_overlay)
    paths["overlay"] = str(overlay_path)

    heatmap_overlay_path = save_dir / f"{image_name}_heatmap_overlay.jpg"
    cv2.imwrite(str(heatmap_overlay_path), heatmap_overlay)
    paths["heatmap_overlay"] = str(heatmap_overlay_path)

    return paths
```

### 통합 결과 비교

**모든 모델이 동일한 출력 형식**:

| 모델 | 분류 | 불량 유형 | 위치 정보 | 시각화 |
|------|------|-----------|-----------|--------|
| **CT CNN** | ✅ defect (0.95) | ✅ 5클래스 (porosity, resin_overflow 등) | ✅ Grad-CAM BBox | ✅ Heatmap + BBox |
| **RGB AE** | ✅ anomaly (score) | ❌ (이상 탐지만, Binary) | ❌ (전역 이상 감지) | ✅ 재구성 오차 맵 |
| **VLM** | ✅ defect | ✅ "porosity 결함" | ✅ [x:115, y:335, w:55, h:85] | ✅ BBox + 텍스트 설명 |
| **VLG** | ✅ defect | ✅ Query별 검출 | ✅ [x:118, y:338, w:52, h:82] | ✅ BBox |

> **불량 유형 (Defect Types)**:
> - CT 5클래스: `cell_normal`, `cell_porosity`, `module_normal`, `module_porosity`, `module_resin_overflow`
> - RGB: Binary 이상탐지 (`normal` vs `defect`)

### 평가 지표 (공정 비교)

#### 1. 분류 성능
- Metric: F1, Precision, Recall, Accuracy
- 모든 모델 공통 평가

#### 2. 위치 정확도 (Localization)
- Metric: **IoU (Intersection over Union)**
- Ground Truth BBox와 비교
- 평가 대상: CNN (Grad-CAM), VLM (텍스트→BBox 변환), VLG

#### 3. 종합 평가
- **F1@IoU>0.5**: COCO 방식
- 분류가 맞고 + 위치도 맞아야 True Positive

### 의존성 추가

```txt
# requirements.txt에 추가
pytorch-grad-cam>=1.4.0
opencv-python>=4.8.0
```

---

## 📝 다음 작업

### Phase 1: 폴더 구조 및 Config (1일)
1. **폴더 구조 생성**: `backend/`, `frontend/`, `training/` + `experiments/` 하위 구조
2. **Config YAML 파일 작성**: cnn.yaml, autoencoder.yaml, evaluation.yaml, logging.yaml
3. **Config Loader 구현**: YAML 파일 로딩 유틸리티
4. **Backend 디렉토리 초기화**: FastAPI 기본 구조 + `__init__.py` 파일

### Phase 2: Training 기본 구조 (2-3일)
5. **Dataset/DataLoader 구현**: RGB/CT 데이터 로딩
6. **TensorBoard Logger 구현**: run 이름 자동 생성 + 스칼라/이미지 로깅
7. **Evaluation Metrics 구현**: F1, Accuracy, Precision, Recall, ROC-AUC
8. **Threshold Finder 구현**: mean + k * std 방식

### Phase 3: 모델 학습 (3-4일)
9. **CNN (CT) 학습**: ResNet18 + Early Stopping + F1 기반 저장
10. **AutoEncoder (RGB) 학습**: 불량 데이터 기반 + Threshold 계산
11. **AutoEncoder (CT) 학습**: 정상 데이터 기반 + Threshold 계산
12. **체크포인트 + Threshold 저장**: `.pt` + `_threshold.json`

### Phase 4: Backend API (2-3일)
13. **Pydantic Schema 정의**: Request/Response + BoundingBox
14. **CNN/AE Predictor 구현**: 체크포인트 로드 + 추론
15. **InferencePipeline 구현**: 모델 병렬 실행
16. **Result Saver 구현**: job_id 기반 결과 저장
17. **FastAPI 엔드포인트**: `/infer`, `/infer/batch`, `/upload`, `/models`

### Phase 5: Frontend UI (2-3일)
18. **Streamlit 기본 UI**: 레이아웃 + 사이드바
19. **이미지 업로더 컴포넌트**: 단일/배치 업로드
20. **API 클라이언트 구현**: Backend 호출 + job_id 생성
21. **ResultViewer 컴포넌트**: 모델별 탭 (CNN, AE, VLM, VLG)
22. **job_id 관리**: 이전 결과 불러오기 기능

### Phase 6: VLM/VLG (선택적, 3-4일)
23. ⏸️ VLM (Qwen3-VL) 로컬 추론
24. ⏸️ VLG (GroundingDINO) 연동
25. ⏸️ Frontend에 결과 연동

### Phase 7: 고도화 (2-3일)
26. ⏸️ Anomaly Score 히스토그램 (배치 결과 분석)
27. ⏸️ Confusion Matrix 시각화
28. ⏸️ DB 마이그레이션 (선택적)

---

## 📚 통합된 설계 문서

본 문서(`implementation_structure.md`)는 다음 설계 문서들을 통합하여 실제 구현 가능한 구조로 변환한 최종 설계서입니다:

### 통합 문서 목록
1. ✅ **vision_pipeline_design.md**
   - Web 기반 모델 비교 시각화
   - FastAPI + Streamlit 아키텍처
   - JSON Schema 통합

2. ✅ **config_and_evaluation_design.md**
   - YAML 기반 Config 관리
   - F1 중심 평가 지표
   - Threshold 관리 전략
   - CSV/JSON 로그 구조

3. ✅ **tensor_board_and_web_visualization_architecture.md**
   - TensorBoard vs Web UI 역할 분리
   - job_id 기반 결과 관리
   - 실험(run) 이름 규칙
   - 로그 기반 Web 시각화

### 핵심 설계 원칙 (통합)

1. **"비교 실험 단계에서는 단순하고 결정적인 파이프라인을 유지하고, 해석과 확장은 결과가 쌓인 이후에 수행한다."**

2. **"코드는 고정하고, 실험은 설정으로 바꾼다"**
   - 모든 실험 설정은 YAML 파일로 관리
   - 동일 config → 동일 결과 재현 가능

3. **"학습 과정은 TensorBoard로, 최종 결과는 로그 기반 Web UI로 본다."**
   - TensorBoard: 학습 모니터링 전용
   - Web UI: 추론 결과 시각화 전용
   - 두 시스템은 같은 로그를 공유하지 않음

4. **"실험은 설정으로, 판단은 지표로, 결과는 로그로 남긴다"**
   - F1 Score = 주 평가 지표
   - Threshold = 설정값 (코드 로직 아님)
   - 모든 결과는 job_id 기반으로 저장

---

**본 문서를 기반으로 바로 구현을 시작할 수 있습니다.**
