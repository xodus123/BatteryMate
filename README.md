# Battery Defect Inspection AI System

리튬이온 배터리의 내부(CT) 및 외부(RGB) 결함을 자동 검출하는 AI 기반 품질 검사 시스템

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 주요 기능

- **멀티모달 분석**: CT 이미지(내부 결함) + RGB 이미지(외부 결함) 동시 분석
- **3-Way 검사 시스템**: CNN + AutoEncoder + VLM/VLG 다중 모델 검증
- **15종 아키텍처 실험**: ResNet18, ConvNeXt, EfficientNet, CBAM, Late Fusion, DRN+ASPP 등
- **설명 가능한 AI**: Grad-CAM, Error Map, Bounding Box 시각화
- **실시간 웹 대시보드**: Streamlit 기반 즉시 결과 확인

## 시스템 아키텍처

```
[배터리 이미지 입력: CT + RGB]
           ↓
┌──────────────────────────────────────────────────┐
│  System 1: CNN+AE+Grad-CAM 통합 검사              │
│  ┌────────────────┬──────────────────┐           │
│  │  CT CNN        │  RGB AutoEncoder │           │
│  │  (ResNet18)    │  (CAE)           │           │
│  │  5클래스 분류   │  이상 탐지        │           │
│  └────────────────┴──────────────────┘           │
│           ↓                ↓                     │
│  ┌──────────────────────────────────┐            │
│  │ 논리적 조합 (AND/OR)              │            │
│  │ → 정상/내부불량/외부불량/복합불량   │            │
│  └──────────────────────────────────┘            │
│  + Grad-CAM 히트맵                                │
└──────────────────────────────────────────────────┘
                    VS (비교)
┌──────────────────────────────────────────────────┐
│  System 2: VLM (Gemini / Qwen3-VL)               │
│  → 자연어 기반 결함 해석 + 위치 설명 + 소견서 생성  │
└──────────────────────────────────────────────────┘
                    VS
┌──────────────────────────────────────────────────┐
│  System 3: VLG (GroundingDINO)                    │
│  → 불량 영역 BBox 검출                            │
└──────────────────────────────────────────────────┘
           ↓
┌──────────────────────────────────────────────────┐
│  Web UI: 3개 시스템 결과 비교 시각화               │
└──────────────────────────────────────────────────┘
```

## 3-Way 모델 비교 분석

본 시스템은 3개의 독립적인 분석 모델을 사용하여 다각도로 결함을 검출합니다. 각 모델의 특성과 한계를 분석하여 상호 보완적인 다중 모델 시스템을 구축했습니다.

### 핵심 설계 철학: "AI가 이미지를 어떻게 이해하는가?"

3개 모델은 각각 **다른 인지 패러다임**으로 CT 이미지를 분석합니다:

| 분석 모델 | 인지 패러다임 | 핵심 역할 (Core Value) | CT 데이터 최적화 전략 |
|-----------|-------------|----------------------|---------------------|
| **CNN + AE (논리 결합)** | 통계적 인지 | 수치적 이상 탐지 | 픽셀 분포의 편차를 계산하여 미세 기공/공극의 존재 여부를 확률로 산출 |
| **VLM (Gemini)** | 추론적 인지 | 문맥적 원인 분석 | 이미지 전체의 구조적 관계를 파악하여 결함의 발생 원인(제조 공정 등)을 추론 |
| **VLG (DINO)** | 언어적 인지 | 지시적 위치 특정 | 자연어 프롬프트(예: "void")를 시각적 좌표와 매칭하여 논리적 근거(BBox) 제시 |

```
┌─────────────────────────────────────────────────────────────────┐
│               3가지 인지 패러다임 기반 다각적 검증               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   CNN + AE (통계적)     VLM (추론적)       VLG (언어적)          │
│   ┌───────────────┐   ┌───────────────┐   ┌───────────────┐    │
│   │ 픽셀 분포에서  │   │ "제조 공정 중 │   │   ┌─────┐      │    │
│   │ 정상 패턴과의  │   │  가스 배출 불 │   │   │ void│      │    │
│   │ 편차 계산     │   │  량으로 기공  │   │   └─────┘      │    │
│   │               │   │  발생 추정"   │   │   ↑ [0.45,0.52]│    │
│   │  Score: 2.3   │   │               │   │   BBox 좌표    │    │
│   │  → 불량 95%   │   │  원인 분석    │   │   제공         │    │
│   └───────────────┘   └───────────────┘   └───────────────┘    │
│                                                                 │
│   "수치로 판정"        "원인을 추론"        "위치를 특정"         │
└─────────────────────────────────────────────────────────────────┘
```

### 모델별 역할 및 특성

| 구분 | 통합 검사 (CNN+AE) | VLM (Gemini/Qwen3-VL) | VLG (GroundingDINO) |
|------|-------------------|----------------------|---------------------|
| **인지 패러다임** | 통계적 인지 | 추론적 인지 | 언어적 인지 |
| **결합 방식** | 논리 기반 (AND/OR) | 독립 추론 | 독립 탐지 |
| **역할** | 수치적 이상 탐지 | 문맥적 원인 분석 | 지시적 위치 특정 |
| **출력** | 클래스 확률 | 자연어 소견서 | 바운딩 박스 |
| **강점** | 높은 정확도, 확률 산출 | 원인 추론, 설명 가능성 | 정확한 위치 좌표 |
| **약점** | 원인 설명 불가 | 위치 부정확 | 분류 불가 |

> **설계 포인트**: 단일 AI 모델의 한계를 인식하고, 통계적·추론적·언어적 인지 패러다임을 결합하여 **다각적 검증 시스템**을 구축

## 모델 성능

### CT CNN (내부 결함 분류, 5클래스)

| 모델 | Test F1 | Accuracy | ROC-AUC | 비고 |
|------|---------|----------|---------|------|
| **Late Fusion v2** | **0.803** | **80.3%** | **0.944** | ★ 최고 성능 (ResNet18 + 메타데이터 late concat) |
| DRN+ASPP | 0.794 | 78.2% | 0.957 | DeepLabV3+ 스타일 |
| Metadata v3 | 0.791 | 78.0% | 0.965 | Early Fusion |
| ResNet18 (no_x) | 0.788 | 78.7% | 0.948 | 이미지만 |
| CBAM (768, x축 포함) | 0.862 | 86.4% | 0.972 | x축 포함 학습 |

> 15종 아키텍처 체계적 실험 수행. 35,529 테스트 샘플, 배터리 ID 기준 split (데이터 누수 방지)

### RGB AutoEncoder (외부 결함 탐지)

| 지표 | Test |
|------|------|
| ROC-AUC | 0.9095 |

> 배터리 ID 분리 후 신버전 split 기준. Threshold 기반 Accuracy/F1은 재평가 필요.

### VLM (자연어 분석)

| 모델 | 용도 | 비고 |
|------|------|------|
| **Gemini 2.0 Flash** | 결함 해석, 소견서 생성 | ★ 권장 (API) |
| Qwen3-VL (2B/8B) | 오프라인 결함 해석 | 로컬 GPU |

> zero-shot 정량 분류에는 부적합. 자연어 기반 결함 해석/소견서 생성 용도로 활용.

### VLG (결함 위치 탐지)

| 모델 | 성능 | 비고 |
|------|------|------|
| **GroundingDINO** | 미세 결함 탐지 우수 | ★ 채택 |

## CT CNN 최고 성능 모델: ResNet18 + Late Fusion

### 아키텍처
```
이미지 → ResNet18 → 512차원 ─┐
                              ├─ concat [514차원] → FC → 5클래스
메타데이터 (x, y 좌표) → 2차원 ─┘
```

### 학습 설정
```yaml
model: ResNet18 + Late Fusion (메타데이터 raw concat)
image_size: 512x512
loss: Focal Loss (gamma=2.0, label_smoothing=0.1)
class_weights: [1.5, 1.2, 0.8, 1.0, 8.0]
optimizer: AdamW (lr=0.0001, weight_decay=0.03)
scheduler: CosineAnnealingWarmRestarts (T_0=10, T_mult=2)
early_stopping: patience=7, monitor=val_f1_macro
```

### 데이터 증강
```yaml
- RandomHorizontalFlip, RandomVerticalFlip
- RandomRotation (90°)
- RandomAffine (translate=0.1, scale=0.9~1.1)
```

## 기술 스택

| 분류 | 기술 |
|------|------|
| Deep Learning | PyTorch, TorchVision, Transformers |
| 모델 | ResNet18, ConvNeXt, EfficientNet, CBAM, DRN+ASPP, ConvAutoEncoder |
| VLM/VLG | Qwen3-VL, Gemini 2.0 Flash, GroundingDINO |
| 시각화 | Grad-CAM, TensorBoard |
| 웹 프레임워크 | Streamlit |

## 프로젝트 구조

```
battery-inspection/
├── models/
│   ├── ct_cnn/                    # CT CNN (15종 아키텍처)
│   │   ├── model.py               # 모델 디스패치 (create_model)
│   │   ├── model_late_fusion.py   # ★ Late Fusion (최고 F1=0.803)
│   │   ├── model_cbam.py          # CBAM 어텐션
│   │   ├── model_drn_aspp.py      # DRN+ASPP (DeepLabV3+ 스타일)
│   │   ├── model_deeplabv3.py     # DeepLabV3+ 전이학습
│   │   ├── model_timm.py          # ConvNeXt/EfficientNet (timm)
│   │   ├── model_metadata.py      # Early Fusion (메타데이터 MLP)
│   │   ├── model_hdcnn.py         # HD-CNN (계층적 분류)
│   │   ├── model_hierarchical.py  # Hierarchical 분류
│   │   ├── train.py / test.py     # 범용 학습/평가
│   │   ├── train_late_fusion.py   # Late Fusion 전용 학습
│   │   ├── train_metadata.py      # Metadata 전용 학습
│   │   ├── checkpoints/           # 모델 체크포인트
│   │   └── results/               # 테스트 결과 JSON + Confusion Matrix
│   │
│   ├── rgb_ae/                    # RGB AutoEncoder (이상 탐지)
│   │   ├── model.py               # ConvAutoEncoder 정의
│   │   ├── train.py / test.py     # 학습/평가
│   │   ├── checkpoints/           # 체크포인트 + threshold.json
│   │   └── results/               # 테스트 결과
│   │
│   ├── ct_ae/                     # CT AutoEncoder (실험)
│   │   └── train.py / test.py
│   │
│   ├── ct_yolo/                   # CT YOLO (실험)
│   │   └── train.py / test.py
│   │
│   ├── inspector/                 # 통합 검사 모듈
│   │   ├── inspector.py           # CNN+AE 논리 결합
│   │   ├── ct_ensemble_inspector.py  # CT 앙상블 검사기
│   │   ├── predictor.py           # CT/RGB 개별 예측기
│   │   └── gradcam.py             # Grad-CAM 시각화
│   │
│   ├── vlm/                       # VLM (Qwen3-VL, Gemini)
│   │   ├── inference.py           # Qwen3-VL 추론 (CT/RGB)
│   │   ├── inference_gemini.py    # Gemini API 추론
│   │   ├── prompts.py             # 분석 프롬프트 (CT/RGB)
│   │   ├── test_vlm.py            # 단일 이미지 테스트
│   │   └── test_vlm_eval.py       # 대규모 평가 스크립트
│   │
│   └── vlg/                       # VLG (GroundingDINO)
│       ├── inference.py           # GroundingDINO 추론
│       ├── inference_yoloworld.py # YOLO-World (비교용)
│       └── prompts.py             # 탐지 프롬프트
│
├── training/
│   ├── configs/                   # 학습 설정 YAML (20+ 실험 config)
│   │   ├── cnn_ct_late_fusion.yaml   # ★ Late Fusion 설정
│   │   ├── cnn_ct_drn_aspp.yaml      # DRN+ASPP 설정
│   │   ├── cnn_ct_cbam.yaml          # CBAM 설정
│   │   ├── vlm_eval.yaml / vlm_eval_rgb.yaml  # VLM 평가
│   │   └── config_loader.py          # YAML 설정 로더
│   ├── data/                      # 데이터 파이프라인
│   │   ├── dataset.py             # BatteryDataset
│   │   ├── dataset_metadata.py    # 메타데이터 Dataset
│   │   ├── dataloader.py          # DataLoader + WeightedSampler
│   │   ├── transforms.py          # 데이터 증강
│   │   └── splits/                # train/val/test split 파일
│   ├── evaluation/                # 평가 메트릭
│   └── visualization/             # TensorBoard Logger
│
├── scripts/                       # 데이터 전처리/분할 스크립트
│   ├── preprocess.py              # CT 이미지 전처리 (4000→512)
│   ├── fix_ct_split_by_battery.py # 배터리 ID 기준 split
│   ├── fix_rgb_split_by_battery.py
│   └── ...
│
├── webapp/                        # Streamlit 웹 대시보드
│   ├── app.py                     # 메인 앱
│   ├── pages/
│   │   ├── home.py                # 업로드 페이지
│   │   ├── processing.py          # 추론 처리
│   │   ├── summary.py             # 결과 요약
│   │   └── detail.py              # 상세 분석
│   └── utils/                     # 세션, 스타일, 결함 정보
│
├── docs/                          # 문서
│   ├── MODEL_PERFORMANCE.md       # 전체 모델 성능 비교표
│   └── ...
│
├── config.py                      # 중앙 설정 관리 (.env 로드)
├── PORTFOLIO.md                   # 상세 포트폴리오
└── requirements.txt
```

## Demo

![Demo](assets/demo.gif)

## 라이선스

MIT License

## 참고 문서

- [PORTFOLIO.md](PORTFOLIO.md) - 상세 포트폴리오 (모델 아키텍처, 실험 결과, 분석)
- [docs/MODEL_PERFORMANCE.md](docs/MODEL_PERFORMANCE.md) - 전체 모델 성능 비교표

---

*Developed with PyTorch, Streamlit, and Google Gemini API*

*Last Updated: 2026-02-20*
