# Battery Defect Inspection AI System

리튬이온 배터리의 내부(CT) 및 외부(RGB) 결함을 자동 검출하는 AI 기반 품질 검사 시스템

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 주요 기능

- **멀티모달 분석**: CT 이미지(내부 결함) + RGB 이미지(외부 결함) 동시 분석
- **3-Way 앙상블**: CNN + AutoEncoder + VLM/VLG 다중 모델 검증
- **설명 가능한 AI**: Grad-CAM, Error Map, Bounding Box 시각화
- **실시간 웹 대시보드**: Streamlit 기반 즉시 결과 확인

## 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                    입력 이미지 (CT + RGB)                    │
└─────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│    Ensemble     │ │      VLM        │ │      VLG        │
│    (CNN+AE)     │ │  (Gemini API)   │ │ (GroundingDINO) │
│                 │ │                 │ │                 │
│ • 분류 정확도   │ │ • 자연어 설명   │ │ • 위치 탐지     │
│ • 신뢰도 점수   │ │ • AI 소견서    │ │ • 바운딩 박스   │
└────────┬────────┘ └────────┬────────┘ └────────┬────────┘
         │                   │                   │
         └───────────────────┼───────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────┐
│              최종 판정: 정상 / 내부불량 / 외부불량 / 복합불량   │
└─────────────────────────────────────────────────────────────┘
```

## 모델 성능

| 모델 | 역할 | 정확도 | 비고 |
|------|------|--------|------|
| **CT CNN** (ResNet18) | 내부 결함 분류 | **99.2%** | 5클래스 분류 |
| **RGB AutoEncoder** | 외부 결함 탐지 | **98.85%** | 이상 탐지 |
| **VLM** (Gemini 2.0) | AI 소견서 생성 | 95% | API 기반 |
| **VLG** (GroundingDINO) | 결함 위치 탐지 | - | Open-vocab |

## 기술 스택

| 분류 | 기술 |
|------|------|
| Deep Learning | PyTorch, TorchVision, Transformers |
| 모델 | ResNet18, ConvAutoEncoder, Qwen2-VL, GroundingDINO |
| 외부 API | Google Gemini 2.0 Flash |
| 시각화 | Grad-CAM, TensorBoard |
| 웹 프레임워크 | Streamlit |

## 프로젝트 구조

```
battery-inspection/
├── models/
│   ├── ct_cnn/           # CT CNN 모델 (ResNet18)
│   │   ├── model.py      # 모델 정의
│   │   ├── train.py      # 학습 스크립트
│   │   ├── test.py       # 평가 스크립트
│   │   └── checkpoints/  # 모델 체크포인트
│   ├── rgb_ae/           # RGB AutoEncoder 모델
│   │   ├── model.py      # ConvAutoEncoder 정의
│   │   ├── train.py      # 학습 스크립트
│   │   └── checkpoints/  # 모델 체크포인트
│   ├── ensemble/         # 앙상블 통합 모듈
│   │   ├── ensemble.py   # CNN+AE 통합
│   │   └── predictor.py  # 개별 예측기
│   ├── vlm/              # VLM (Qwen2-VL, Gemini)
│   │   ├── inference.py  # Qwen2-VL 추론
│   │   └── inference_gemini.py  # Gemini API
│   └── vlg/              # VLG (GroundingDINO, YOLO-World)
│       ├── inference.py  # GroundingDINO
│       └── inference_yoloworld.py  # YOLO-World
├── webapp/
│   ├── app.py            # Streamlit 메인
│   └── pages/            # 페이지 컴포넌트
│       ├── home.py       # 업로드 페이지
│       ├── processing.py # 처리 페이지
│       └── summary.py    # 결과 페이지
├── training/
│   └── configs/          # 학습 설정 YAML
├── scripts/              # 데이터 처리 스크립트
├── PORTFOLIO.md          # 상세 포트폴리오
└── requirements.txt
```

## 라이선스

MIT License

## 참고 문서

- [PORTFOLIO.md](PORTFOLIO.md) - 상세 포트폴리오 (모델 아키텍처, 실험 결과, 분석)

---

*Developed with PyTorch, Streamlit, and Google Gemini API*

*Last Updated: 2026-01-08*
