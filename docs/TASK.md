# TASK.md - 배터리 불량 검사 프로젝트 작업 현황

> **최종 업데이트**: 2026-02-14
> **현재 Phase**: 전체 학습/테스트 완료, 결과 분석 및 문서화
> **다음 작업**: 과적합 해결 (정규화 강화, Temperature Scaling) / 추가 데이터 확보

---

## 📌 빠른 상태 확인

### 🎯 현재 상태
- ✅ **프로젝트 구조 정리 완료** (2026-01-03)
- ✅ **CT 통합 CNN 학습 완료** (2026-01-06) - 18 epoch (Early Stop), Best acc=83.07%
- ✅ **CT CNN Test 평가 완료** (2026-01-06) - Test acc=77.45%, F1=0.7881
- ✅ **CBAM 실험 완료** (2026-01-07) - F1=0.8022 (기본 모델보다 -3.1% 하락, 과적합)
- ✅ **RGB AE v2 학습 완료** (2026-02-08) - ROC-AUC=0.9781, 모델 개선
- ✅ **VLM/VLG 구현 완료** (2026-01-05) - 4클래스 분류 적용 (2026-01-07)
- ✅ **VLM Qwen3-VL 업그레이드** (2026-02-06) - BBox 탐지 지원 추가
- ✅ **Streamlit UI 구현 완료** (2026-01-05)
- ✅ **통합 검사기 구현 완료** (2026-01-06) - CT CNN + RGB AE + Grad-CAM
- ✅ **웹페이지 CT+RGB 듀얼 업로드 지원** (2026-01-06)
- ✅ **RGB AE test.py TensorBoard 추가** (2026-01-07)
- ✅ **전처리 분리 및 1024 이미지 적용** (2026-01-18) - Albumentations 추가
- ✅ **이미지 전처리 완료** (2026-01-18) - 260,665개 → D드라이브 저장
- ✅ **Config 기능 전면 구현** (2026-01-20) - 누락된 모든 기능 코드 연결 완료
- ✅ **전처리 좌표 오류 수정** (2026-01-27) - 원본 4000x4000에서 직접 crop
- ✅ **Battery outline crop v2 완료** (2026-01-27) - 179,024개, 1024x1024
- ✅ **CT 앙상블 검사기 구현** (2026-01-27) - CNN+Metadata + AutoEncoder
- ✅ **CT AE 학습/테스트 스크립트 생성** (2026-01-27)
- ✅ **CT AE 학습 완료** (2026-01-28) - ROC-AUC=0.653, Cell/Module 분리 문제 발견
- ✅ **Cell/Module 별도 Threshold 적용** (2026-01-28) - Cell 0.12, Module 0.28
- ✅ **CNN+Metadata 학습 문제 발견** (2026-01-28) - 이미지 스타일 차이 학습 (99.99% F1 → 과적합 아닌 데이터 문제)
- ✅ **축(axis) 상관관계 문제 발견** (2026-01-29) - x축=99.97% 정상, y/z축=결함 혼재 → 축 학습 문제
- ✅ **CNN+Metadata 모델 수정** (2026-01-29) - axis 메타데이터 추가 (METADATA_DIM: 1→2)
- ✅ **전처리 스타일 통일 수정** (2026-01-29) - 정상도 가늘고 긴 영역 crop → 검은 패딩 비율 통일
- ✅ **전처리 파라미터 수정** (2026-01-29) - thin_width: 30~80→5~35, thin_height: 500~1200→400~2500
- ✅ **메타데이터 의존도 감소** (2026-01-29) - output_dim: 128→32, dropout: 0.5×2
- ❌ **스타일 통일 전처리 폐기** (2026-01-29) - Black 비율 99%로 유효 정보 부족
- ✅ **패치 전략 구현** (2026-01-29) - 512x512 고정 크기 패치, 유효 정보 100%
- ✅ **Split 파일 형식 개선** (2026-01-29) - 메타데이터 포함 (path, label, battery_type, axis)
- ✅ **패치 전처리 완료** (2026-01-30) - Train 1.33M, Val 295K, Test 355K
- ✅ **CNN+Metadata 패치 학습 완료** (2026-01-30) - Test F1=0.874, Acc=93.2%
- ✅ **앙상블 테스트 완료** (2026-02-01) - CNN+AE 앙상블 효과 없음 (75% 불일치)
- ✅ **클래스 밸런싱 구현** (2026-02-01) - Class 3: 83%→31%, Train 325K
- ✅ **Late Fusion 학습 완료** (2026-02-03) - **Test F1=0.826** (현재 최고 성능)
- ❌ **HD-CNN 실험 실패** (2026-02-04) - Test F1=0.690, cell/module 혼동 심각
- ✅ **EfficientNet/ConvNeXt Config 생성** (2026-02-05) - timm backbone 지원 추가
- ✅ **model_timm.py 생성** (2026-02-05) - 다양한 timm 모델 지원
- ✅ **x축 라벨 분석 완료** (2026-02-12) - x축은 결함 배터리에서도 defects: null (물리적 한계)
- ✅ **x축 제외 split 생성** (2026-02-12) - Train 105,224 / Val 20,751 (x축 ~25% 제거)
- ✅ **no_x Config 5개 생성** (2026-02-12) - CBAM, Unified, ConvNeXt, EfficientNet-B4, HD-CNN
- ✅ **전 모델 resize512 통일** (2026-02-12) - Late Fusion, HD-CNN, Hierarchical, Metadata Balanced
- ✅ **VLM 평가 스크립트 구현** (2026-02-13) - test_vlm_eval.py + vlm_eval.yaml
- ✅ **no_x 모델 4종 학습 완료** (2026-02-12) - ResNet18, ConvNeXt, CBAM, EfficientNet-B4 (모두 과적합)
- ✅ **HD-CNN v2, Metadata v3 학습 완료** (2026-02-13) - resize512, 현재 split
- ✅ **Late Fusion v2 학습 완료** (2026-02-14) - resize512, **현재 split 최고 F1=0.803**
- ✅ **전체 모델 테스트 완료** (2026-02-14) - 8개 모델 현재 split(35,529) 테스트
- ✅ **MODEL_PERFORMANCE.md 전면 업데이트** (2026-02-14) - 전체 학습/테스트 결과 + 데이터 정보
- ⚠️ **이전 split 결과 무효 확인** (2026-02-14) - fix_all_ct_splits.py로 split 재생성, 이전 고성능 결과(F1=0.976) 무효
- ⚠️ **예측 과신뢰 문제 발견** (2026-02-14) - 오답에도 84%+ 신뢰도, Temperature Scaling 필요

### 📋 작업 계획 (우선순위 순)

| 단계 | 작업 | 상태 | 비고 |
|------|------|------|------|
| 1 | x축 라벨 분석 | ✅ 완료 | x축 = 결함 배터리도 defects: null |
| 2 | x축 제외 split 생성 | ✅ 완료 | resize512_no_x (train/val만 제외) |
| 3 | no_x Config 생성 | ✅ 완료 | 5개 모델 config |
| 4 | 전 모델 resize512 통일 | ✅ 완료 | cropped/patch → resize512 |
| 5 | CBAM no_x 학습 | ✅ 완료 | Val F1=0.731, Test F1=0.540 (과적합) |
| 6 | ResNet18 no_x 학습 | ✅ 완료 | Val F1=0.712, Test F1=0.545 (과적합) |
| 7 | ConvNeXt no_x 학습 | ✅ 완료 | Val F1=0.771, Test F1=0.571 (과적합) |
| 8 | EfficientNet-B4 no_x 학습 | ✅ 완료 | Val F1=0.766, Test F1=0.679 (과적합) |
| 9 | HD-CNN v2 학습 | ✅ 완료 | Val F1=0.547, Test F1=0.337 ❌ |
| 10 | Metadata v3 재학습 | ✅ 완료 | Val F1=0.793, Test F1=0.791 ✅ |
| 11 | Late Fusion v2 재학습 | ✅ 완료 | Val F1=0.824, **Test F1=0.803** ★ |
| 12 | 아키텍처 비교 분석 | ✅ 완료 | MODEL_PERFORMANCE.md 전면 업데이트 |
| 13 | 과적합 해결 | ⏳ 대기 | 정규화 강화, Temperature Scaling |
| 14 | 추가 데이터 확보 | ⏳ 대기 | 92개 배터리 → 200개+ 필요 |

**결론**: 현재 split 기준 Late Fusion v2 (F1=0.803)가 최고. 메타데이터 포함 모델이 순수 이미지 모델보다 우수.

### 📊 CT CNN 학습 결과 비교 (현재 split 35,529 샘플 기준)
| 모델 | Test Acc | Test F1 | ROC-AUC | 비고 |
|------|----------|---------|---------|------|
| **Late Fusion v2** | **80.3%** | **0.803** | **0.944** | **🏆 현재 최고 (메타데이터+이미지)** |
| Metadata v3 | 78.0% | 0.791 | 0.965 | ✅ 메타데이터 효과 |
| CBAM 768 | 86.3% | 0.862 | 0.968 | ✅ x축 포함 학습 |
| EfficientNet-B4 no_x | 66.7% | 0.679 | 0.912 | no_x 학습, 전체 테스트 |
| ConvNeXt no_x | 64.6% | 0.571 | 0.891 | no_x 학습, 전체 테스트 |
| ResNet18 no_x | 60.1% | 0.545 | 0.864 | no_x 학습, 전체 테스트 |
| CBAM 768 no_x | 60.3% | 0.540 | 0.895 | no_x 학습, 전체 테스트 |
| HD-CNN v2 | 38.5% | 0.337 | - | ❌ 성능 극히 저조 |

> ⚠️ 이전 split 고성능 결과 (ConvNeXt F1=0.976, EfficientNet-B0 F1=0.987)는 split 재생성으로 무효

### 📊 RGB AE v2 학습 결과 (2026-02-08)
| 항목 | 값 | 비고 |
|------|-----|------|
| ROC-AUC | **0.9781** | 🏆 최고 성능 |
| Normal Score | 1.2525 ± 0.2237 | 낮은 재구성 오류 |
| Defect Score | 2.0957 ± 0.0979 | 높은 재구성 오류 |
| Threshold | 1.4990 | ROC 최적값 |
| 테스트 샘플 | 6,134개 | Normal: 841 / Defect: 5,293 |
| **모델 개선** | | |
| Bottleneck | 4×4 | 공간 정보 유지 (기존 1×1) |
| Loss | MSE+SSIM (7:3) | 구조적 유사도 반영 |
| 리사이즈 | 비율 유지+패딩 | 1920×1080 → 512×512 |
| Train 데이터 | 정상만 5,746개 | 배터리 ID 분리 완료 |

### ⚠️ 분석된 문제점
- **과적합**: 전 모델(7개) 과적합 - Train Loss 감소, Val Loss 증가 (Late Fusion만 경미)
- **이전 split 무효**: fix_all_ct_splits.py 실행 → 이전 고성능 결과(F1=0.976) 무효
- **예측 과신뢰**: 오답에도 84%+ 신뢰도 → 신뢰도 기반 필터링 무의미
- **클래스 혼동**: cell_normal ↔ cell_porosity 구분 어려움 (전 모델 공통)
- **no_x 학습/테스트 불일치**: no_x 학습 → 전체 테스트 시 Val F1 대비 -0.09~0.20 하락
- **배터리 수 부족**: 92개 학습 배터리로 일반화 한계 → 200개+ 필요
- **원인**: 이미지 축소(4000→512), 위치 라벨 없음 (Weakly Supervised), 데이터 부족

### 📋 학습/평가 실행 명령어

```bash
# CT CNN + CBAM 학습 (개선 버전)
python models/ct_cnn/train.py --config cnn_ct_cbam

# VLM 평가 (500샘플, Qwen3-VL 8B)
python models/vlm/test_vlm_eval.py --config vlm_eval

# VLM 평가 (Gemini)
python models/vlm/test_vlm_eval.py --config vlm_eval --model-type gemini

# VLM 소규모 테스트 (50샘플, 2B 모델)
python models/vlm/test_vlm_eval.py --config vlm_eval --model-size 2b --num-samples 50

# TensorBoard 모니터링
tensorboard --logdir models/ct_cnn/logs --port 6006
tensorboard --logdir models/vlm/logs --port 6007
```

---

## 📊 CT 통합 CNN 학습 설정

### 데이터 분할
| Split | 이미지 수 | 배치 수 |
|-------|-----------|---------|
| Train | 138,316 | 4,323 |
| Val | 26,662 | 834 |
| Test | 36,424 | 1,139 |

### 클래스 분포 (Train)
| 클래스 | 개수 | 비율 |
|--------|------|------|
| cell_normal | 39,343 | 28.4% |
| cell_porosity | 12,755 | 9.2% |
| module_normal | 39,572 | 28.6% |
| module_porosity | 45,165 | 32.7% |
| module_resin_overflow | 1,481 | 1.1% ⚠️ |

### 학습 파라미터 (2026-01-20 업데이트 - 이중 보정 완화)
```yaml
model:
  name: resnet18
  pretrained: true
  num_classes: 5
  dropout: 0.5              # 0.3 → 0.5 (과적합 방지)

data:
  image_size: 1024          # 512 → 1024 (전처리된 이미지)
  batch_size: 16
  num_workers: 4            # 16 → 4 (RAM OOM 방지)
  class_balancing:
    enabled: true
    method: weighted_sampler  # ✅ 역빈도 기반 자동 계산

training:
  optimizer: AdamW
  lr: 0.00005               # 0.0001 → 0.00005 (과적합 방지)
  weight_decay: 0.03        # 0.01 → 0.03 (가중치 제한 강화)
  epochs: 50
  amp: true  # Mixed Precision
  gradient_clip: 1.0

scheduler:
  name: CosineAnnealingWarmRestarts
  T_0: 10
  T_mult: 2
  eta_min: 1e-6

criteria:
  loss: FocalLoss           # ✅ Softmax 기반 multi-class
  focal_loss:
    enabled: true
    gamma: 1.5              # 3.0 → 1.5 (이중 보정 완화)
  label_smoothing: 0.07     # 0.15 → 0.07 (이중 보정 완화)
  class_weights: [1.0, 4.0, 1.0, 0.9, 18.0]  # resin 25.0 → 18.0

early_stopping:
  monitor: val_f1_macro     # ✅ 동적 모니터링 구현 완료
  mode: max
  patience: 4               # 5 → 4 (과적합 조기 방지)
  min_delta: 0.001

checkpoint:
  save_top_k: 3             # ✅ Top-K 저장 구현 완료

logging:
  tensorboard:
    enabled: true           # ✅ 플래그 연동 완료
    log_grad_cam: true      # ✅ Grad-CAM 로깅 구현 완료
```

### 데이터 증강 (Train) - 2026-01-19 강화
- RandomHorizontalFlip (p=0.5)
- RandomVerticalFlip (p=0.5)
- RandomRotation (30°)              # 15° → 30°
- ColorJitter (brightness=0.3, contrast=0.3)  # 0.2 → 0.3
- RandomAffine (translate=0.1, scale=0.9~1.1)  # 추가
- GaussianBlur (kernel=3, p=0.3)    # 추가

### 출력 경로
- 체크포인트: `models/ct_cnn/checkpoints/`
- TensorBoard: `models/ct_cnn/logs/`

---

## 🗺️ 전체 로드맵

### Phase 1: 프로젝트 구조 설계 ✅
- ✅ 폴더 구조 정리
- ✅ Config 파일 작성 (`training/configs/cnn_ct_unified.yaml`)
- ✅ 데이터 Split 생성 (배터리 ID 기반, Data Leakage 방지)
- ✅ Dataset/DataLoader 구현 (5클래스 다중분류)
- ✅ TensorBoard Logger 구현 (Confusion Matrix 포함)

### Phase 2: CT CNN 학습 🔄 (현재)
- ✅ ResNet18 모델 정의 (5클래스 출력)
- ✅ Trainer 구현 (Focal Loss, Label Smoothing)
- ✅ 클래스 불균형 처리 (class_weights)
- ⏳ **학습 실행**
- ⏳ 학습 완료 및 평가

### Phase 3: RGB AutoEncoder 학습 ✅ (완료)
- ✅ AutoEncoder 모델 구현 (`models/rgb_ae/model.py`)
- ✅ Trainer 구현 (`models/rgb_ae/train.py`)
- ✅ Tester 구현 (`models/rgb_ae/test.py`)
- ✅ 데이터 복사 스크립트 (`scripts/copy_rgb_images.py`)
- ✅ RGB 데이터 복사 완료 (~59,263개 이미지)
- ✅ 학습 및 평가 완료 (ROC-AUC: 0.9644, Acc: 97.86%)

### Phase 4: VLM/VLG 구현 ✅
- ✅ Qwen2-VL 연동 (`models/vlm/`) - Zero-shot 결함 분석
- ✅ **Qwen3-VL 업그레이드** (`models/vlm/`) - BBox 탐지 지원 추가
- ✅ GroundingDINO 연동 (`models/vlg/`) - BBox 검출
- ✅ 추론 파이프라인 구현 (inference.py)
- ✅ 테스트 코드 작성 (test_vlm.py, test_vlg.py)

### Phase 5: FastAPI Backend ⏳
- ⏳ 3개 모델 통합 API (통합 검사 + VLM + VLG)
- ⏳ 추론 엔드포인트 구현
- ⏳ 결과 비교 API

### Phase 6: Streamlit UI ✅
- ✅ 이미지 업로드 인터페이스 (Home)
- ✅ 3개 시스템 결과 비교 화면 (Summary)
- ✅ TensorBoard 스타일 상세 대시보드 (Detail)
  - 통합 검사: Grad-CAM, 클래스 확률, AE 이상점수 분포
  - VLM: AI 소견서, 텍스트 Grounding
  - VLG: BBox 시각화, 신뢰도 분포, 임계값 조절

---

## 🔧 CT CNN 개선 방향 (학습 완료 후)

현재 학습에서 정상↔기공 혼동 문제가 발생. 아래 방법으로 개선 예정:

### 우선순위 1: 이미지 크기 증가
```yaml
# 현재
image_size: 512
batch_size: 32

# 개선안 (기공 디테일 보존)
image_size: 768
batch_size: 16~20
```
- 4000→512 (7.8배 축소) → 4000→768 (5.2배 축소)
- 작은 기공 특징이 더 잘 보존됨

### 우선순위 2: ResNet + CBAM (Attention)
```python
# Spatial Attention으로 "어느 위치가 중요한지" 학습
model.layer3 = nn.Sequential(model.layer3, CBAM(256))
model.layer4 = nn.Sequential(model.layer4, CBAM(512))
```
- 위치 라벨 없이도 중요 영역에 집중
- 기존 pretrained weights 유지 가능

### 우선순위 3: Focal Loss gamma 증가
```yaml
# 현재
gamma: 2.0

# 개선안 (어려운 샘플에 더 집중)
gamma: 3.0 또는 4.0
```

### 우선순위 4: 데이터 증강 강화
```python
# 랜덤 크롭 방식 (축소 비율 감소)
transforms.RandomCrop(1024)  # 4000에서 1024 크롭
transforms.Resize(512)        # 2배만 축소
```

### 참고: 데이터 한계
- ❌ 위치 라벨(BBox/Mask) 없음 - 이미지 레벨 라벨만 존재
- ❌ Weakly Supervised 상황 - 모델이 스스로 결함 위치 학습 필요

---

## 🔑 프로젝트 구조

```
battery-inspection/
├── CLAUDE.md, TASK.md, README.md
├── data -> /home/ubuntu/battery-data (심볼릭 링크)
│
├── docs/
│   ├── implementation_structure.md (전체 설계) ← 업데이트됨
│   ├── inspector_design.md (통합 검사기 설계)
│   ├── MODEL_ARCHITECTURE.md
│   └── TENSORBOARD_GUIDE.md
│
├── models/
│   ├── ct_cnn/
│   │   ├── train.py, test.py, model.py
│   │   ├── checkpoints/ (ct_unified_best_*.pt, ct_unified_last_*.pt)
│   │   └── logs/ (TensorBoard 로그)
│   ├── rgb_ae/
│   │   ├── model.py, train.py, test.py ← 신규 구현
│   │   └── checkpoints/
│   ├── vlm/ (inference.py, prompts.py) - Qwen3-VL (BBox 지원)
│   └── vlg/
│       ├── inference.py, prompts.py - GroundingDINO
│       └── weights/groundingdino_swint_ogc.pth (662MB) ← 다운로드됨
│
├── webapp/  # Streamlit UI (라이트 테마)
│   ├── app.py (메인 앱, 페이지 라우팅)
│   ├── pages/
│   │   ├── home.py (이미지 업로드)
│   │   ├── processing.py (3개 모델 분석)
│   │   └── summary.py (3-Way 비교 결과)
│   └── utils/
│       ├── session.py (세션 상태)
│       ├── styles.py (라이트 테마 CSS)
│       └── defect_info.py (5클래스 결함 정보 매핑) ← 신규
│
├── scripts/
│   ├── create_splits_final.py (데이터 분할)
│   ├── copy_rgb_images.py (D드라이브→Linux 복사) ← 신규
│   ├── check_data_leakage.py (검증)
│   └── check_label_consistency.py (검증)
│
└── training/
    ├── configs/ (cnn_ct_unified.yaml, autoencoder_rgb.yaml, inspector.yaml)
    ├── data/
    │   ├── dataset.py, dataloader.py
    │   └── splits/ct/ (train.txt, val.txt, test.txt)
    ├── evaluation/ (metrics.py)
    └── visualization/ (tensorboard_logger.py)
```

---

## 📝 최근 작업 기록

### 2026-02-14 - 전체 모델 테스트 완료 및 결과 분석

#### 학습 완료 (02-12 ~ 02-14, 7개 모델)
| 모델 | 학습일 | Val F1 | Best Epoch | 과적합 |
|------|--------|--------|------------|--------|
| ResNet18 no_x | 02-12 | 0.712 | 5/12 | 심각 |
| ConvNeXt no_x | 02-12 | 0.771 | 4/11 | 심각 |
| CBAM 768 no_x | 02-12 | 0.731 | 1/8 | 극심 |
| EfficientNet-B4 no_x | 02-12 | 0.766 | 5/12 | 심각 |
| HD-CNN v2 | 02-13 | 0.547 | 7/14 | 극심 (val_loss 폭등) |
| Metadata v3 | 02-13 | 0.793 | 5/13 | 심각 |
| **Late Fusion v2** | **02-14** | **0.824** | **5/12** | **경미 (가장 안정적)** |

#### 테스트 완료 (02-14, 현재 split 35,529 샘플)
- **Late Fusion v2**: F1=0.803, Acc=80.3% → **현재 split 최고 성능**
- **Metadata v3**: F1=0.791, Acc=78.0% → 메타데이터 효과 확인
- **no_x 모델 4종**: F1=0.540~0.679 → x축 미경험 모델의 전체 테스트 성능 저조
- **HD-CNN v2**: F1=0.337 → 성능 극히 저조, 폐기

#### 핵심 발견
1. **이전 split 결과 무효**: `fix_all_ct_splits.py`로 split 재생성 → 이전 고성능 결과(F1=0.976, 0.987) 현재 split과 불일치
2. **메타데이터가 핵심**: Late Fusion(0.803) vs 순수 이미지 최고 EfficientNet-B4(0.679) → **+0.124 차이**
3. **예측 과신뢰**: 오답에도 평균 84%+ 신뢰도 → Temperature Scaling 필요
4. **cell_porosity 난제**: 모든 모델에서 가장 낮은 F1 (0.27~0.65)
5. **전 모델 과적합**: 92개 배터리로 일반화 한계

#### 문서 업데이트
- ✅ `docs/MODEL_PERFORMANCE.md` 전면 업데이트 (822줄)
  - 현재 split / 이전 split 결과 분리
  - 학습 7개 + 테스트 8개 결과 추가
  - 과적합 분석, 과신뢰 문제, split 변경 문제 추가
  - 결론 및 권장 체크포인트 재정리

#### 수정 파일
- `docs/MODEL_PERFORMANCE.md` - 전면 업데이트
- `docs/TASK.md` - 현재 상태 업데이트

---

### 2026-02-12 - x축 제외 재학습 준비 및 전처리 통일

#### x축 라벨 데이터 분석
- **핵심 발견**: 결함 배터리(porosity)의 x축 라벨이 `defects: null`, `is_normal: true`
  - 배터리 109: x축 0% 결함 / y축 100% / z축 100%
  - 배터리 133: x축 0% / y축 30% / z축 0%
  - 배터리 137: x축 1.7% / y축 100% / z축 97.5%
- **원인**: CT 스캔 x축 단면에서는 기공(porosity)이 물리적으로 보이지 않음
  - x축: 넓은 직사각형 단면 → 결함 구분 불가
  - y/z축: 얇은 단면 → 기공이 검은 점/끊김으로 보임
- **영향**: 모델이 "x축 이미지 패턴 → 정상" shortcut 학습
  - x축 포함 시 98% 정확도, x축 제외 시 59%로 폭락

#### x축 제외 split 파일 생성
- **경로**: `training/data/splits/ct/resize512_no_x/`
- **방법**: 기존 split에서 `_x_` 패턴 포함 라인 제거
  | Split | 원본 | x축 제외 후 | 제거 수 |
  |-------|------|-------------|---------|
  | Train | 138,334 | 105,224 | -33,110 |
  | Val | 27,539 | 20,751 | -6,788 |
  | Test | 사용 안 함 (x축 포함 원본 사용) | | |

#### no_x Config 5개 생성
- `training/configs/cnn_ct_cbam_no_x.yaml` — CBAM (experiment: ct_cbam_768_no_x)
- `training/configs/cnn_ct_unified_no_x.yaml` — ResNet18 (experiment: ct_unified_resnet18_no_x)
- `training/configs/cnn_ct_convnext_no_x.yaml` — ConvNeXt-Tiny (experiment: ct_convnext_tiny_no_x)
- `training/configs/cnn_ct_efficientnet_b4_no_x.yaml` — EfficientNet-B4 신규 (experiment: ct_efficientnet_b4_no_x)
- `training/configs/cnn_ct_hdcnn_no_x.yaml` — HD-CNN (experiment: ct_hdcnn_no_x)
- **공통**: train/val = resize512_no_x, test = resize512 (x축 포함, 실운영 시뮬레이션)

#### 전 모델 전처리 resize512 통일
- 기존 cropped(1024), patch(512), 원본(1024) → 전부 resize512로 변경
  | Config | 변경 전 | 변경 후 |
  |--------|---------|---------|
  | `cnn_ct_late_fusion.yaml` | cropped 1024 | resize512 |
  | `cnn_ct_hdcnn.yaml` | cropped 1024 | resize512 |
  | `cnn_ct_hierarchical.yaml` | 원본 1024 | resize512 |
  | `cnn_ct_metadata_balanced.yaml` | patch 512 | resize512 |

#### 메타/Late Fusion은 x축 제외 불필요
- `model_metadata.py`, `model_late_fusion.py`에 axis 메타데이터 (x=0, y=1, z=2) 이미 포함
- 모델이 축별로 다른 판단 가능 → x축 데이터 포함 학습이 설계 의도에 맞음

#### 추론 파이프라인 설계
```
이미지 입력
├─ 파일명 _x_ → 자동 정상 판정 (모델 불필요)
└─ 파일명 _y_ 또는 _z_ → CNN 모델 → 결함 분류
       └─ 배터리 단위 종합 → 최종 불량 여부
```

#### 학습 명령어
```bash
# CBAM (x축 제외)
python models/ct_cnn/train.py --config cnn_ct_cbam_no_x

# ResNet18 (x축 제외)
python models/ct_cnn/train.py --config cnn_ct_unified_no_x

# ConvNeXt-Tiny (x축 제외)
python models/ct_cnn/train.py --config cnn_ct_convnext_no_x

# EfficientNet-B4 (x축 제외)
python models/ct_cnn/train.py --config cnn_ct_efficientnet_b4_no_x

# HD-CNN (x축 제외)
python models/ct_cnn/train_hdcnn.py --config cnn_ct_hdcnn_no_x
```

---

### 2026-02-08 - RGB AE v2 모델 개선 및 학습

#### 모델 구조 개선
- **Bottleneck 변경**: 1×1 → 4×4 (공간 정보 유지)
  - Encoder: 32×32×512 → 4×4×512 → 1024 (latent)
  - Decoder: 1024 → 4×4×512 → 32×32×512
- **Loss 함수**: MSE → MSE+SSIM 혼합 (7:3)
  - MSE: 픽셀 단위 차이 감지
  - SSIM: 구조적 유사도 (패턴/텍스처 차이)
- **리사이즈**: 비율 유지 + 패딩 (ResizeWithPadding)
  - 1920×1080 → 512×288 → 패딩 → 512×512

#### 데이터 분리
- **배터리 ID 분리**: Train 211 / Val 60 / Test 31 (누수 없음)
- **Train**: 정상만 5,746개 (`rgb_train_normal.txt`)
- **Val/Test**: 정상 + 결함 전체

#### 테스트 결과
| 지표 | 값 |
|------|-----|
| ROC-AUC | **0.9781** |
| Normal Score | 1.25 ± 0.22 |
| Defect Score | 2.10 ± 0.10 |
| Threshold | 1.499 |

#### 수정 파일
- `models/rgb_ae/model.py` - Bottleneck 4×4, CombinedLoss 추가
- `training/data/transforms.py` - ResizeWithPadding 클래스 추가
- `training/configs/autoencoder_rgb.yaml` - 512, MSE+SSIM 설정
- `training/data/splits/rgb/rgb_train_normal.txt` - 정상 데이터만 추출

---

### 2026-02-07 - 데이터 Split 검증 완료

#### 검증 결과: 누수 없음 ✅
| Split | 배터리 수 | Train | Val | Test | 누수 |
|-------|----------|-------|-----|------|------|
| CT | 1,934 | - | - | - | 0개 ✅ |
| RGB | 302 | 211 | 60 | 31 | 0개 ✅ |

- CT/RGB 모두 배터리 단위로 올바르게 분리되어 있음
- 스크립트: `scripts/fix_split_by_battery.py` (검증/재생성용)

#### 다음 단계
- [ ] 모델 학습 및 평가 진행

---

### 2026-02-06 - Qwen3-VL 업그레이드 및 BBox 탐지 지원

#### VLM 모델 업그레이드: Qwen2-VL → Qwen3-VL
- **이유**: Qwen2-VL은 텍스트 분석만 가능, Qwen3-VL은 BBox 출력 지원
- **새 기능**:
  - `detect_defects()` - 결함 위치 탐지 (BBox 좌표 출력)
  - `analyze_with_grounding()` - 텍스트 분석 + BBox 동시 출력
  - 0-1000 정규화 좌표 → 픽셀 좌표 자동 변환
- **지원 모델**: 2B, 4B, 8B, 32B (Instruct)
- **요구사항**: transformers >= 4.57.0

#### 수정 파일
- **`models/vlm/inference.py`** - Qwen3-VL 전면 재작성
  - `Qwen3VLForConditionalGeneration` 사용
  - `detect_defects()`, `analyze_with_grounding()` 메서드 추가
  - `_parse_detection_response()` - BBox JSON 파싱
  - `get_model_info()` - capabilities에 bbox, 2d_grounding 추가
- **`models/vlm/test_vlm.py`** - 테스트 스크립트 업데이트
  - 모델 크기 목록 Qwen3-VL로 변경
  - `test_detection_with_model()` 함수 추가
  - `--detection` 옵션 추가 (BBox 탐지 테스트)

#### Qwen3-VL 주요 특징
- **DeepStack Technology**: ViT 다중 레이어 → LLM 다중 레이어 주입
- **2D/3D Grounding**: 객체 위치 추론, embodied AI 지원
- **Visual Agent**: PC/모바일 GUI 조작 가능
- **릴리즈**: 2025.10~11월 순차 공개

#### VLG 비교 의미
- **기존**: VLM(텍스트) + VLG(BBox) 별도 비교
- **현재**: Qwen3-VL이 둘 다 지원 → VLG와 BBox 정확도 비교 가능
- **비교 포인트**: Qwen3-VL vs GroundingDINO BBox 정확도

#### 테스트 명령어
```bash
# VLM 기본 테스트
python models/vlm/test_vlm.py

# BBox 탐지 테스트
python models/vlm/test_vlm.py --detection

# 전체 테스트 (모델 로드 포함)
python models/vlm/test_vlm.py --full
```

#### 다음 단계
- [ ] Qwen3-VL 모델 다운로드 및 실제 테스트
- [ ] VLG(GroundingDINO)와 BBox 정확도 비교
- [ ] 웹앱에 BBox 시각화 통합

---

### 2026-02-05 - 모델 아키텍처 비교 분석 및 신규 모델 설정

#### HD-CNN 실험 결과 (실패)
- **Test F1**: 0.690 (Late Fusion 0.826 대비 -16.5%)
- **주요 문제**: module_normal → cell_normal 오분류 (5,113개, 48.8%)
- **원인**: Coarse 브랜치(cell/module 분류)가 Cell/Module 구분 실패
- **결론**: HD-CNN 구조는 본 데이터셋에 부적합, 폐기

#### Late Fusion 결과 (현재 최고 성능)
- **Test F1**: 0.826 (Best)
- **Accuracy**: 89.8%
- **클래스별 성능**:
  | Class | F1 | Recall |
  |-------|-----|--------|
  | cell_normal | 0.865 | 91.0% |
  | cell_porosity | 0.730 | 64.1% |
  | module_normal | 0.674 | 85.9% |
  | module_porosity | 0.942 | 96.9% |
  | module_resin | 0.921 | 100% |

#### CNN 아키텍처 비교 분석 (3×3 vs 7×7 커널)
- **ResNet18**: 11M params, 3×3 커널
  - 장점: 작은 특징 감지, 연산 효율
  - 단점: receptive field 작음, 넓은 문맥 파악 어려움
- **EfficientNet-B0**: 5.3M params, 3×3~5×5 커널 + SE blocks
  - 장점: 적은 파라미터 (과적합 감소 기대), Squeeze-and-Excitation
  - 오버피팅 경향이 있는 본 데이터에 적합 가능성
- **ConvNeXt-Tiny**: 28M params, 7×7 커널
  - 장점: 넓은 receptive field, 최신 아키텍처 (2022)
  - 문맥적 패턴 파악에 유리 (배터리 전체 구조 고려)

#### 신규 Config 생성
- **EfficientNet-B0**: `training/configs/cnn_ct_efficientnet.yaml`
  - backbone: timm, batch_size: 32, dropout: 0.3
- **ConvNeXt-Tiny**: `training/configs/cnn_ct_convnext.yaml`
  - backbone: timm, batch_size: 16, drop_path_rate: 0.1
- **공통 설정**: Raw Resize 512 (ResNet18 01-05와 동일 조건)

#### timm 모델 지원 추가
- **파일 생성**: `models/ct_cnn/model_timm.py`
  - TimmClassifier, EfficientNetClassifier, ConvNeXtClassifier
  - timm 라이브러리 기반 다양한 모델 지원
- **파일 수정**: `models/ct_cnn/model.py`
  - create_model() 함수에 timm backbone 분기 추가

#### 데이터 분할 검증 (누수 없음 확인)
- **배터리 ID 분리**: Train 92개, Val 18개, Test 24개 (총 134개)
- **중복 없음**: 모든 split 간 배터리 ID 교집합 없음
- **축 분포 일관성**: 84.3%가 x,y,z 모든 축 보유

#### 다음 단계 작업 목록
| # | 작업 | 설명 |
|---|------|------|
| 1 | EfficientNet-B0 학습 | Raw 512, 기본 조건 |
| 2 | ConvNeXt-Tiny 학습 | Raw 512, 7×7 커널 효과 검증 |
| 3 | HD-CNN + Metadata | cell/module 혼동 해결 시도 |
| 4 | Defect Attention | bbox 위치 정보 활용 집중 학습 |
| 5 | Late Fusion + Focal 강화 | cell_porosity 개선 |
| 6 | 아키텍처 비교 분석 | 최종 성능 비교 문서화 |

---

### 2026-02-01 - 패치 전략 테스트 및 클래스 밸런싱

#### CNN+Metadata 패치 전략 테스트 결과
- **Test 결과**: Accuracy=93.2%, F1 Macro=0.874
- **클래스별 성능**:
  | Class | F1 | Precision | Recall |
  |-------|-----|-----------|--------|
  | cell_normal | 0.84 | 0.78 | 0.91 |
  | cell_porosity | 0.94 | 0.97 | 0.91 |
  | module_normal | **0.64** | **0.47** | 0.99 |
  | module_porosity | 0.96 | 1.00 | 0.93 |
  | module_resin | 0.99 | 0.99 | 1.00 |
- **문제점**: module_normal precision 47% (module_porosity 20K개가 module_normal로 오분류)

#### 앙상블 테스트 (CNN+Metadata + CT AE)
- **결과**: 앙상블 효과 없음 (F1 변화 0%)
- **원인 분석**:
  - CNN/AE Agreement: 75%가 불일치 (cnn_only_defect)
  - AE가 대부분 "정상"으로 판정 → 앙상블이 CNN만 따라감
  - AE ROC-AUC 0.65로 낮아서 결함 탐지 못함
- **수정 파일**:
  - `models/inspector/ct_ensemble_inspector.py` - import 오류 수정, AE 튜플 반환 처리
  - `models/inspector/test_ct_ensemble.py` - 앙상블 테스트 스크립트 생성

#### 클래스 밸런싱 구현
- **문제**: Class 3 (module_porosity)이 83% 차지 → 과적합
- **해결**: Class 3 언더샘플링 (1,108,345 → 100,000)
- **결과**:
  | Class | 이전 | 이후 |
  |-------|------|------|
  | 0 (cell_normal) | 3.3% | 13.7% |
  | 1 (cell_porosity) | 4.0% | 16.4% |
  | 2 (module_normal) | 8.7% | 35.8% |
  | 3 (module_porosity) | 83.1% | 30.7% |
  | 4 (module_resin) | 0.8% | 3.3% |
  | Total | 1,333,606 | 325,261 |
- **생성 파일**:
  - `scripts/balance_split.py` - 밸런싱 스크립트
  - `training/data/splits/ct/patch/battery_train_balanced.txt` - 밸런싱된 train split
  - `training/configs/cnn_ct_metadata_balanced.yaml` - 밸런싱 config

#### 다음 단계
```bash
# 1. 밸런싱 데이터로 학습
python -m models.ct_cnn.train_metadata --config training/configs/cnn_ct_metadata_balanced.yaml

# 2. 성능 확인 후 CBAM 추가 실험
```

---

### 2026-01-29 - 전처리 스타일 통일 및 Axis 메타데이터 추가

#### 문제 1: 축(Axis) 상관관계 발견
- **증상**: 랜덤 crop 후에도 Val F1 = 99.4% (여전히 높음)
- **원인 분석**: 축별 라벨 분포 불균형
  - x축: 31,583 정상, 8 결함 (**99.97% 정상**)
  - y/z축: 정상/결함 혼재
- **결론**: 모델이 "어떤 축인지"를 학습 → 결함 패턴 학습 X

#### 해결책 1: Axis 메타데이터 추가
- **수정 파일**:
  - `models/ct_cnn/model_metadata.py` - METADATA_DIM: 1→2, axis 추출 함수 추가
  - `training/data/dataset_metadata.py` - 파일명에서 axis 추출 (x=0, y=1, z=2)
  - `models/ct_cnn/train_metadata.py` - dummy_metadata 크기 수정
- **메타데이터 구조**: `[battery_type, axis]` (이전: `[battery_type]`)
- **효과**: 모델이 축 정보를 명시적으로 받아 이미지 스타일로 축 추론 불가

#### 문제 2: 이미지 스타일 차이 (검은 패딩 비율)
- **분석 결과**:
  | 클래스 | 검은 영역 | 원인 |
  |--------|----------|------|
  | cell_normal | **0%** | 큰 영역 crop → 배터리로 가득 참 |
  | cell_porosity | **78.4%** | 가늘고 긴 defect bbox → 정사각형화 시 검은 패딩 |
- **결함 bbox 특성**: Width 평균 5px, Height 평균 659px (종횡비 144:1)
- **모든 결함이 세로로 긴 형태** → 정사각형화 시 100% 배터리 바깥 포함

#### 해결책 2: 정상 이미지 스타일 통일
- **수정 파일**: `scripts/preprocess_defect_direct.py`
- **변경 내용**: `random_crop_in_outline()` 함수 수정
  - 이전: 배터리 내부에서 큰 정사각형 영역 crop
  - 이후: 배터리 내부에서 **가늘고 긴 영역** crop → 정사각형화
- **결과**: 정상 이미지도 71-83% 검은 영역 (결함과 동일)
- **Split 저장 경로**: `defect_direct` → `defect_random`으로 변경 (덮어쓰기)

#### 전처리 실행 명령어
```bash
python scripts/preprocess_defect_direct.py \
  --output /mnt/d/battery-defect-random \
  --size 512 --normal-mode random
```

#### 학습 실행 명령어 (전처리 완료 후)
```bash
python -m models.ct_cnn.train_metadata --config training/configs/cnn_ct_random_crop.yaml
```

---

### 2026-01-28 - CT AE 분석 및 전처리 문제 해결

#### CT AE 테스트 결과 분석
- **ROC-AUC**: 0.653 (낮음)
- **핵심 문제**: Cell과 Module 점수 분포가 완전히 다름
  - cell_normal: 0.150, cell_porosity: 0.152 (거의 동일 → 분리 불가)
  - module_normal: 0.253, module_porosity: 0.310 (분리 가능)
- **결과 파일**: `models/ct_ae/results/test_ct_ae_20260128_165451.json`

#### Cell/Module 별도 Threshold 적용
- **수정 파일**: `models/ct_ae/checkpoints/threshold.json`
- **Cell Threshold**: 0.12 (Recall 우선, 결함→결함 70%)
- **Module Threshold**: 0.28 (균형, 68%/68%)
- **앙상블 코드 수정**: `models/inspector/ct_ensemble_inspector.py`
  - 파일명에서 cell/module 자동 판별
  - 타입별 threshold 적용

#### CNN+Metadata 학습 문제 발견
- **증상**: Val F1 = 99.99% (비정상적으로 높음)
- **원인**: 데이터 누수 아님, **이미지 스타일 차이 학습**
  - 정상 이미지: battery_outline 전체 축소 (어둡고 단순, ~23KB)
  - 결함 이미지: defect bbox 확대 crop (밝고 복잡, ~50KB)
  - 모델이 "결함 패턴"이 아닌 "이미지 스타일"을 학습

#### 전처리 스크립트 수정
- **수정 파일**: `scripts/preprocess_defect_direct.py`
- **변경 사항**: `--normal-mode random` 옵션 추가
  - 정상 이미지도 배터리 내부에서 랜덤 crop
  - 결함 이미지와 동일한 스타일로 통일
- **새 전처리 실행**:
  ```bash
  python scripts/preprocess_defect_direct.py \
    --output /mnt/d/battery-defect-random \
    --size 512 --defect-padding 200 \
    --normal-mode random --workers 8
  ```
- **진행 상황**: 77k/179k (43%)

---

### 2026-01-27 - CT 앙상블 아키텍처 구현 및 전처리 수정

#### 전처리 좌표 오류 수정
- **문제**: 이전 전처리에서 1024x1024 리사이즈된 이미지에 4000x4000 좌표 적용 → 배터리가 왼쪽으로 치우침
- **해결**: 원본 4000x4000 이미지에서 직접 crop 후 리사이즈
- **수정 파일**: `scripts/preprocess.py` (IMAGE_BASE 경로 변경)

#### Battery Outline Crop v2 완료
- **경로**: `/mnt/d/battery-cropped-v2/`
- **파일 수**: 179,024개
- **이미지 크기**: 1024x1024
- **Split 파일**: `training/data/splits/ct/cropped/battery_*.txt`

#### Defect Direct Crop 전처리 (진행중)
- **스크립트**: `scripts/preprocess_defect_direct.py`
- **출력 경로**: `/mnt/d/battery-defect-direct/`
- **이미지 크기**: 512x512
- **결함 이미지**: defect bbox + 200px padding → 512x512 crop
- **정상 이미지**: battery_outline crop → 512x512

```bash
# 전처리 명령어
python scripts/preprocess_defect_direct.py \
  --output /mnt/d/battery-defect-direct \
  --size 512 --defect-padding 200 \
  --normal-mode outline --workers 8
```

#### CT 앙상블 검사기 구현
- **파일**: `models/inspector/ct_ensemble_inspector.py`
- **구조**:
  ```
  [Defect Crop 512x512] → CNN+Metadata → 5클래스 분류
                                    ↓
                              앙상블 결합 → 최종 판정
                                    ↑
  [Outline Crop 1024x1024] → AutoEncoder → 이상 점수
  ```
- **앙상블 가중치**: CNN 0.7, AE 0.3
- **결합 전략**:
  | CNN | AE | 결과 | 확신도 |
  |-----|-----|------|--------|
  | 결함 | 이상 | 결함 | 가중평균 |
  | 결함 | 정상 | 결함 | 80% |
  | 정상 | 이상 | 정상+경고 | 70% |
  | 정상 | 정상 | 정상 | 가중평균 |

#### CT AutoEncoder 학습 스크립트 생성
- **학습**: `models/ct_ae/train.py`
- **테스트**: `models/ct_ae/test.py`
- **Config**: `training/configs/autoencoder_ct.yaml`
- **특징**:
  - 정상 이미지만으로 학습 (Anomaly Detection)
  - ROC 기반 threshold 자동 계산
  - CSV + TensorBoard 로깅

```bash
# CT AE 학습 명령어
python models/ct_ae/train.py --config autoencoder_ct
```

#### CNN+Metadata Config 생성
- **Config**: `training/configs/cnn_ct_defect_crop.yaml`
- **데이터**: Defect direct crop (512x512)
- **Split**: `training/data/splits/ct/defect_direct/` (생성 필요)

```bash
# CNN+Metadata 학습 명령어 (전처리 완료 후)
python models/ct_cnn/train_metadata.py --config cnn_ct_defect_crop
```

#### 불필요한 파일 삭제
- ~~`/mnt/d/battery-cropped/`~~ (좌표 오류 버전)
- ~~`/mnt/d/battery-defect-crop/`~~ (좌표 오류 버전)
- ~~`training/data/splits/ct/defect_crop/`~~ (이전 split)

---

### 2026-01-27 - 향후 개선 사항 (학습 결과 확인 후 적용)

#### 1. AE Gaussian Blur 적용
- **목적**: 미세 노이즈 무시, 큰 형태적 이상에만 집중
- **적용 위치**: MSE loss 계산 전 blur 적용
- **적용 조건**: 정상 이미지 노이즈로 오탐 많을 경우

#### 2. Cell/Module 별도 Threshold
- **이유**: Module이 Cell보다 구조 복잡 → baseline reconstruction error 높을 수 있음
- **구현**: threshold.json에 cell_threshold, module_threshold 분리 저장
- **적용 조건**: Validation에서 cell/module 점수 분포 차이 클 경우

#### 3. XGBoost Meta-Learner
- **현재**: 규칙 기반 앙상블 (하드코딩된 가중치)
- **개선**: 학습 기반 최적 결합
- **Features**: CNN 5클래스 확률, AE anomaly_score, battery_type
- **장점**: 최적 가중치 자동 학습, 비선형 결합, Feature importance 해석

---

### 2026-01-20 - 이중 보정 완화 및 FocalLoss 안정성 개선

#### 이중 보정 완화 (WeightedSampler + FocalLoss 동시 사용으로 인한 과보정 방지)

| 항목 | 이전 | 이후 | 이유 |
|------|------|------|------|
| `gamma` | 3.0 | **1.5** | Sampler가 이미 희소 클래스 보정 |
| `label_smoothing` | 0.15 | **0.07** | 타겟 분포 덜 흐리게 |
| `resin_overflow alpha` | 25.0 | **18.0** | Loss 가중치 완화 |

#### FocalLoss 코드 안정성 개선 (`models/ct_cnn/train.py`)

1. **p_t clamp 추가** (수치 안정성)
   ```python
   p_t = p_t.clamp(min=1e-6, max=1-1e-6)
   ```
   - p_t가 0 또는 1 근처일 때 log/pow 연산 안정화

2. **alpha register_buffer** (device/dtype 자동 동기화)
   ```python
   if alpha is not None:
       self.register_buffer('alpha', alpha)
   ```
   - `criterion.to(device)` 호출 시 alpha도 자동으로 GPU 이동

#### 수정 파일
- `training/configs/cnn_ct_unified.yaml`
- `training/configs/cnn_ct_cbam.yaml`
- `models/ct_cnn/train.py` (FocalLoss 클래스)

---

### 2026-01-20 - Config 기능 전면 구현 및 문서 업데이트

#### Config 설정 → 코드 연결 완료
기존에 Config에 정의되어 있었지만 **실제로 코드에서 사용되지 않던 기능들** 전부 구현:

| 기능 | 파일 | 상태 |
|------|------|------|
| **WeightedRandomSampler** | `training/data/dataloader.py` | ✅ 구현 |
| **FocalLoss** | `models/ct_cnn/train.py` | ✅ 구현 |
| **Label Smoothing** | `models/ct_cnn/train.py` (FocalLoss 내장) | ✅ 구현 |
| **Config 기반 Augmentation** | `training/data/transforms.py` | ✅ 구현 |
| **동적 Early Stopping** | `models/ct_cnn/train.py` | ✅ 구현 |
| **save_top_k** | `models/ct_cnn/train.py` | ✅ 구현 |
| **tensorboard.enabled** | `models/ct_cnn/train.py` | ✅ 구현 |
| **log_grad_cam** | `training/visualization/tensorboard_logger.py` | ✅ 구현 |

#### 수정된 파일 목록
- `training/data/transforms.py` - `build_transforms_from_config()` 함수 추가
- `training/data/dataloader.py` - `_create_weighted_sampler()` 함수 추가, `class_balancing` 파라미터
- `models/ct_cnn/train.py` - FocalLoss 클래스, 동적 config 처리, Top-K 체크포인트
- `models/ct_cnn/test.py` - Config 기반 transform 적용
- `models/rgb_ae/train.py` - Config 기반 transform 적용
- `models/rgb_ae/test.py` - Config 기반 transform 적용
- `training/visualization/tensorboard_logger.py` - `log_gradcam()` 메서드 추가

#### Config 값 조정
- `class_weights[1]` (cell_porosity): 5.0 → **4.0** (WeightedSampler와 함께 사용 시 과도한 가중치 방지)
- `focal_loss.gamma`: **3.0** 유지

#### TensorBoard 가이드 문서 업데이트 (`docs/TENSORBOARD_GUIDE.md`)
- **설정값 현행화**: Image Size 512→1024, Batch Size 32→16, Focal Gamma 2.0→3.0
- **새 섹션 추가**:
  - 섹션 5: 핵심 평가 지표 상세 설명 (TP/FP/FN, Precision/Recall, Focal Loss 수식)
  - 섹션 6: 주요 모니터링 체크리스트 (매 Epoch/5 Epoch/10 Epoch별)
  - 섹션 7: 문제 상황별 대응 가이드 (Recall 낮을 때, 과적합, 정체)
  - 섹션 9: 현재 학습 설정 요약 테이블
- **Grad-CAM 시각화** 항목 추가
- **Top-K 체크포인트** 파일 구조 추가

#### 학습 실행 준비 완료
```bash
python models/ct_cnn/train.py --config cnn_ct_unified
```

---

### 2026-01-19 - cell_porosity 성능 개선을 위한 Config 수정
- ✅ **cell_porosity 문제 분석**
  - Recall: 33% (67%가 cell_normal로 오분류)
  - 원인: 데이터 불균형 + 시각적 유사성 + 과적합
- ✅ **Config 개선 적용** (`training/configs/cnn_ct_unified.yaml`)
  - `class_weights`: cell_porosity 3.0 → **5.0** (더 강한 가중치)
  - `focal_loss.gamma`: 2.0 → **3.0** (어려운 샘플에 집중)
  - `early_stopping.patience`: 10 → **5** (과적합 조기 방지)
  - `num_workers`: 16 → **4** (RAM OOM 방지)
- ✅ **데이터 증강 강화**
  - RandomRotation: 15° → **30°**
  - ColorJitter: 0.2 → **0.3**
  - **RandomAffine 추가** (translate, scale)
  - **GaussianBlur 추가** (블러 내성)
- ⏳ **CT CNN 재학습 대기** (모든 config 기능 구현 완료 후)
  - 이전 학습: Epoch 8까지 진행, Best F1: 0.8275 (Epoch 5)
  - 새 학습: WeightedSampler + FocalLoss + Label Smoothing 전부 적용 예정

### 2026-01-18 - 전처리 분리 및 이미지 크기 1024 적용
- ✅ **전처리 스크립트 생성** (`scripts/preprocess.py`)
  - 원본 이미지 → 1024x1024 리사이즈 후 D드라이브에 PNG 저장
  - `--skip-existing` 옵션으로 이미 처리된 파일 건너뛰기
  - `preprocessed_*.txt` 분할 파일 자동 생성
  - 사용법: `python scripts/preprocess.py --size 1024 --output /mnt/d/battery-preprocessed --format PNG`
- ✅ **Albumentations 지원 추가** (`training/data/transforms.py`)
  - CLAHE (대비 향상), Sharpen (선명화), ElasticTransform (변형)
  - 미세 결함 탐지 성능 향상 기대
  - `get_albumentations_transforms()` 함수 추가
- ✅ **학습/테스트 코드 전처리 옵션 적용**
  - `models/ct_cnn/train.py`, `test.py` - `preprocessed`, `use_albumentations` 옵션
  - `models/rgb_ae/train.py`, `test.py` - `get_transforms()` 사용
  - `models/inspector/predictor.py` - config 기반 transform 적용
  - `training/data/dataloader.py` - 새 옵션 파라미터 추가
  - `training/data/dataset.py` - `preprocessed` 옵션 지원
- ✅ **모든 config 파일 업데이트**
  - `image_size: 512 → 1024` (CT, RGB 모두)
  - `preprocessed: true` 옵션 추가
  - `use_albumentations: true` 옵션 추가
  - 수정 파일: `cnn_ct_unified.yaml`, `cnn_ct_cbam.yaml`, `autoencoder_rgb.yaml`, `autoencoder_rgb_normal.yaml`, `autoencoder_rgb_defect.yaml`
- ✅ **오래된 분할 파일 정리**
  - `rgb/cell/` 폴더 삭제 (78,844개 중복 파일)
  - `backup_defect_training` 폴더 삭제
- 🔄 **이미지 전처리 실행 중**
  - 총 260,665개 이미지 처리 중
  - 출력: `/mnt/d/battery-preprocessed/`

### 2026-01-07 - 웹앱 개선 및 TensorBoard 추가
- ✅ **VLM/VLG 4클래스 분류 적용** (`webapp/pages/processing.py`, `summary.py`)
  - 기존: 정상/불량 2분류 → 정상/내부불량/외부불량/복합불량 4분류
  - CT 결함 → 내부불량, RGB 결함 → 외부불량, 둘 다 → 복합불량
- ✅ **VLG 외부결함 label 매핑 수정** (`models/vlg/prompts.py`)
  - pollution, contamination, scratch, damage 키워드 추가
  - RGB 이미지 결함 탐지 정상 작동
- ✅ **RGB AE threshold 수정** (`models/rgb_ae/checkpoints/threshold.json`)
  - 기존 1.5665 → 2.9961 (mean + 2.5*std, 불량 학습 AE용)
  - 불량 데이터로 학습한 AE이므로 정상이 낮은 점수
- ✅ **RGB AE test.py TensorBoard 로깅 추가**
  - Confusion Matrix, ROC Curve, PR Curve, Score Distribution
  - 재구성 결과 이미지 (Original/Reconstructed/Difference)
  - 사용법: `python models/rgb_ae/test.py --checkpoint <path>`
- ✅ **CBAM 학습 실험 완료**
  - ResNet18 기본: F1=0.8335 (Best)
  - ResNet18+CBAM: F1=0.8022 (-3.1%)
  - 결론: 과적합 문제로 CBAM 미적용, 기본 ResNet18 유지
- ✅ **웹앱 바운딩 박스 두께 증가** (`summary.py`, `detail.py`)
  - width=3 → width=6 (가시성 개선)

### 2026-01-06 (저녁) - 웹앱 실제 모델 연동
- ✅ **통합 검사기 웹앱 연동** (`webapp/pages/processing.py`)
  - `@st.cache_resource`로 모델 싱글톤 로드
  - CT CNN + RGB AE 실제 추론 연결
  - 임시 파일 저장 후 추론 → 정리
- ✅ **RGB AE predictor 로직 수정** (`models/inspector/predictor.py`)
  - 점수 해석 수정: `score > threshold → defect`
  - 신뢰도 계산 로직 수정
- ✅ **추론 테스트 성공**
  - CT: cell_normal 예측 (신뢰도 100%)
  - RGB: anomaly_score 1.80 > threshold 1.57 → 외부불량

### 2026-01-06 (저녁) - RGB AE 테스트 완료
- ✅ **RGB AE 테스트 실행** (`models/rgb_ae/test.py`)
  - 테스트 데이터: 11,719개 (Normal: 4,053 / Defect: 7,666)
  - ROC-AUC: **0.9644** (우수한 분리 성능)
  - Accuracy: **97.86%**, F1 Score: **98.39%**
  - Normal Score: 0.9349 ± 0.3615 (낮음)
  - Defect Score: 2.0760 ± 0.1580 (높음)
- ✅ **스코어 해석 수정**
  - 모델이 Defect 데이터로 학습 → Normal 이미지가 더 잘 재구성됨
  - 따라서 높은 점수 = Defect (원래 예상과 반대)
  - test.py 로직 수정 완료
- ✅ **Threshold 최적화**
  - 기존: 2.9961 (k-sigma 방식)
  - 최적: 1.5665 (ROC 곡선 기반 TPR-FPR 최대화)
  - 결과: `models/rgb_ae/results/test_results.json`

### 2026-01-06 (오후) - 통합 검사기 구현 및 웹페이지 연동
- ✅ **통합 검사 모듈 구현** (`models/inspector/`)
  - `predictor.py`: CTCNNPredictor, RGBAEPredictor 클래스
  - `ensemble.py`: EnsemblePredictor 클래스 (CT+RGB 종합 판정)
  - `gradcam.py`: GradCAM, GradCAMPlusPlus 구현
  - 최종 판정: 정상, 내부불량, 외부불량, 복합불량
- ✅ **웹페이지 CT+RGB 듀얼 업로드 지원**
  - `webapp/pages/home.py`: CT/RGB 분리 업로드 UI
  - `webapp/pages/processing.py`: 분석 모드별 추론 (inspector/ct_only/rgb_only)
  - `webapp/pages/summary.py`: CT+RGB 결과 나란히 표시
  - `webapp/utils/session.py`: set_uploaded_images(), reset_analysis() 수정
- ✅ **VLM/VLG CT+RGB 지원**
  - CT 이미지: 내부 결함 분석 (porosity, resin)
  - RGB 이미지: 외부 결함 분석 (pollution, scratch, damage)
  - 통합 검사 모드: CT/RGB 각각 분석 후 종합

### 2026-01-06 - RGB AutoEncoder 코드 구현
- ✅ **RGB AE 모델 구현** (`models/rgb_ae/model.py`)
  - `ConvAutoEncoder`: Encoder-Bottleneck-Decoder 구조
  - 256x256 입력, latent_dim=512
  - `get_anomaly_score()`: 재구성 오류 기반 이상 점수 계산
- ✅ **RGB AE Trainer 구현** (`models/rgb_ae/train.py`)
  - MSE Loss 기반 재구성 학습
  - Mixed Precision (AMP) 지원
  - ReduceLROnPlateau 스케줄러
  - k-sigma 방식 threshold 자동 계산
  - TensorBoard 로깅 통합
- ✅ **RGB AE Tester 구현** (`models/rgb_ae/test.py`)
  - ROC-AUC, Accuracy, F1 Score 계산
  - 재구성 결과 시각화 (원본/재구성/에러맵)
  - Score 분포 시각화 (Normal vs Defect)
  - Optimal Threshold 탐색
- ✅ **RGB 데이터 복사 스크립트** (`scripts/copy_rgb_images.py`)
  - D드라이브(/mnt/d/) → Linux(/home/ubuntu/battery-data/)
  - Split 파일 기준 필요 이미지만 복사 (~59,263개)
  - 병렬 복사 (8 workers)

### 2026-01-06 - CT CNN 학습 완료 및 CBAM 구현
- ✅ **CT CNN 학습 완료** (18 epoch, Early Stop)
  - Best Val Accuracy: 83.07% (epoch 8)
  - Best Val F1: 0.8329
- ✅ **Test 평가 실행**
  - Test Accuracy: 77.45% (-5.6% vs Val)
  - Test F1 Macro: 0.7881
  - ROC-AUC: 0.9534
  - 과적합 확인 (Val→Test 성능 하락)
- ✅ **CBAM 모듈 구현** (`models/ct_cnn/model.py`)
  - `CBAM` 클래스: Channel Attention + Spatial Attention
  - `ResNet18CBAM` 클래스: layer3, layer4 뒤에 CBAM 추가
- ✅ **CBAM Config 생성** (`training/configs/cnn_ct_cbam.yaml`)
  - 이미지 크기: 512 → 768 (기공 디테일 보존)
  - Batch size: 32 → 16 (GPU 메모리)
  - Focal Loss gamma: 2.0 → 3.0 (어려운 샘플 집중)
- ✅ **test.py 수정** - 다중분류(5클래스) 지원
  - BCEWithLogitsLoss → CrossEntropyLoss
  - config에서 split 경로 읽도록 수정

### 2026-01-05 (오후) - UI 개선 및 문서 업데이트
- ✅ **Webapp 라이트 테마 적용** - 참조 디자인 기반으로 변경
- ✅ **5클래스 통일 체계 구현**
  - VLM: 정상 판정 프롬프트 추가
  - VLG: 키워드→5클래스 매핑 (`prompts.py`)
  - Webapp: 결함 정보 매핑 (`defect_info.py`)
- ✅ **Summary 페이지 개선**
  - 상세보기에서 이미지 제거, 점수/매핑 정보만 표시
  - 기술 용어 → 사용자 친화적 한글 변환
  - VLG "Label, Conf" → "결함 유형, 신뢰도"
- ✅ **VLG 가중치 다운로드** - `models/vlg/weights/groundingdino_swint_ogc.pth` (662MB)
- ✅ **문서 업데이트**
  - `docs/implementation_structure.md` - VLM/VLG/Webapp 구조 추가
  - `TASK.md` - 현재 학습 상태, 프로젝트 구조 업데이트

### 2026-01-05 (오전) - Streamlit UI 구현 완료
- ✅ **Streamlit 웹앱 구현** (`webapp/`)
  - `app.py`: 메인 앱 (페이지 라우팅)
  - `pages/home.py`: 이미지 업로드, 모달리티 선택
  - `pages/processing.py`: 3개 모델 추론 진행 애니메이션
  - `pages/summary.py`: 3개 모델 결과 요약, 종합 판정
  - `pages/detail.py`: TensorBoard 스타일 상세 대시보드
  - `utils/styles.py`: 다크 테마 CSS (TensorBoard 스타일)
  - 실행: `streamlit run webapp/app.py`

### 2026-01-05 - VLM/VLG 구현 완료
- ✅ **VLM (Qwen2-VL) 구현** (`models/vlm/`)
  - `inference.py`: VLMInference 클래스 (이미지 분석, 배치 처리, Zero-shot 분류)
  - `prompts.py`: CT/RGB 이미지 분석용 프롬프트 템플릿
  - 지원 모델: 2B, 7B, 72B
  - 테스트: `python models/vlm/test_vlm.py --full`
- ✅ **VLG (GroundingDINO) 구현** (`models/vlg/`)
  - `inference.py`: VLGInference 클래스 (결함 위치 탐지, 시각화)
  - `prompts.py`: 결함 유형별 텍스트 프롬프트 (porosity, resin, pollution 등)
  - 지원 모델: SwinT, SwinB
  - 테스트: `python models/vlg/test_vlg.py --full`
- ✅ CT CNN 학습 진행 중 (num_workers=16으로 속도 개선)

### 2026-01-04 - TensorBoard Logger 기능 강화
- ✅ **TensorBoard 시각화 기능 대폭 확장** (`training/visualization/tensorboard_logger.py`)
  - Confusion Matrix 이미지 로깅
  - 클래스별 TP/FP/FN 에러 분석
  - Error Summary 테이블 이미지
  - 클래스별 F1/Precision/Recall 스칼라
  - **PR Curve** (Precision-Recall Curve) - 클래스별
  - **클래스 분포 시각화** (데이터셋 불균형 확인)
  - **예측 확률 히스토그램** (클래스별 Softmax 분포)
  - **예측 신뢰도 분포** (정답/오답 비교)
  - 모델 구조 그래프
- ✅ **train.py 업데이트**
  - 모든 TensorBoard 로깅 기능 연동
  - 스케줄러 타입 변환 버그 수정
- ✅ **TensorBoard 가이드 문서 작성** (`docs/TENSORBOARD_GUIDE.md`)
  - 로그 구조 및 사용법
  - 탭별 지표 해석 방법
  - 학습 상태 진단 체크리스트

### 2026-01-03 - 학습 준비 완료
- ✅ **프로젝트 구조 정리**
  - 불필요한 디렉토리/스크립트 삭제 (experiments/, ct_ae/, 다운로드 스크립트 등)
  - config 구조 통합 (`training/configs/`)
- ✅ **학습 코드 수정**
  - `dataloader.py`: 5클래스 다중분류 지원
  - `tensorboard_logger.py`: Confusion Matrix 로깅 추가
  - `train.py`: config 경로 수정
- ✅ **CT 통합 CNN 설정 완료**
  - 5클래스: cell_normal, cell_porosity, module_normal, module_porosity, module_resin_overflow
  - Focal Loss + Label Smoothing + Class Weights

### 2026-01-03 - CT 통합 + RGB 데이터 분할
- ✅ CT Cell + Module 통합 (5클래스 CNN)
- ✅ Train: 138,316 / Val: 26,662 / Test: 36,424
- ✅ 배터리 ID 기준 분할 (Data Leakage 방지)

---

## 📂 데이터 경로 (2026-02-12 기준)

| 경로 | 내용 | 크기 | 파일 수 |
|------|------|------|---------|
| `/mnt/d/battery-512/` | **단순 리사이즈 (현재 표준)** | 512x512 | ~201,402 |
| `/mnt/d/battery-cropped-v2/` | Battery outline crop (레거시) | 1024x1024 | 179,024 |
| `/mnt/d/battery-preprocessed/` | 전체 리사이즈 이미지 (레거시) | 1024x1024 | 260,665 |
| `/home/ubuntu/battery-data/` | 원본 이미지 | 4000x4000 | - |
| `/mnt/d/103.배터리 불량 이미지 데이터/` | 원본 라벨링 데이터 (JSON) | - | 179,024 |

### Split 파일 경로
| Split | 경로 | 용도 |
|-------|------|------|
| **현재 표준** | `training/data/splits/ct/resize512/battery_*.txt` | 전 모델 공통 (x축 포함) |
| **x축 제외** | `training/data/splits/ct/resize512_no_x/battery_{train,val}.txt` | no_x 학습용 |
| 레거시 | `training/data/splits/ct/cropped/battery_*.txt` | Outline crop (미사용) |

---

## 📞 Quick Commands

```bash
# ============================================================
# CT 앙상블 학습 (2026-01-27 추가)
# ============================================================

# 1. Defect direct crop 전처리 (진행중)
python scripts/preprocess_defect_direct.py \
  --output /mnt/d/battery-defect-direct \
  --size 512 --defect-padding 200 \
  --normal-mode outline --workers 8

# 2. CT AutoEncoder 학습 (outline crop)
python models/ct_ae/train.py --config autoencoder_ct

# 3. CT AutoEncoder 테스트
python models/ct_ae/test.py --config autoencoder_ct \
  --checkpoint models/ct_ae/checkpoints/ct_ae_best_*.pt

# 4. CNN+Metadata 학습 (defect crop, 전처리 완료 후)
python models/ct_cnn/train_metadata.py --config cnn_ct_defect_crop

# 5. 앙상블 테스트
python models/inspector/ct_ensemble_inspector.py

# TensorBoard (CT AE)
tensorboard --logdir models/ct_ae/logs --port 6008

# ============================================================
# 기존 명령어
# ============================================================

# 이미지 전처리 (1024x1024 리사이즈)
python scripts/preprocess.py --size 1024 --output /mnt/d/battery-preprocessed --format PNG --skip-existing

# RGB AE 학습 (데이터 복사 완료 후)
python models/rgb_ae/train.py --config autoencoder_rgb

# RGB AE 테스트
python models/rgb_ae/test.py --checkpoint models/rgb_ae/checkpoints/<best>.pt --visualize

# CT CNN 학습 시작
python models/ct_cnn/train.py --config cnn_ct_unified

# CT CNN + CBAM 학습
python models/ct_cnn/train.py --config cnn_ct_cbam

# ============================================================
# x축 제외 학습 (2026-02-12 추가)
# ============================================================

# CBAM (x축 제외)
python models/ct_cnn/train.py --config cnn_ct_cbam_no_x

# ResNet18 (x축 제외)
python models/ct_cnn/train.py --config cnn_ct_unified_no_x

# ConvNeXt-Tiny (x축 제외)
python models/ct_cnn/train.py --config cnn_ct_convnext_no_x

# EfficientNet-B4 (x축 제외)
python models/ct_cnn/train.py --config cnn_ct_efficientnet_b4_no_x

# HD-CNN (x축 제외)
python models/ct_cnn/train_hdcnn.py --config cnn_ct_hdcnn_no_x

# EfficientNet-B0 학습 (timm backbone)
python models/ct_cnn/train.py --config cnn_ct_efficientnet

# ConvNeXt-Tiny 학습 (timm backbone, 7x7 커널)
python models/ct_cnn/train.py --config cnn_ct_convnext

# TensorBoard 실행
tensorboard --logdir models/ct_cnn/logs --port 6006

# GPU 상태 확인
nvidia-smi

# VLM 테스트 (Qwen3-VL)
python models/vlm/test_vlm.py             # 기본 테스트
python models/vlm/test_vlm.py --detection # BBox 탐지 테스트
python models/vlm/test_vlm.py --full      # 전체 테스트 (모델 로드 포함)

# VLG 테스트
python models/vlg/test_vlg.py        # 기본 테스트
python models/vlg/test_vlg.py --full # 모델 로드 포함
python models/vlg/test_vlg.py --viz  # 시각화 테스트

# Streamlit 웹앱 실행
streamlit run webapp/app.py --server.port 8501
```

---

**작업 완료 후 이 파일을 업데이트하세요!**
