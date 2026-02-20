# TASK.md - 배터리 검사 AI 작업 관리

## 🎯 현재 상태

- **현재 Phase**: CT CNN 모델 고도화
- **최고 성능**: Late Fusion v2 (F1=0.803)
- **완료**: DeepLabV3+ 학습 실패 (Val F1=0.302), DRN+ASPP (F1=0.794), VLM Qwen3-VL 2B 500샘플 평가 (F1=0.082, zero-shot 무의미 확정)

---

## 📝 최근 작업 기록

### 2026-02-20
- ✅ **PORTFOLIO.md 최신성 검증 및 이슈 수정**
  - 이진 분류 성능 수치 검증 → 정확함 확인 (탐색 오류였음)
  - RGB AE `test.py` NaN threshold 버그 수정: `math.isnan()` 체크 추가하여 threshold.json fallback 정상 동작
  - RGB AE `test_results.json` 정리: NaN threshold로 인한 잘못된 accuracy/f1/confusion_matrix 제거, 재평가 필요 표기
  - PORTFOLIO.md RGB AE 섹션: 구버전 split 기준 수치 제거, 신버전 threshold(1.3878) 반영
  - PORTFOLIO.md VLM 섹션: RGB 3클래스 평가 결과 반영 (CT/RGB 모두 zero-shot 부적합), 수치 제외
  - VLM 역할 재정의: 정량적 분류 → 자연어 기반 결함 해석/소견서 생성 용도
- ✅ **README.md 전면 업데이트**
  - CT CNN: ResNet18 77.4% → Late Fusion v2 80.3% (15종 아키텍처 실험 반영)
  - RGB AE: 구버전 수치 제거, ROC-AUC 0.9095 (신버전 split 기준)
  - VLM: Qwen2-VL → Qwen3-VL, 자연어 해석 용도로 역할 재정의
  - 시스템 아키텍처 다이어그램 최신화 (Late Fusion, Qwen3-VL, GroundingDINO)
  - 학습 설정: image_size 1024→512, Late Fusion v2 config 반영
  - 기술 스택/프로젝트 구조: 전체 모델 목록 및 파일 구조 최신화
  - MODEL_PERFORMANCE.md 참고 문서 링크 추가

### 2026-02-19
- ✅ **VLM RGB 평가 지원 추가**
  - `models/vlm/prompts.py`: `RGB_CLASSES` 3클래스 정의 + `ZERO_SHOT_CLASSIFICATION_RGB` 프롬프트 추가
  - `models/vlm/inference.py`: `zero_shot_classify(modality='rgb')` 파라미터 추가
  - `models/vlm/test_vlm_eval.py`: modality별 클래스/키워드 매핑 동적 설정, RGB fallback 매핑 추가
  - `training/configs/vlm_eval_rgb.yaml`: RGB 3클래스 VLM 평가 Config 신규 생성
  - `scripts/fix_rgb_split_by_battery.py`: 구버전 RGB split 배터리 ID 기준 재분리 스크립트
  - 사용법: `python models/vlm/test_vlm_eval.py --config vlm_eval_rgb --model-size 2b --num-samples 500`
- ✅ **Qwen3-VL 2B 대규모 평가 실행** (500샘플, stratified)
  - 결과: Accuracy=20.8%, F1 macro=0.082, ROC-AUC=0.547
  - 거의 모든 이미지를 cell_normal로 예측 (489/500) → zero-shot 분류 무의미 확정
  - 소요시간: ~18분 (0.46 samples/sec, GPU)
  - 결과 파일: `models/vlm/results/test_vlm_qwen3vl_2b_sampled_500_20260219_152042.json`
  - TensorBoard: `models/vlm/logs/vlm_qwen3vl_2b_sampled_500_20260219_152042`
  - 8B는 GPU 12GB 제약 (CPU 오프로드 필요)으로 미실행
- ✅ **MODEL_PERFORMANCE.md 전체 업데이트** (`docs/MODEL_PERFORMANCE.md`)
  - DeepLabV3+ (freeze) 학습 결과 추가: Val F1=0.302, 9 epochs, backbone freeze로 학습 실패
  - VLM Qwen3-VL 평가 섹션 업데이트: 2B 500샘플 대규모 평가 결과 추가
  - 체크포인트 목록에 DeepLabV3 추가
  - 결론/권장 다음 단계 업데이트 (DRN+ASPP, DeepLabV3, VLM 반영)
- ✅ **PORTFOLIO.md 전면 업데이트** (`PORTFOLIO.md`)
  - CT CNN 성능: ResNet18 77.4% → 현재 최고 Late Fusion v2 80.3% 반영
  - 15종 아키텍처 실험 결과 요약 테이블 추가
  - Late Fusion v2 아키텍처 다이어그램 추가
  - 현재 split 기준 성능표 (7개 모델) 및 클래스별 F1 추가
  - VLM 모델 Qwen2-VL → Qwen3-VL 전면 업그레이드 반영
    - 아키텍처 다이어그램, 결과 페이지 UI, 프로젝트 구조, 권장 설정, 웹앱 설정 → Qwen3-VL
    - 역사적 실험 결과(5-2절 웹앱 비교 실험)에 Qwen3-VL 업그레이드 주석 추가
  - 핵심 성과 섹션 최신화 (체계적 실험 15종 강조)
  - 향후 개선 방향 업데이트

### 2026-02-18
- ✅ **DeepLabV3+ 사전학습 가중치 활용 분류 모델 구현** (`models/ct_cnn/model_deeplabv3.py`)
  - 원본 세그멘테이션 모델(DRN-D-54 + ASPP)의 backbone+ASPP 가중치를 그대로 로드
  - 원본 코드를 최대한 그대로 포팅하여 state_dict 키 100% 호환 (378개 키 로드 성공)
  - SynchronizedBatchNorm2d → nn.BatchNorm2d 교체
  - 세그멘테이션 decoder → GAP + Dropout + FC(256→5) 분류 헤드로 교체
  - freeze 시 backbone+ASPP를 eval 모드로 고정 (BatchNorm 안정성)
  - 39.4M total / 1,285 trainable (freeze 시 classifier FC만 학습)
- ✅ **학습 config 생성** (`training/configs/cnn_ct_deeplabv3.yaml`)
  - 2단계 학습 전략: freeze→classifier 학습(lr=0.001) → fine-tuning(선택)
  - batch_size=8, AdamW, CosineAnnealingWarmRestarts
- ✅ **create_model 디스패치 추가** (`models/ct_cnn/model.py`)
  - `model.name: deeplabv3` → `create_deeplabv3_model(config)` 연결
- 사용법:
  ```bash
  # 학습
  python models/ct_cnn/train.py --config training/configs/cnn_ct_deeplabv3.yaml
  # 테스트
  python models/ct_cnn/test.py --checkpoint models/ct_cnn/checkpoints/deeplabv3_best_*.pt
  ```

### 2026-02-15
- ✅ **DRN+ASPP 분류 모델 구현** (`models/ct_cnn/model_drn_aspp.py`)
  - DeepLabV3+ 스타일 다중 스케일 특징 추출 기반 분류 모델
  - ResNet50 backbone (output_stride=16) + ASPP (rates=[6,12,18]) + low-level feature fusion
  - Depthwise Separable Conv으로 파라미터 효율화 (26.5M trainable / 26.7M total)
  - stem + layer1 freeze로 과적합 방지
  - batch_size=1/8 모두 정상 동작 확인
- ✅ **학습 config 생성** (`training/configs/cnn_ct_drn_aspp.yaml`)
  - batch_size=8, weight_decay=0.05, lr=0.0001
  - 순수 이미지 모델 (메타데이터 미사용)
- ✅ **create_model 디스패치 추가** (`models/ct_cnn/model.py`)
  - `model.name: drn_aspp` → `create_drn_aspp_model(config)` 연결

- ✅ **원본 DeepLabV3+ 레퍼런스와 비교 분석 완료**
  - 소스 경로: `D:\모델\1.모델소스코드\모델1_DeepLabv3\pytorch-deeplab-xception-eval`
- ✅ **D드라이브 학습 모델 파일 심볼릭 링크로 연결**
  - `models/ct_cnn/checkpoints/deeplabv3_drn_ct.pt` → `D:\모델\2.AI학습모델파일\weights\모델1batteryct.pt` (467MB, CT 세그멘테이션, DRN-D-54, 4클래스)
  - `models/rgb_ae/checkpoints/deeplabv3_drn_rgb.pt` → `D:\모델\2.AI학습모델파일\weights\모델2batteryrgb.pt` (467MB, RGB 세그멘테이션, DRN-D-54, 4클래스)
  - 원본 모델은 세그멘테이션용 (픽셀별 결함 마스킹), 우리 DRN+ASPP는 분류용

---

## 🔍 원본 DeepLabV3+ vs DRN+ASPP 비교 분석

### 원본 구조 (`D:\모델`)
```
modeling/
├── deeplab.py          # 전체 모델 (backbone → ASPP → decoder → upsample)
├── aspp.py             # ASPP 모듈 (5 브랜치: 1x1 + rate 6/12/18 + image pooling)
├── decoder.py          # Decoder (low-level fusion + conv → 픽셀별 예측)
├── backbone/
│   ├── resnet.py       # ResNet101 (output_stride 16/8, Multi-Grid dilation)
│   ├── drn.py          # DRN-D-54 (8단계 layer, dilation 2/4)
│   ├── xception.py
│   └── mobilenet.py
└── sync_batchnorm/     # 분산 학습용 SyncBN
```

### Backbone 비교

| 구분 | 원본 | 우리 DRN+ASPP |
|------|------|---------------|
| Backbone | ResNet-**101** (직접 구현) 또는 DRN-D-54 | torchvision ResNet-**50** |
| Pretrained | `resnet101-5d3b4d8f.pth` 수동 로드 | `ResNet50_Weights.IMAGENET1K_V2` |
| Dilation | layer4: `_make_MG_unit` (multi-grid: dilation=[1,2,4]×base) | `replace_stride_with_dilation=[F,F,True]` (layer4 dilation=2) |
| Output stride | 16 (resnet) 또는 8 (drn) | 16 고정 |
| Low-level | layer1 출력 (256ch) | layer1 출력 (256ch) — 동일 |
| Freeze | freeze_bn 옵션만 | stem + layer1 전체 freeze (과적합 방지 강화) |

### ASPP 비교

| 구분 | 원본 (`aspp.py`) | 우리 (`ASPP` 클래스) |
|------|------|---------------|
| Dilation rates | OS=16: [1, 6, 12, 18], OS=8: [1, 12, 24, 36] | [6, 12, 18] (config 변경 가능) |
| 1x1 conv 브랜치 | `_ASPPModule(dilation=1)` 일반 conv | 별도 `nn.Sequential(Conv2d 1x1)` — 동일 역할 |
| Atrous conv | 일반 `nn.Conv2d` 3x3 | **DepthwiseSeparableConv** (파라미터 절약) |
| Image pooling | GAP → Conv2d 1x1 → BN → ReLU | GAP → Conv2d 1x1 → ReLU (BN 제거, batch=1 호환) |
| Projection | Conv2d(1280→256) → BN → ReLU → Dropout(0.5) | Conv2d(1280→256) → BN → ReLU → Dropout(0.5) — 동일 |
| 가중치 초기화 | Kaiming Normal 수동 적용 | torchvision 기본 초기화 |

### Decoder / Classification Head 비교

| 구분 | 원본 (`decoder.py`) | 우리 (DRNASPPClassifier) |
|------|------|---------------|
| Low-level 축소 | Conv2d(256→48) → BN → ReLU | Conv2d(256→48) → BN → ReLU — 동일 |
| Upsample | bilinear, align_corners=True | bilinear, align_corners=False |
| Concat 후 | 일반 Conv2d(304→256) × 2 + Dropout(0.5, 0.1) | DepthwiseSepConv(304→256) × 2 |
| **최종 출력** | Conv2d(256→num_classes) → upsample → 픽셀별 세그멘테이션맵 | **GAP → Dropout(0.5) → FC(256→5)** — 분류 헤드 |

### 원본에서 그대로 가져온 것
- ASPP 5-브랜치 구조 (1x1 + rate 6/12/18 + image pooling)
- Low-level feature (layer1, 256→48ch) + ASPP 출력 fusion
- Concat 채널 수 304 = 256 + 48
- Projection 1280→256

### 분류 태스크에 맞게 변경한 것
- 세그멘테이션 decoder → GAP + FC 분류 헤드
- ResNet101 → ResNet50 (분류에 충분, 메모리 효율)
- 일반 Conv → Depthwise Separable Conv (경량화)
- stem+layer1 freeze 추가 (분류 태스크 과적합 방지)
- Multi-Grid dilation → 단순 `replace_stride_with_dilation` (torchvision API 활용)

### 원본 모델 데이터 설정

#### 데이터 구조 (SimpleSegmentation 데이터로더)
```
base_dir/
├── frames/          ← 원본 이미지 (RGB로 로드)
│   ├── train/
│   ├── val/
│   └── test/
└── masks/           ← 세그멘테이션 마스크 (픽셀별 클래스 라벨)
    ├── train/
    ├── val/
    └── test/
```
이미지-마스크 1:1 매핑. 이미지는 `.convert('RGB')`로 로드.

#### 모델별 추론 설정

| 구분 | CT 모델 (`모델1batteryct.pt`) | RGB 모델 (`모델2batteryrgb.pt`) |
|------|------|------|
| backbone | DRN-D-54 | DRN-D-54 |
| num_classes | 4 | 4 |
| crop 방식 | `none` (960px 전체 입력) | `slide` (640px 윈도우 슬라이딩) |
| 입력 데이터 | `dataset/CT` | `dataset/RGB` |
| 태스크 | 픽셀별 세그멘테이션 | 픽셀별 세그멘테이션 |

#### test2.py — 2-Stage 앙상블 구조
1. `model` (4클래스): slide crop으로 결함 종류별 세그멘테이션
2. `model_base` (2클래스): 전체 이미지 리사이즈 → 배터리 윤곽선(outline) 검출
3. 윤곽선 영역 위에 결함 예측을 오버레이

#### 학습 설정 (train.py)
- 데이터셋: Pascal VOC/COCO/Cityscapes 형식 (배터리 데이터를 이 형식에 맞춰 변환)
- 학습률: backbone 1x, ASPP+decoder **10x** (차등 학습률)
- optimizer: SGD (momentum=0.9, weight_decay=5e-4)
- scheduler: poly
- loss: CE 또는 Focal Loss

#### 원본 vs 우리 프로젝트 데이터 차이

| 구분 | 원본 | 우리 프로젝트 |
|------|------|-------------|
| 데이터 형식 | frames+masks 폴더 구조 | split txt 파일 (경로\t라벨) |
| 라벨 | 픽셀별 마스크 이미지 | 이미지 단위 정수 라벨 (0~4) |
| 데이터 경로 | `testset/CT/`, `testset/RGB/` | `/mnt/d/battery-512/` |
| CT/RGB | **별도 모델, 별도 데이터** | CT 전용 (RGB는 별도 rgb_ae 모델) |
| 클래스 수 | 4 (세그멘테이션) | 5 (분류) |

---

## 🏗️ DeepLabV3+ 분류 모델 상세 (`model_deeplabv3.py`)

### 설계 철학

기존 `model_drn_aspp.py`는 torchvision ResNet50 backbone + 자체 구현 ASPP를 사용하여 ImageNet 가중치만 활용.
반면 `model_deeplabv3.py`는 **배터리 CT 이미지로 이미 학습된** 원본 세그멘테이션 모델의 가중치를 직접 재활용.

핵심 차이: **domain-specific pretrained** (배터리 CT) vs **generic pretrained** (ImageNet)

### 아키텍처

```
Input (3, 512, 512)
  → DRN-D-54 backbone (output_stride=8):
    ┌─ layer0: Conv7x7 stride=1 → 16ch (512×512)
    ├─ layer1: Conv3x3 → 16ch (512×512)
    ├─ layer2: Conv3x3 stride=2 → 32ch (256×256)
    ├─ layer3: 3×Bottleneck stride=2 → 256ch (128×128) ← low_level_feat (미사용)
    ├─ layer4: 4×Bottleneck stride=2 → 512ch (64×64)
    ├─ layer5: 6×Bottleneck dilation=2 → 1024ch (64×64)
    ├─ layer6: 3×Bottleneck dilation=4 → 2048ch (64×64)
    ├─ layer7: Conv3x3 dilation=2 → 512ch (64×64)
    └─ layer8: Conv3x3 dilation=1 → 512ch (64×64)
  → ASPP (512ch 입력, output_stride=8):
    ┌─ aspp1: 1×1 conv → 256ch
    ├─ aspp2: 3×3 conv (rate=12) → 256ch
    ├─ aspp3: 3×3 conv (rate=24) → 256ch
    ├─ aspp4: 3×3 conv (rate=36) → 256ch
    ├─ global_avg_pool: GAP → 1×1 conv → 256ch
    └─ concat 1280ch → 1×1 conv → 256ch → dropout(0.5)
  → Classification head (원본 decoder 대체):
    └─ GAP → Dropout(0.5) → FC(256→5)
```

### 모델 3종 비교

| 구분 | 원본 세그멘테이션 | DRN+ASPP (`model_drn_aspp.py`) | DeepLabV3+ 분류 (`model_deeplabv3.py`) |
|------|------|------|------|
| Backbone | DRN-D-54 | torchvision ResNet50 | DRN-D-54 (원본과 동일) |
| Pretrained | 배터리 CT 세그멘테이션 | ImageNet-1K V2 | **배터리 CT 세그멘테이션** |
| Output stride | 8 | 16 | 8 (원본과 동일) |
| ASPP rates | [1, 12, 24, 36] | [6, 12, 18] | [1, 12, 24, 36] (원본과 동일) |
| ASPP Conv | 일반 Conv2d | DepthwiseSeparable | 일반 Conv2d (원본과 동일) |
| Low-level fusion | decoder에서 사용 | 사용 (304ch) | **미사용** (ASPP 출력만) |
| 최종 출력 | 픽셀별 세그멘테이션 | GAP→FC (분류) | GAP→FC (분류) |
| 파라미터 | 39.4M | 26.7M | 39.4M |
| Trainable (freeze 시) | 전체 | 21.2M (stem+layer1 freeze) | **1,285** (FC만) |

### 사전학습 가중치 로드 상세

```
체크포인트: models/ct_cnn/checkpoints/deeplabv3_drn_ct.pt (467MB)
  ├── epoch, optimizer, best_pred (메타데이터)
  └── state_dict (398개 키)
      ├── backbone.* (378개 중 348개) → self.backbone에 로드 ✅
      ├── aspp.*     (378개 중 30개)  → self.aspp에 로드 ✅
      └── decoder.*  (20개)           → 스킵 (분류 헤드로 대체) ⏭️
```

### 학습 전략 (2단계)

| 단계 | freeze_backbone | lr | 학습 대상 | 목적 |
|------|------|------|------|------|
| 1단계 | `true` | 0.001 | FC(256→5)만 (1,285 params) | 분류 헤드 빠른 수렴 |
| 2단계 (선택) | `false` | 0.00001 | 전체 39.4M params | backbone fine-tuning |

### 구현 주의사항

1. **state_dict 키 호환**: 원본 코드를 최대한 그대로 포팅 (클래스명, 변수명 일치)
2. **SyncBN → BN**: `SynchronizedBatchNorm2d` → `nn.BatchNorm2d` (가중치 형식 동일, 로드 호환)
3. **freeze + eval**: backbone/ASPP freeze 시 반드시 eval 모드 고정 → `train()` 오버라이드
   - 이유: ASPP global_avg_pool 내 BN이 (B, 256, 1, 1) 입력 받음 → batch_size=1에서 BN training 모드 실패
4. **low_level_feat 미사용**: backbone은 `(x, low_level_feat)` 반환하지만, 분류 헤드에서는 ASPP 출력만 사용

---

## 🚀 다음 작업

- [x] D드라이브 학습 모델 파일 가져오기 (심볼릭 링크 완료)
- [x] DeepLabV3+ 사전학습 가중치 활용 분류 모델 구현
- [ ] DeepLabV3+ 학습 실행 (`python models/ct_cnn/train.py --config training/configs/cnn_ct_deeplabv3.yaml`)
- [ ] DRN+ASPP 학습 실행 (`python models/ct_cnn/train.py --config training/configs/cnn_ct_drn_aspp.yaml`)
- [ ] 테스트 및 성능 비교 (Late Fusion v2 F1=0.803 vs DeepLabV3+ vs DRN+ASPP)

---

## 📊 CT CNN 모델 성능 비교

| 모델 | F1 Macro | 비고 |
|------|----------|------|
| Late Fusion v2 | **0.803** | 최고 성능 (이미지 + 메타데이터) |
| HDCNN | 0.68 | 계층적 분류 |
| CBAM 768 | 0.66 | Attention 기반 |
| ConvNeXt Tiny | 0.64 | 순수 이미지 |
| EfficientNet B0 | 0.60 | 순수 이미지 |
| ResNet18 Unified | 0.54 | 기본 베이스라인 |
| **DRN+ASPP** | **미측정** | 다중 스케일 특징 추출 (순수 이미지) |
| **DeepLabV3+** | **미측정** | CT 사전학습 DRN-D-54 + ASPP (순수 이미지) |
