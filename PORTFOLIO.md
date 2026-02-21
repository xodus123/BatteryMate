# 배터리 검사 AI 시스템 (Battery Inspection AI System)

## 프로젝트 개요

### 목적
리튬이온 배터리 셀/모듈의 내부 결함(CT 이미지) 및 외부 결함(RGB 이미지)을 자동으로 검출하는 AI 기반 품질 검사 시스템 개발

### 핵심 특징
- **멀티모달 분석**: CT 이미지(내부 결함)와 RGB 이미지(외부 결함) 동시 분석
- **3-Way 검사 시스템**: CNN + AutoEncoder + VLM/VLG 다중 모델 검증
- **실시간 웹 인터페이스**: Streamlit 기반 대시보드로 즉시 결과 확인
- **설명 가능한 AI**: Grad-CAM, Error Map, Grounding 시각화 제공

### 기술 스택
| 분류 | 기술 |
|------|------|
| Deep Learning | PyTorch, TorchVision, timm, Transformers |
| CNN 모델 | ResNet18, ConvNeXt-Tiny, EfficientNet-B0/B4, DRN-D-54, CBAM |
| AE 모델 | ConvAutoEncoder (RGB/CT) |
| VLM 모델 | Qwen3-VL (2B/4B/8B/32B), Gemini 2.5 Pro |
| VLG 모델 | GroundingDINO (Swin-T), YOLO-World (YOLOv8s) |
| 외부 API | Google Gemini 2.5 Pro |
| 시각화 | Grad-CAM, TensorBoard, Matplotlib |
| 웹 프레임워크 | Streamlit |
| 언어 | Python 3.10 |

### 사용 모델 카탈로그

#### CNN + AE (통계적 인지 - 수치적 이상 탐지)

> CT 내부 결함 분류 + RGB 외부 이상 탐지, 논리 기반(AND/OR) 결합

| 구분 | 모델 | Backbone | 특징 | 상태 |
|------|------|----------|------|------|
| **CT CNN** | Late Fusion v2 | ResNet18 + MLP | 이미지 + 메타데이터(축/슬라이스) late concat | **★ 최고 F1=0.803** |
| CT CNN | DRN+ASPP | ResNet50 (dilated) + ASPP | 다중스케일 특징 추출, Depthwise Separable Conv | ★ 순수 이미지 최고 F1=0.794 |
| CT CNN | Metadata v3 | ResNet18 + MLP | 메타데이터 early fusion | ★ ROC-AUC 최고 0.965 |
| CT CNN | CBAM (768) | ResNet18 + CBAM | 채널/공간 어텐션, 768px 입력 | F1=0.862 (x축 포함) |
| CT CNN | ConvNeXt-Tiny | ConvNeXt-Tiny (timm) | 최신 CNN, ImageNet pretrained | F1=0.571 (no_x) |
| CT CNN | EfficientNet-B0 | EfficientNet-B0 (timm) | Compound Scaling | 이전 split (무효) |
| CT CNN | EfficientNet-B4 | EfficientNet-B4 (timm) | 더 큰 Compound Scaling | F1=0.679 (no_x) |
| CT CNN | DeepLabV3+ (freeze) | DRN-D-54 + ASPP (사전학습) | 세그멘테이션 backbone 전이 | ❌ 학습 실패 (F1=0.302) |
| CT CNN | HD-CNN v2 | ResNet18 (Coarse/Fine) | 계층적 Normal→Defect→세부 분류 | ❌ 실패 (F1=0.337) |
| CT CNN | Image-Only | ResNet18 | 메타데이터 없이 순수 이미지만 | 기준선 F1=0.792 |
| CT CNN | ResNet18 기본 | ResNet18 | ImageNet pretrained, 첫 모델 | 초기 F1=0.788 |
| CT CNN | ResNet18+CBAM (초기) | ResNet18 + CBAM | 초기 어텐션 실험 | ❌ 성능 하락 |
| CT CNN | Hierarchical | ResNet18 | 계층 분류 (cell/module → 세부) | ❌ 실패 (F1=0.130) |
| **RGB AE** | ConvAutoEncoder | Conv [3,64,128,256,512] | 정상 데이터만 학습, MSE 재구성 오류 | **★ ROC-AUC=0.909** |
| CT AE | ConvAutoEncoder | Conv [1,64,128,256,512] | CT용 이상 탐지 | ❌ 성능 저조 (0.653) |

#### VLM (추론적 인지 - 문맥적 원인 분석)

> CT/RGB 이미지를 보고 결함 여부, 원인, 심각도를 자연어로 추론

| 모델 | 크기 | 실행 환경 | 속도 | zero-shot 분류 | 상태 |
|------|------|-----------|------|----------------|------|
| **Gemini 2.5 Pro** | - | Google API (클라우드) | ~5초 | 미평가 | **★ 권장** (자연어 분석 + Grounding) |
| Qwen3-VL | 32B | 로컬 GPU (고사양) | ~300초+ | 미평가 | 대규모 평가 필요 |
| Qwen3-VL | 8B | 로컬 GPU | ~150초 | 미평가 | GPU 12GB 제약 |
| Qwen3-VL | 4B | 로컬 GPU | ~50초 | 미평가 | - |
| Qwen3-VL | 2B | 로컬 GPU | ~2초/장 | ❌ CT/RGB 모두 부적합 | zero-shot 분류 한계 확정 |
| ~~Qwen2-VL~~ | 2B | 로컬 GPU | ~25초 | - | Qwen3-VL로 교체 |

> **Qwen3-VL 2B zero-shot 분류 평가 결과**: CT 5클래스 500샘플 및 RGB 3클래스 평가 모두 부적합 확정. VLM은 정량적 분류가 아닌 **자연어 기반 결함 해석/소견서 생성** 용도로 활용.

#### VLG (언어적 인지 - 지시적 위치 특정)

> 자연어 프롬프트("porosity", "void" 등)를 시각적 좌표(BBox)와 매칭

| 모델 | Backbone | 크기 | 탐지 성능 | 속도 | 상태 |
|------|----------|------|----------|------|------|
| **GroundingDINO** | Swin-T + BERT | 694MB | 미세 결함 탐지 우수 | ~1.5초 | **★ 채택** |
| YOLO-World | YOLOv8s + CLIP | 91MB | ❌ 배터리 결함 탐지 실패 | ~0.3초 | 비교 실험용만 유지 |

---

## 시스템 아키텍처

```
[배터리 이미지 입력: CT + RGB]
           ↓
┌──────────────────────────────────────────────────┐
│  System 1: CNN+AE+Grad-CAM 통합 검사                │
│  ┌────────────────┬──────────────────┐          │
│  │  CT CNN        │  RGB AutoEncoder │          │
│  │  (ResNet18)    │  (CAE)           │          │
│  │  5클래스 분류  │  이상 탐지       │          │
│  └────────────────┴──────────────────┘          │
│           ↓                ↓                    │
│  ┌──────────────────────────────────┐           │
│  │ 논리적 조합 (AND/OR)             │           │
│  │ → 정상/내부불량/외부불량/복합불량 │           │
│  └──────────────────────────────────┘           │
│  + Grad-CAM 히트맵                              │
└──────────────────────────────────────────────────┘
                    VS (비교)
┌──────────────────────────────────────────────────┐
│  System 2: VLM (Gemini 2.5 Pro / Qwen3-VL)      │
│  → Zero-shot 판정 + 불량 원인 설명 + 위치 Grounding │
└──────────────────────────────────────────────────┘
                    VS
┌──────────────────────────────────────────────────┐
│  System 3: VLG (GroundingDINO / YOLO-World)     │
│  → 불량 영역 BBox 검출                          │
└──────────────────────────────────────────────────┘
           ↓
┌──────────────────────────────────────────────────┐
│  Web UI: 3개 시스템 결과 비교 시각화            │
└──────────────────────────────────────────────────┘
```

### 검사 흐름
1. **이미지 업로드**: CT/RGB 이미지 입력
2. **병렬 추론**: 3개 모델이 동시에 분석 수행
3. **결과 비교**: 각 모델의 독립적인 판정 결과를 나란히 표시
4. **시각화**: Grad-CAM, Error Map, Bounding Box 등 근거 제시

### 결과 표시 방식

3개 모델이 각각 독립적으로 판정한 결과를 나란히 표시합니다.

| 모델 | 역할 | 판정 방식 |
|------|------|----------|
| **통합 검사기** | 정량적 분류 | CT+RGB 논리적 결합 (내부/외부/복합불량) |
| **VLM** | 정성적 분석 | 자연어 기반 결함 설명 |
| **VLG** | 위치 검출 | Bounding Box 시각화 |

> **통합 검사기**: CT CNN과 RGB AE 결과를 논리적으로 결합(AND/OR)하여 최종 판정

---

## 데이터셋

### 클래스 구성 (5클래스 통일)
| 클래스 | 설명 | 유형 |
|--------|------|------|
| cell_normal | 셀 정상 | 정상 |
| cell_porosity | 셀 기공 결함 | 불량 |
| module_normal | 모듈 정상 | 정상 |
| module_porosity | 모듈 기공 결함 | 불량 |
| module_resin_overflow | 모듈 레진 오버플로우 | 불량 |

### 데이터 분할

#### CT CNN (5클래스 분류)
| Split | 샘플 수 | 비율 |
|-------|---------|------|
| Train | ~70% | 학습용 |
| Validation | ~15% | 검증용 |
| Test | ~15% | 평가용 |

#### RGB AutoEncoder (이상 탐지)
| Split | Normal | Defect | 합계 | 용도 |
|-------|--------|--------|------|------|
| Train | 5,610 | 0 | 5,610 | 정상 패턴 학습 |
| Validation | 1,087 | 25,817 | 26,904 | Threshold 최적화 |
| Test | 1,505 | 25,244 | 26,749 | 최종 평가 |

*Battery ID 기반 분할로 데이터 누수 방지*
*⚠️ Test 데이터 불균형: 불량이 94.4%로 압도적*

---

## 모델 상세

### 1. CT CNN (내부 결함 분류)

#### 아키텍처 실험 요약

다양한 모델 아키텍처를 실험하여 최적 구성을 탐색했습니다.

| 아키텍처 | Backbone | 특징 | 채택 |
|----------|----------|------|------|
| **Late Fusion v2** | ResNet18 | 이미지 + 메타데이터 결합 | **★ 최고 성능** |
| **DRN+ASPP** | DRN + ASPP 모듈 | 다중스케일 특징 추출 | ★ 순수 이미지 최고 |
| **Metadata v3** | ResNet18 + MLP | 메타데이터(축, 슬라이스) 활용 | ★ ROC-AUC 최고 |
| CBAM (768) | ResNet18 + CBAM | 채널/공간 어텐션, 768px | O |
| ConvNeXt-Tiny | ConvNeXt-Tiny | 최신 CNN 아키텍처 | △ |
| EfficientNet-B0/B4 | EfficientNet | 효율적 스케일링 | △ |
| HD-CNN | ResNet18 (Coarse/Fine) | 계층적 분류 | ❌ 실패 |
| Image-Only | ResNet18 | 순수 이미지만 | △ 기준선 |
| ResNet18 기본 | ResNet18 | ImageNet Pretrained | △ 초기 모델 |
| ResNet18+CBAM | ResNet18 + CBAM | 어텐션 추가 | ❌ 성능 하락 |
| Hierarchical | ResNet18 | 계층 분류 | ❌ 실패 |

#### 현재 최고 성능 모델: Late Fusion v2

```
┌─────────────────────────────────────────────────────────────┐
│                  Late Fusion v2 아키텍처                     │
│                                                             │
│  ┌─────────────────────────┐   ┌─────────────────────────┐ │
│  │    CT 이미지 (512×512)   │   │  메타데이터 (raw 입력)   │ │
│  └────────────┬────────────┘   │  • battery_type (0/1)   │ │
│               ↓                │  • axis (x=0 / y=1)     │ │
│  ┌─────────────────────────┐   └────────────┬────────────┘ │
│  │       ResNet18           │                │              │
│  │   (ImageNet Pretrained)  │                │              │
│  │                          │                │              │
│  │  conv1 → bn → relu      │                │              │
│  │  maxpool                 │                │              │
│  │  layer1 (64ch)           │                │              │
│  │  layer2 (128ch)          │                │              │
│  │  layer3 (256ch)          │                │              │
│  │  layer4 (512ch)          │                │              │
│  │  AdaptiveAvgPool → 512-d │                │              │
│  └────────────┬────────────┘                │              │
│               │                  ┌──────────┘              │
│               │     512-d        │     2-d                  │
│               └────────┬─────────┘                          │
│                        ↓                                    │
│               ┌─────────────────┐                           │
│               │   Concat (514-d)│  ← 인코더 없이 raw concat │
│               └────────┬────────┘                           │
│                        ↓                                    │
│               ┌─────────────────┐                           │
│               │   Classifier     │                           │
│               │  FC(514→256)     │                           │
│               │  BN → ReLU       │                           │
│               │  Dropout(0.5)    │                           │
│               │  FC(256→128)     │                           │
│               │  BN → ReLU       │                           │
│               │  Dropout(0.25)   │                           │
│               │  FC(128→5)       │                           │
│               └────────┬────────┘                           │
│                        ↓                                    │
│          ┌──────────────────────────┐                       │
│          │  5클래스 출력 (Softmax)    │                       │
│          │  cell_normal              │                       │
│          │  cell_porosity            │                       │
│          │  module_normal            │                       │
│          │  module_porosity          │                       │
│          │  module_resin_overflow    │                       │
│          └──────────────────────────┘                       │
└─────────────────────────────────────────────────────────────┘

핵심 설계:
• Early Fusion(v1)은 메타데이터를 MLP 인코더로 학습 → 과적합 유발
• Late Fusion(v2)은 메타데이터를 raw 2차원으로 직접 concat → 안정적 학습
• 메타데이터가 "힌트" 역할만 하고 이미지 특징에 의존 → 일반화 성능 향상
```

#### 학습 설정
```yaml
model:
  backbone: ResNet18 (ImageNet Pretrained)
  dropout: 0.5
  num_classes: 5

data:
  image_size: 512           # 원본 4000→512 직접 리사이즈
  batch_size: 32
  num_workers: 4

optimizer:
  name: AdamW
  learning_rate: 0.0001
  weight_decay: 0.03

scheduler:
  name: CosineAnnealingWarmRestarts
  T_0: 10
  T_mult: 2
  eta_min: 1e-6

loss:
  name: Focal Loss          # 클래스 불균형 대응
  gamma: 2.0                # 어려운 샘플에 집중
  label_smoothing: 0.1      # 과신 방지
  class_weights: [1.5, 1.2, 0.8, 1.0, 8.0]

class_balancing:
  method: WeightedRandomSampler

early_stopping:
  monitor: val_f1_macro
  patience: 7
  min_delta: 0.001

training:
  epochs: 50
  amp: true
  gradient_clip: 1.0
```

#### 성능 (현재 split 기준, 35,529 테스트 샘플)

| 순위 | 모델 | Test F1 | Accuracy | ROC-AUC | 비고 |
|------|------|---------|----------|---------|------|
| **1** | **Late Fusion v2** | **0.803** | **80.3%** | **0.944** | **메타데이터+이미지, 가장 안정적** |
| 2 | DRN+ASPP | 0.794 | 78.2% | 0.943 | 다중스케일, 순수 이미지 최고 |
| 3 | Metadata v3 | 0.791 | 78.0% | 0.965 | ROC-AUC 최고 |
| 4 | CBAM (768) | 0.862 | 86.3% | 0.968 | x축 포함 학습 |
| 5 | EfficientNet-B4 no_x | 0.679 | 66.7% | 0.912 | no_x 학습 |
| 6 | ConvNeXt no_x | 0.571 | 64.6% | 0.891 | no_x 학습 |
| 7 | HD-CNN v2 | 0.337 | 38.5% | - | ❌ 실패 |

#### 클래스별 F1 Score (현재 split 기준)

| 모델 | cell_normal | cell_porosity | module_normal | module_porosity | module_resin |
|------|-------------|---------------|---------------|-----------------|--------------|
| **Late Fusion v2** | **0.796** | **0.488** | **0.883** | **0.846** | **1.000** |
| DRN+ASPP | 0.667 | 0.492 | 0.917 | 0.897 | 1.000 |
| Metadata v3 | 0.720 | 0.481 | 0.889 | 0.866 | 1.000 |
| CBAM (768) | 0.860 | 0.645 | 0.912 | 0.895 | 1.000 |

#### 이진 분류 성능 (정상 vs 불량)

| 모델 | 불량 Recall | 불량 Precision | 정상 Recall | Accuracy |
|------|-------------|----------------|-------------|----------|
| **Late Fusion v2** | 72.1% | 75.9% | 85.5% | 80.3% |
| DRN+ASPP | 83.0% | 67.9% | 75.1% | 78.2% |
| Metadata v3 | 79.2% | 68.9% | 77.3% | 78.0% |
| CBAM (768) | 81.9% | 82.8% | 89.2% | 86.4% |

#### 클래스 불균형 문제 및 해결

**문제 분석:**
```
클래스 분포 (Train 기준):
├── cell_normal:           39,343 (28.4%)
├── cell_porosity:         12,755 (9.2%)   ← 희소
├── module_normal:         39,572 (28.6%)
├── module_porosity:       45,165 (32.7%)
└── module_resin_overflow:  1,481 (1.1%)   ← 매우 희소

문제점:
- resin_overflow가 전체의 1.1%로 극심한 불균형
- cell_porosity도 9.2%로 상대적 희소
- 다수 클래스(normal, module_porosity)가 학습 지배
```

**해결 방법 1: Focal Loss (CrossEntropy 대체)**

| 손실 함수 | 수식 | 특징 |
|----------|------|------|
| CrossEntropy | CE = -log(p_t) | 모든 샘플 동등 취급 |
| **Focal Loss** | FL = -(1-p_t)^γ × log(p_t) | 어려운 샘플에 집중 |

```
Focal Loss 효과:
- γ(gamma) = 3.0 설정
- 쉬운 샘플 (p_t ≈ 1.0): 가중치 ↓↓ (거의 0)
- 어려운 샘플 (p_t ≈ 0.3): 가중치 ↑ (집중 학습)
- 결과: 희소 클래스/혼동 샘플에 더 집중
```

**해결 방법 2: Class Weights**
```python
class_weights = [1.0, 4.0, 1.0, 0.9, 25.0]
#                 ↑    ↑    ↑    ↑    ↑
#              cell  cell  mod  mod  resin
#              norm  poro  norm poro overflow
```
- resin_overflow: 25배 가중치 (1.1% → 실질적으로 27.5% 효과)
- cell_porosity: 4배 가중치 (9.2% → 실질적으로 36.8% 효과)

**해결 방법 3: WeightedRandomSampler**
```
동작 원리:
- 각 샘플에 클래스 빈도의 역수를 가중치로 부여
- 희소 클래스 샘플이 더 자주 선택됨
- 한 epoch 내에서 클래스 분포가 균등해짐

효과:
- resin_overflow: 1.1% → ~20% (약 18배 증가)
- cell_porosity: 9.2% → ~20% (약 2배 증가)
```

**해결 방법 4: Label Smoothing**
```
기존: [0, 0, 1, 0, 0] (정답 클래스만 1.0)
적용: [0.025, 0.025, 0.9, 0.025, 0.025] (정답 0.9, 나머지 균등)

효과:
- 모델의 과신(overconfidence) 방지
- 일반화 성능 향상
- 희소 클래스 예측 시 안정성 증가
```

**종합 효과:**
| 방법 | 역할 |
|------|------|
| Focal Loss | 어려운 샘플에 집중 |
| Class Weights | 손실 함수에서 희소 클래스 중요도 ↑ |
| WeightedRandomSampler | 데이터 레벨에서 균형 맞춤 |
| Label Smoothing | 일반화 향상, 과신 방지 |

#### 데이터 전처리 및 증강

**전처리 파이프라인:**
```
원본 이미지 (4000×4000)
       ↓
   Resize (1024×1024)     ← 사전 전처리 (PNG 저장)
       ↓
   Data Augmentation      ← 학습 시 동적 적용
       ↓
   ToTensor + Normalize
       ↓
   모델 입력
```

**데이터 증강 (Config 기반 동적 적용):**
```yaml
augmentation:
  train:
    - RandomHorizontalFlip: {p: 0.5}
    - RandomVerticalFlip: {p: 0.5}
    - RandomRotation: {degrees: 30}
    - ColorJitter: {brightness: 0.3, contrast: 0.3}
    - RandomAffine: {translate: [0.1, 0.1], scale: [0.9, 1.1]}
    - GaussianBlur: {kernel_size: 3, p: 0.3}
  val: []  # Validation/Test는 증강 없음
```

**증강 효과:**
| 증강 기법 | 목적 |
|----------|------|
| Flip (H/V) | 방향 불변성 학습 |
| Rotation | 회전 불변성 학습 |
| ColorJitter | 조명 변화 대응 |
| RandomAffine | 위치/크기 변화 대응 |
| GaussianBlur | 블러 내성, 노이즈 대응 |

#### 시각화
- **Grad-CAM**: 결함 위치 히트맵 생성
- Layer4 특징맵 기반 활성화 영역 시각화

#### ResNet18 Backbone 선택 이유

다양한 아키텍처 실험 결과, ResNet18 기반 모델이 일관되게 최고 성능을 달성했습니다.

**1. 데이터셋 규모와 모델 복잡도의 균형**
```
배터리 CT 데이터셋: ~14만 장 (92개 배터리)
→ 대형 모델(ConvNeXt, EfficientNet-B4) 사용 시 심각한 과적합
→ ResNet18의 11M 파라미터가 적절한 복잡도
```

**2. 실험적 검증 결과 (현재 split 기준)**

| 모델 | Backbone | Test F1 | 과적합 정도 | 비고 |
|------|----------|---------|-----------|------|
| **Late Fusion v2** | ResNet18 | **0.803** | 경미 | ★ 최고 |
| DRN+ASPP | DRN | 0.794 | 심각 | 다중스케일 |
| Metadata v3 | ResNet18+MLP | 0.791 | 심각 | 메타데이터 |
| CBAM (768) | ResNet18+CBAM | 0.862 | - | x축 포함 |
| ConvNeXt no_x | ConvNeXt-Tiny | 0.571 | 심각 | 성능 저조 |
| EfficientNet-B4 no_x | EfficientNet | 0.679 | 심각 | 성능 저조 |
| HD-CNN v2 | ResNet18 (계층) | 0.337 | 극심 | 완전 실패 |

**3. 산업 환경 배포 요구사항**
| 요구사항 | ResNet18 | EfficientNet-B7 | ViT-Large |
|----------|----------|-----------------|-----------|
| 모델 크기 | **46MB** | 256MB | 1.2GB |
| 추론 속도 | **~15ms** | ~45ms | ~120ms |
| GPU 메모리 | **2GB** | 6GB | 12GB+ |
| Edge 배포 | ✅ 가능 | △ 제한적 | ❌ 어려움 |

**4. Transfer Learning 효과**
```
✅ ImageNet-1K pretrained 가중치 활용
✅ 배터리 CT 이미지의 "질감, 형태, 구조" 특징 → ImageNet 특징 재사용 가능
✅ 적은 데이터로도 빠른 수렴 및 높은 성능 달성
```

**5. 핵심 교훈**
```
✅ 다양한 아키텍처 실험 → ResNet18 기반 Late Fusion이 가장 안정적
✅ 대형 모델(ConvNeXt, EfficientNet)은 과적합이 심각
✅ 메타데이터 결합이 복잡한 아키텍처보다 효과적
✅ 92개 배터리라는 데이터 한계에서 단순 모델이 유리
```

> 💡 **교훈**: 최신 SOTA 모델이 항상 최선은 아님. 데이터 규모, 배포 환경, 유지보수성을 고려하여 **적정 기술** 선택이 중요.

---

### 1-1. CBAM 적용 실험 및 비교 분석

#### CBAM (Convolutional Block Attention Module) 개요
CBAM은 채널(Channel)과 공간(Spatial) 두 가지 어텐션 메커니즘을 결합한 모듈로, 작은 결함(기공, 미세 균열) 탐지 성능 향상을 목표로 실험하였습니다.

```
┌─────────────┐    ┌──────────────────┐    ┌──────────────────┐
│ Input (F)   │ → │ Channel Attention │ → │ Spatial Attention │ → Output (F')
└─────────────┘    └──────────────────┘    └──────────────────┘
                         ↓                        ↓
                   Avg/Max Pool              Avg/Max along
                   + MLP + Sigmoid           channel + Conv
```

#### 아키텍처: ResNet18 + CBAM
```python
ResNet18CBAM:
├── conv1, bn1, relu, maxpool
├── layer1 (64ch)
├── layer2 (128ch)
├── layer3 (256ch) → CBAM(256, reduction=16)
├── layer4 (512ch) → CBAM(512, reduction=16)
├── avgpool
├── dropout(0.3)
└── fc(512 → 5)
```

#### 성능 비교
| 모델 | Val Accuracy | Test Accuracy | Test F1 Macro | 모델 크기 |
|------|--------------|---------------|---------------|-----------|
| **ResNet18 (기본)** | 83.1% | **77.4%** | **78.8%** | 46MB |
| ResNet18 + CBAM | 80.1% | - | - | 135MB |

#### 발견된 문제점

**1. 성능 하락**
- CBAM 추가 시 오히려 성능이 소폭 하락
- 원인 분석:
  - 데이터셋 크기 대비 모델 복잡도 증가 → 과적합 가능성
  - 배터리 CT 이미지 특성상 전역 특징이 중요하며, 국소적 어텐션이 오히려 방해
  - 충분한 하이퍼파라미터 튜닝 미진행

**2. State Dict 키 불일치 문제**
```python
# ResNet18 기본 모델 키
"model.conv1.weight", "model.bn1.weight", ...

# ResNet18+CBAM 모델 키
"conv1.weight", "bn1.weight", "cbam3.channel_mlp.0.weight", ...
```
- CBAM 모델은 `model.` prefix 없이 직접 레이어 정의
- 체크포인트 로드 시 키 불일치로 에러 발생
- **해결**: CBAM 체크포인트를 별도 폴더로 분리 관리

**3. 모델 크기 증가**
- 기본 ResNet18: 46MB
- ResNet18+CBAM: 135MB (약 3배 증가)
- 추론 속도 저하 우려

#### 결론 및 교훈
| 항목 | 내용 |
|------|------|
| 채택 여부 | ❌ 미채택 (기본 ResNet18 사용) |
| 주요 원인 | 성능 향상 없음, 복잡도만 증가 |
| 교훈 | Attention이 항상 성능 향상을 보장하지 않음 |
| 대안 | 데이터 증강, 다중 모델 등 다른 방법 적용 |

#### 체크포인트 관리
```
models/ct_cnn/checkpoints/
├── ct_unified_best_20260105_140553.pt  # 기본 ResNet18 (사용 중)
├── ct_unified_last_20260105_140553.pt
└── cbam/                                # CBAM 모델 (미사용)
    ├── ct_unified_best_20260106_213526.pt
    └── ct_unified_last_20260106_213526.pt
```

---

### 2. RGB AutoEncoder (외부 결함 탐지)

#### 아키텍처
```
Encoder: [3, 64, 128, 256, 512] → Latent (32x32x512)
Decoder: [512, 256, 128, 64, 3] → Reconstructed Image

입력: 512x512x3 RGB (또는 1024x1024 전처리 이미지)
Latent Dimension: 1024
```

#### 학습 방식
- **정상 데이터 학습**: 정상 패턴만 학습하여 이상 탐지
- **손실 함수**: MSE (재구성 오류)
- **Threshold 최적화**: ROC 기반 Youden's J 통계량 최대화

#### 판정 로직
```python
# 정상 학습 모델
if anomaly_score > threshold:
    prediction = "defect"  # 정상 패턴과 다름
else:
    prediction = "normal"  # 정상 패턴과 유사
```

#### 성능
| 지표 | Validation | Test |
|------|------------|------|
| ROC-AUC | 0.9999 | 0.9095 |

> **참고**: 데이터 누수 수정 후 (배터리 ID 기준 split) Validation ROC-AUC가 0.9999 → 신버전에서도 유사 수준, Test ROC-AUC = 0.9095. Threshold 기반 Accuracy/F1은 재평가 필요 (테스트 스크립트 NaN threshold 버그 수정 완료, 재실행 필요).

#### Threshold 설정 (신버전 split 기준)
```json
{
  "normal_mean": 1.2537,
  "defect_mean": 1.9068,
  "threshold": 1.3878,
  "note": "ROC optimal threshold (배터리 ID 분리 후)"
}
```

---

### 2-1. AutoEncoder 학습 데이터 비교 실험

AutoEncoder 기반 이상 탐지에서 **어떤 데이터로 학습하느냐**에 따라 판정 로직과 성능이 크게 달라집니다.

#### 두 가지 접근 방식

| 구분 | 불량 데이터 학습 | 정상 데이터 학습 |
|------|------------------|------------------|
| 학습 데이터 | 결함 이미지만 사용 | 정상 이미지만 사용 |
| 학습 목표 | 불량 패턴 재구성 학습 | 정상 패턴 재구성 학습 |
| 불량 입력 시 | 재구성 오류 **낮음** | 재구성 오류 **높음** |
| 정상 입력 시 | 재구성 오류 **높음** | 재구성 오류 **낮음** |
| 판정 조건 | score < threshold → 불량 | score > threshold → 불량 |

#### 실험 1: 불량 데이터 학습 (초기 접근)

```python
# 불량 데이터로 학습
Train: 결함 이미지 2,592장
Val/Test: 정상 + 결함 혼합

# 판정 로직
is_defect = anomaly_score < threshold  # 불량 패턴과 유사하면 불량
```

**문제점:**
- Threshold 계산 방식 오류 (학습 데이터로 계산 → 과적합)
- k-sigma 방식의 threshold가 너무 높게 설정됨 (2.996)
- 모든 샘플이 정상으로 분류되는 현상 발생

| 지표 | 값 |
|------|-----|
| Validation ROC-AUC | 0.9644 |
| Test Accuracy | 50.0% (사실상 랜덤) |
| 문제 | Threshold 2.996으로 전부 정상 판정 |

#### 실험 2: 정상 데이터 학습 (최종 채택)

```python
# 정상 데이터로 학습
Train: 정상 이미지 5,610장 (결함 이미지 0장)
Val: 정상 1,087장 + 결함 25,817장 (Threshold 계산용)
Test: 정상 1,505장 + 결함 25,244장

# 판정 로직
is_defect = anomaly_score > threshold  # 정상 패턴과 다르면 불량
```

**개선점:**
- Validation 데이터로 ROC 기반 threshold 최적화
- Youden's J 통계량 (TPR - FPR) 최대화 지점 선택
- Battery ID 기반 분할로 데이터 누수 방지

| 지표 | 값 |
|------|-----|
| Validation ROC-AUC | **0.9999** |
| Test ROC-AUC | 0.9091 |
| Test Accuracy | **98.85%** |
| Test F1-Score | **99.40%** |

#### 성능 비교 요약

| 항목 | 불량 데이터 학습 | 정상 데이터 학습 |
|------|------------------|------------------|
| Validation ROC-AUC | 0.9644 | **0.9999** |
| Test Accuracy | 50.0% | **98.85%** |
| Threshold 방식 | k-sigma (학습 데이터) | ROC 최적화 (검증 데이터) |
| 채택 여부 | ❌ | ✅ |

#### Threshold 계산 방식 비교

**Before: k-sigma 방식 (문제 있음)**
```python
# 학습 데이터의 score 분포 기반
threshold = mean + k * std  # k=3
# 문제: 학습 데이터가 불량이므로 threshold가 너무 높음
```

**After: ROC 최적화 방식 (개선)**
```python
# Validation 데이터 (정상+불량 혼합) 기반
fpr, tpr, thresholds = roc_curve(labels, scores)
j_scores = tpr - fpr  # Youden's J statistic
optimal_idx = np.argmax(j_scores)
threshold = thresholds[optimal_idx]
```

#### Score 분포 비교 (정상 학습 모델)

```
Score Distribution:
├── 정상 이미지: mean=0.747, std=0.150 (낮은 재구성 오류)
├── 불량 이미지: mean=2.075, std=0.164 (높은 재구성 오류)
└── Threshold: 1.155 (두 분포 사이 최적점)

     정상          Threshold        불량
      ▼               ▼              ▼
  ████████           │         ████████
  ████████           │         ████████
  ████████           │         ████████
──────────────────────────────────────────→ Score
  0.5    0.75   1.0   1.155  1.5   2.0   2.5
```

#### 결론 및 교훈

| 항목 | 내용 |
|------|------|
| 최종 선택 | ✅ 정상 데이터 학습 |
| 핵심 이유 | 분리도 높은 score 분포, ROC-AUC 0.9999 달성 |
| Threshold | 검증 데이터 기반 ROC 최적화 필수 |
| 데이터 분할 | Battery ID 기반으로 누수 방지 |
| 교훈 | AE 이상탐지는 "정상 학습"이 일반적으로 더 효과적 |

#### 데이터 분할 스크립트
```bash
# 정상 데이터 학습용 분할 생성
python scripts/create_rgb_normal_splits.py

# 결과:
# - Train: 5,610 정상 이미지
# - Val: 1,087 정상 + 25,817 불량
# - Test: 1,505 정상 + 25,244 불량
# - 기존 분할 백업: backup_defect_training/
```

---

### 3. 통합 검사기 (CNN + AE 논리 결합)

> **앙상블(Ensemble)이 아닌 논리 기반(Logic-based) 결합**: 두 모델의 출력을 가중 평균하거나 투표하는 앙상블 방식이 아니라, CT(내부)와 RGB(외부)의 이진 판정 결과를 **AND/OR 논리 규칙**으로 조합하여 4가지 판정 유형을 도출합니다.

#### 판정 로직
| CT 결과 | RGB 결과 | 최종 판정 |
|---------|----------|-----------|
| 정상 | 정상 | 정상 |
| 불량 | 정상 | 내부불량 |
| 정상 | 불량 | 외부불량 |
| 불량 | 불량 | 복합불량 |

#### 신뢰도 계산
```python
# 정상: 두 모델의 정상 확신도 평균
confidence = (ct_normal_prob + rgb_confidence) / 2

# 내부불량: CT 신뢰도
confidence = ct_confidence

# 외부불량: RGB 신뢰도
confidence = rgb_confidence

# 복합불량: 두 모델의 불량 확신도 평균
confidence = (ct_confidence + rgb_confidence) / 2
```

---

### 4. VLM (Vision-Language Model)

#### 모델 선정 기준

VLM의 역할은 **결함 위치 시각화(Grounding)** 및 **자연어 기반 소견 생성**이며, 로컬 모델과 API 모델을 이원화하여 운용합니다.

| 선정 기준 | 중요도 | 설명 |
|----------|--------|------|
| **결함 위치 출력 (Grounding)** | ★★★ | 바운딩 박스 좌표를 네이티브로 출력 가능 여부. 검사 시스템에서 불량 위치 시각화 필수 |
| **Zero-shot 검출 성능** | ★★★ | 배터리 도메인 학습 없이 즉시 사용 가능한 성능. RF100-VL 벤치마크 기준 |
| **한국어 지원** | ★★☆ | 프롬프트 및 소견 출력이 한국어로 가능한지 |
| **비용 효율성** | ★★☆ | 이미지당 API 호출 비용. 대량 검사 시 운용 비용 |
| **로컬 실행 가능 여부** | ★★☆ | 데이터 보안 및 오프라인 환경 대응 |

#### 후보 모델 비교

| 모델 | 바운딩 박스 | Zero-shot mAP (RF100-VL) | 한국어 | 이미지당 비용 | 로컬 실행 |
|------|-----------|--------------------------|--------|-------------|----------|
| **Gemini 2.5 Pro** | O (네이티브) | **13.3 (VLM 1위)** | 우수 | ~$0.01 | X (API 전용) |
| Gemini 2.5 Flash | O (네이티브) | 양호 | 우수 | ~$0.002 | X (API 전용) |
| **Qwen3-VL** | O (절대 픽셀 좌표) | 7.5~9.2 | 최우수 (CJK 특화) | 무료 (로컬) | **O (3B~72B)** |
| GPT-4o | X (부정확) | - | 우수 | ~$0.02 | X |
| Claude Sonnet | X (비일관적) | - | 우수 | ~$0.03 | X |

#### 선정 결과

| 역할 | 선정 모델 | 선정 사유 |
|------|----------|----------|
| **API 모델** | **Gemini 2.5 Pro** | VLM 중 zero-shot 객체 검출 성능 최고(mAP 13.3), 바운딩 박스 네이티브 지원, 합리적 비용 |
| **로컬 모델** | **Qwen3-VL (8B)** | 오픈소스 로컬 GPU 실행, 바운딩 박스 네이티브 지원, 한국어 CJK 최우수, 데이터 외부 유출 없음 |

**탈락 모델:**
- **GPT-4o**: 바운딩 박스 좌표 출력 부정확, 환각(hallucination) 발생 빈번
- **Claude Sonnet/Opus**: 시각적 grounding 능력 부족, 좌표 출력 비일관적
- **Gemini 2.5 Flash**: Pro 대비 검출 정밀도 낮음 (비용 최적화 필요 시 대안)

#### 모델

| 모델 | 실행 환경 | 속도 | zero-shot 분류 | 비고 |
|------|-----------|------|----------------|------|
| **Qwen3-VL-8B** | 로컬 GPU | ~150초 | 미평가 | GPU 12GB 제약 |
| **Qwen3-VL-2B** | 로컬 GPU | ~2초/장 | ❌ CT/RGB 모두 부적합 | 자연어 해석 용도로 활용 |
| **Gemini 2.5 Pro** | Google API | ~5초 | 미평가 | **★ 권장** (자연어 분석 + Grounding) |

> **참고**: Qwen3-VL 2B zero-shot 분류 평가 결과, CT 5클래스(500샘플) 및 RGB 3클래스 모두 분류 부적합 확정. VLM은 정량적 분류 대신 **자연어 기반 결함 해석/소견서 생성 및 결함 위치 Grounding** 용도로 활용합니다.
> 최고 성능 VLM도 산업 결함 검출에서 mAP ~13 수준이므로, VLM은 **1차 스크리닝** 용도로 활용하고 정밀 판정은 CNN 모델이 담당합니다.

#### 프롬프트 설계
```
CT 이미지 분석:
- 기공(porosity), 공극(void), 크랙(crack) 검출
- 결함 위치 및 심각도 설명

RGB 이미지 분석:
- 오염(pollution), 스크래치(scratch), 손상(damage) 검출
- 외관 상태 평가
```

#### 출력
- 결함 여부 (정상/불량)
- 결함 유형 분류
- 자연어 소견서 생성
- 결함 위치 설명

---

### 5. VLG (Vision-Language Grounding)

#### 모델
- **GroundingDINO (Swin-T)**: Open-vocabulary 객체 탐지
- 텍스트 프롬프트 기반 결함 영역 탐지

#### 탐지 프롬프트
```python
CT_PROMPTS = [
    "porosity", "void", "bubble",
    "resin overflow", "defect"
]

RGB_PROMPTS = [
    "pollution", "contamination", "stain",
    "scratch", "damage", "dent"
]
```

#### 출력
- 결함 바운딩 박스 좌표
- 결함 유형 라벨
- 탐지 신뢰도 점수

---

### 5-1. VLG 모델 비교 실험: GroundingDINO vs YOLO-World

Open-vocabulary 객체 탐지 모델 중 배터리 결함 탐지에 최적화된 모델을 선정하기 위해 비교 실험을 진행했습니다.

#### 비교 모델

| 모델 | 아키텍처 | 파라미터 | 특징 |
|------|----------|----------|------|
| **GroundingDINO** | Swin-T + BERT | 172M | 텍스트-이미지 그라운딩 특화 |
| **YOLO-World** | YOLOv8s + CLIP | 28M | 실시간 탐지 최적화 |

#### 성능 비교

| 지표 | GroundingDINO | YOLO-World |
|------|---------------|------------|
| 결함 탐지율 | **높음** | 낮음 |
| 미세 결함 감지 | **우수** | 미흡 |
| 오탐지율 (False Positive) | 낮음 | **높음** |
| 추론 속도 | 느림 (~1.5초) | **빠름** (~0.3초) |
| 모델 크기 | 694MB | **91MB** |

#### 실험 결과 분석

**GroundingDINO 강점:**
```
✅ 텍스트 프롬프트와 이미지 영역 매칭 정확도 높음
✅ "porosity", "void", "resin overflow" 등 세밀한 결함 구분 가능
✅ 작은 결함(기공, 미세 균열)도 안정적으로 탐지
```

**YOLO-World 문제점:**
```
❌ Open-vocabulary 성능이 기대에 미치지 못함
❌ 배터리 도메인 특화 결함 인식률 저조
❌ 일반적인 객체(물체, 텍스처)를 결함으로 오탐지
❌ Confidence threshold 조정으로도 개선 한계
```

#### 실패 원인 분석

| 원인 | 설명 |
|------|------|
| 도메인 갭 | YOLO-World는 일반 객체 탐지에 최적화, 산업 결함에 약함 |
| 프롬프트 민감도 | GroundingDINO가 텍스트 프롬프트 이해도가 높음 |
| 학습 데이터 | YOLO-World의 사전학습 데이터에 산업 결함 이미지 부족 |
| 모델 구조 | Swin Transformer가 미세 특징 추출에 더 효과적 |

#### 구현 및 테스트 환경

```python
# YOLO-World 구현 (models/vlg/inference_yoloworld.py)
from ultralytics import YOLOWorld

class YOLOWorldInference:
    def __init__(self, model_size='s'):
        self.model = YOLOWorld(f'yolov8{model_size}-world.pt')

    def detect(self, image_path, text_prompt, threshold=0.3):
        classes = text_prompt.split(' . ')
        self.model.set_classes(classes)
        results = self.model.predict(image_path, conf=threshold)
        return self._parse_results(results)
```

```python
# 웹앱에서 모델 선택 가능하도록 구현
# webapp/pages/home.py
vlg_options = {
    'groundingdino': '🎯 GroundingDINO (Swin-T)',
    'yoloworld': '🚀 YOLO-World (YOLOv8s)'
}
```

#### 결론 및 교훈

| 항목 | 내용 |
|------|------|
| 최종 선택 | ✅ **GroundingDINO** |
| 선택 이유 | 산업 결함 탐지 정확도가 압도적으로 높음 |
| YOLO-World | ❌ 미채택 (비교 실험용으로만 유지) |
| 교훈 1 | 실시간 속도보다 탐지 정확도가 품질 검사에서 중요 |
| 교훈 2 | Open-vocabulary 모델도 도메인별 성능 차이 존재 |
| 교훈 3 | 모델 선정 시 실제 데이터로 비교 실험 필수 |

#### 참고: 웹앱에서 비교 테스트 가능

```bash
# 웹앱 실행 후 "고급 설정"에서 VLG 모델 선택 가능
streamlit run webapp/app.py --server.port 8501

# 홈페이지 → ⚙️ 고급 설정 → VLG 모델 선택
# - GroundingDINO (기본값, 권장)
# - YOLO-World (비교 테스트용)
```

---

### 5-2. 웹앱 통합 비교 실험: VLM + VLG 모델 조합 테스트

웹앱에서 VLM(Qwen2-VL vs Gemini)과 VLG(GroundingDINO vs YOLO-World) 모델을 자유롭게 선택하여 비교 테스트할 수 있도록 구현했습니다. 동일한 테스트 이미지(RGB 외부 오염 결함)로 4가지 조합을 실험했습니다.

> **참고**: 이 실험은 Qwen2-VL-2B 기준으로 수행되었습니다. 이후 **Qwen3-VL로 업그레이드**되었으며, 별도 대규모 평가에서도 zero-shot 분류 한계가 확인되었습니다 (CT 5클래스 500샘플 Accuracy 20.8%).

#### 테스트 환경

| 항목 | 내용 |
|------|------|
| 테스트 이미지 | 통합 검사 데모 (RGB 외부 오염 결함) |
| Ground Truth | **외부불량** (오염) |
| GPU | CUDA (RTX 시리즈) |
| 테스트 날짜 | 2026-01-08 |

#### 4가지 조합 테스트 결과

| 조합 | VLM 모델 | VLM 결과 | VLM 시간 | VLG 모델 | VLG 결과 | VLG 시간 |
|------|----------|----------|----------|----------|----------|----------|
| 1 | **Gemini 2.0 Flash** | ✅ 외부불량 95% | ~7초 | **GroundingDINO** | ✅ 외부불량 39.9% (1개) | ~1초 |
| 2 | **Qwen2-VL-2B** | ❌ 내부불량 80% | ~25초 | **GroundingDINO** | ✅ 외부불량 39.9% (1개) | ~1초 |
| 3 | **Qwen2-VL-2B** | ❌ 내부불량 80% | ~25초 | **YOLO-World** | ❌ 정상 0% (0개) | ~2초 |
| 4 | **Gemini 2.0 Flash** | ✅ 외부불량 95% | ~5초 | **YOLO-World** | ❌ 정상 0% (0개) | ~2초 |

**통합 검사기 결과**: 모든 테스트에서 일관되게 **외부불량 94.2%** (정확)

#### 상세 로그 분석

**테스트 1: Gemini + GroundingDINO (최적 조합)**
```
[WEBAPP] 15:05:48 - ✅ 통합 검사 추론 완료: 외부불량 (신뢰도: 94.2%)
[WEBAPP] 15:05:56 - ✅ VLM 추론 완료 (Gemini 2.0 Flash): 외부불량 (신뢰도: 95.0%)
[WEBAPP] 15:06:00 - ✅ VLG 추론 완료 (GroundingDINO): 외부불량 - 1개 검출 (최대 신뢰도: 39.9%)

결과: 3개 모델 모두 외부불량 정확 탐지 ✅
```

**테스트 2: Qwen2-VL + GroundingDINO**
```
[WEBAPP] 15:07:37 - ✅ VLM 추론 완료 (Qwen2-VL-2B): 내부불량 (신뢰도: 80.0%)
[WEBAPP] 15:07:38 - ✅ VLG 추론 완료 (GroundingDINO): 외부불량 - 1개 검출 (최대 신뢰도: 39.9%)

결과: VLM이 내부불량으로 오판 ❌ (RGB 이미지인데 CT로 착각)
```

**테스트 3: Qwen2-VL + YOLO-World (최악 조합)**
```
[WEBAPP] 15:09:17 - ✅ VLM 추론 완료 (Qwen2-VL-2B): 내부불량 (신뢰도: 80.0%)
[WEBAPP] 15:09:19 - ✅ VLG 추론 완료 (YOLO-World): 정상 - 0개 검출 (최대 신뢰도: 0.0%)

결과: VLM 오판 + VLG 미탐 ❌❌
```

**테스트 4: Gemini + YOLO-World**
```
[WEBAPP] 15:10:37 - ✅ VLM 추론 완료 (Gemini 2.0 Flash): 외부불량 (신뢰도: 95.0%)
[WEBAPP] 15:10:37 - ✅ VLG 추론 완료 (YOLO-World): 정상 - 0개 검출 (최대 신뢰도: 0.0%)

결과: VLM 정확 ✅, VLG 미탐 ❌
```

#### 모델별 성능 비교

**VLM 비교: Gemini vs Qwen2-VL**

| 항목 | Gemini 2.0 Flash | Qwen2-VL-2B |
|------|------------------|-------------|
| 정확도 | ✅ **정확** (외부불량) | ❌ **오판** (내부불량) |
| 신뢰도 | 95% | 80% |
| 추론 시간 | **~5-7초** (빠름) | ~23-25초 (느림) |
| 결함 유형 | 오염, 손상 (정확) | 없음 (미식별) |
| 실행 환경 | Google API (클라우드) | 로컬 GPU |
| 장점 | 빠른 속도, 높은 정확도 | 오프라인 사용 가능 |
| 단점 | API 비용, 네트워크 필요 | 느린 속도, 오판 가능성 |

**VLG 비교: GroundingDINO vs YOLO-World**

| 항목 | GroundingDINO | YOLO-World |
|------|---------------|------------|
| 탐지 결과 | ✅ **1개 검출** (pollution) | ❌ **0개 검출** (미탐) |
| 신뢰도 | 39.9% | 0% |
| 추론 시간 | ~1초 | ~2초 |
| 결과 | 외부불량 정확 탐지 | 완전 실패 |

#### 조합별 종합 평가

| 순위 | 조합 | VLM | VLG | 정확도 | 추천 |
|------|------|-----|-----|--------|------|
| 🥇 1위 | Gemini + GroundingDINO | ✅ | ✅ | **100%** | ⭐ **권장** |
| 🥈 2위 | Gemini + YOLO-World | ✅ | ❌ | 67% | VLG만 부정확 |
| 🥉 3위 | Qwen2-VL + GroundingDINO | ❌ | ✅ | 67% | VLM만 부정확 |
| 4위 | Qwen2-VL + YOLO-World | ❌ | ❌ | **0%** | ❌ 비권장 |

#### 핵심 발견 및 교훈

**1. 통합 검사기의 안정성**
```
✅ 모든 테스트에서 일관되게 "외부불량 94.2%" 출력
✅ VLM/VLG 조합과 무관하게 항상 정확한 판정
→ 최종 판정은 통합 검사기를 기준으로 해야 함
```

**2. Gemini API의 우수성**
```
✅ Qwen2-VL 대비 정확도 높음 (외부/내부 구분 정확)
✅ 추론 속도 5배 빠름 (25초 → 5초)
✅ 결함 유형까지 정확히 식별 ("오염, 손상")
⚠️ API 비용 및 네트워크 의존성 존재
```

**3. YOLO-World의 한계**
```
❌ 배터리 결함 탐지에 완전 실패 (0개 검출)
❌ Open-vocabulary 성능이 GroundingDINO 대비 현저히 낮음
→ 실무 적용 불가, 비교 실험용으로만 유지
```

**4. 권장 설정**

| 용도 | VLM 모델 | VLG 모델 | 비고 |
|------|----------|----------|------|
| **실무 배포** | Gemini 2.5 Pro | GroundingDINO | 최고 정확도 |
| **오프라인 환경** | Qwen3-VL-8B | GroundingDINO | API 불가 시 |
| **속도 우선** | Gemini 2.5 Pro | GroundingDINO | 총 ~8초 |
| **비교 테스트** | 선택 가능 | 선택 가능 | 웹앱 설정 |

#### 웹앱 설정 방법

```bash
# 웹앱 실행
streamlit run webapp/app.py --server.port 8501

# 홈페이지 → ⚙️ 고급 설정 열기
# VLM 모델 선택:
#   - 🧠 Qwen3-VL (로컬) - GPU 실행, 오프라인 가능
#   - ☁️ Gemini 2.5 Pro (API) - 빠르고 정확, Grounding 지원, 권장
# VLG 모델 선택:
#   - 🎯 GroundingDINO (Swin-T) - 정확도 높음, 권장
#   - 🚀 YOLO-World (YOLOv8s) - 비교 테스트용
```

#### 결론

| 항목 | 결론 |
|------|------|
| 최적 VLM | **Gemini 2.5 Pro** (정확도 + 속도 + Grounding 모두 우수) |
| 최적 VLG | **GroundingDINO** (유일하게 결함 탐지 성공) |
| 최적 조합 | **Gemini + GroundingDINO** (3개 모델 모두 정확) |
| 통합 검사기 역할 | 최종 판정 기준 (VLM/VLG 오판 시 보정) |

---

## 3-Way 모델 비교 분석

본 시스템은 3개의 독립적인 분석 모델을 사용하여 다각도로 결함을 검출합니다. 각 모델의 특성과 한계를 분석하여 상호 보완적인 다중 모델 시스템을 구축했습니다.

### 핵심 설계 철학: "AI가 이미지를 어떻게 이해하는가?"

3개 모델은 각각 **다른 인지 패러다임**으로 CT 이미지를 분석합니다:

| 분석 모델 | 인지 패러다임 | 핵심 역할 (Core Value) | CT 데이터 최적화 전략 |
|-----------|-------------|----------------------|---------------------|
| **CNN + AE (논리 결합)** | 통계적 인지 | 수치적 이상 탐지 | 픽셀 분포의 편차를 계산하여 미세 기공/공극의 존재 여부를 확률로 산출 |
| **VLM (Gemini 2.5 Pro)** | 추론적 인지 | 문맥적 원인 분석 | 이미지 전체의 구조적 관계를 파악하여 결함의 발생 원인(제조 공정 등)을 추론 |
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

> **포트폴리오 강조 포인트**: 단일 AI 모델의 한계를 인식하고, 통계적·추론적·언어적 인지 패러다임을 결합하여 **다각적 검증 시스템**을 구축함

### 모델별 역할 및 특성

| 구분 | 통합 검사 (CNN+AE) | VLM (Gemini 2.5 Pro / Qwen3-VL) | VLG (GroundingDINO) |
|------|-------------------|----------------------|---------------------|
| **인지 패러다임** | 통계적 인지 | 추론적 인지 | 언어적 인지 |
| **결합 방식** | 논리 기반 (AND/OR) | 독립 추론 | 독립 탐지 |
| **역할** | 수치적 이상 탐지 | 문맥적 원인 분석 | 지시적 위치 특정 |
| **출력** | 클래스 확률 | 자연어 소견서 | 바운딩 박스 |
| **강점** | 높은 정확도, 확률 산출 | 원인 추론, 설명 가능성 | 정확한 위치 좌표 |
| **약점** | 원인 설명 불가 | 위치 부정확 | 분류 불가 |

### 상세 분석

#### 🔬 통합 검사기 (CT CNN + RGB AutoEncoder)

**강점:**
```
✅ 정량적 분류 수행 (CT 77.4%, RGB 98.85%)
✅ 내부(CT)/외부(RGB) 결함 구분 명확
✅ 정량적 신뢰도 점수 제공
✅ Grad-CAM으로 관심 영역 시각화
✅ 추론 속도 빠름 (~0.5초)
```

**약점:**
```
❌ 왜 불량인지 자연어 설명 불가
❌ 학습하지 않은 새로운 결함 유형 대응 어려움
❌ 결함의 정확한 바운딩 박스 제공 불가
```

**적합한 용도:** 최종 판정 (Pass/Fail), 자동화 라인 적용

---

#### 🤖 VLM (Gemini 2.5 Pro / Qwen3-VL)

**강점:**
```
✅ 자연어로 결함 설명 생성 (소견서)
✅ Zero-shot: 학습 없이 다양한 결함 인식
✅ 결함의 특성, 심각도, 위치 설명
✅ 사람이 이해하기 쉬운 리포트
✅ 새로운 결함 유형에도 대응 가능
✅ 바운딩 박스 네이티브 지원 (Gemini 2.5 Pro, Qwen3-VL)
```

**약점:**
```
❌ 할루시네이션 (없는 결함을 있다고 할 수 있음)
❌ 추론 속도 느림 (~20초, 로컬 모델 기준)
❌ 정량적 신뢰도 산출 어려움
❌ 산업 결함 검출 zero-shot mAP ~13 수준 (정밀도 한계)
```

**적합한 용도:** AI 소견서 생성, 검사 리포트 작성, 결함 위치 Grounding, 설명 가능한 AI

---

#### 🎯 VLG (GroundingDINO)

**강점:**
```
✅ 정확한 바운딩 박스 좌표 제공
✅ 텍스트 프롬프트로 유연한 결함 지정
✅ 다중 결함 동시 탐지
✅ 시각적 근거 명확 (박스로 표시)
```

**약점:**
```
❌ 결함/정상 이진 분류만 가능 (세부 분류 불가)
❌ 왜 결함인지 설명 불가
❌ 프롬프트 엔지니어링 필요
❌ 추론 속도 보통 (~1.5초)
```

**적합한 용도:** 결함 위치 시각화, 작업자 가이드, 품질 검토

---

### 3-Way 상호 보완 구조

```
┌─────────────────────────────────────────────────────────────┐
│                    입력 이미지 (CT + RGB)                    │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│ 통합검사기  │     │     VLM       │     │     VLG       │
│   (CNN+AE)    │     │  (Qwen3-VL)   │     │(GroundingDINO)│
├───────────────┤     ├───────────────┤     ├───────────────┤
│ • 분류 정확도 │     │ • 자연어 설명 │     │ • 위치 좌표   │
│ • 신뢰도 점수 │     │ • 소견서 생성 │     │ • 바운딩 박스 │
│ • Pass/Fail   │     │ • 결함 해석   │     │ • 다중 탐지   │
└───────┬───────┘     └───────┬───────┘     └───────┬───────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      최종 진단 리포트                        │
│  ┌─────────────┬─────────────────────┬─────────────────┐   │
│  │ 판정: 불량  │ 소견: 중앙부 기공   │ 위치: [x,y,w,h] │   │
│  │ 신뢰도: 95% │ 심각도: 중간        │ 개수: 3개       │   │
│  └─────────────┴─────────────────────┴─────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### 모델별 출력 예시

**동일한 결함 이미지에 대한 3개 모델 출력:**

| 모델 | 출력 예시 |
|------|-----------|
| **통합 검사기** | `prediction: "internal_defect", confidence: 0.95, class: "module_porosity"` |
| **VLM** | `"이미지 중앙부에 직경 약 2mm의 기공(porosity)이 관찰됩니다. 이는 제조 과정에서 가스가 빠져나가지 못해 발생한 것으로 추정되며, 배터리 성능에 영향을 줄 수 있습니다."` |
| **VLG** | `[{label: "porosity", bbox: [0.45, 0.52, 0.08, 0.06], score: 0.87}]` |

### 왜 3개 모델이 필요한가?

| 질문 | 답변 모델 |
|------|-----------|
| "이 배터리는 불량인가요?" | **통합 검사기** (정확한 분류) |
| "왜 불량인가요? 설명해주세요" | **VLM** (자연어 설명) |
| "결함이 정확히 어디에 있나요?" | **VLG** (바운딩 박스) |

### 결론

단일 모델로는 품질 검사의 모든 요구사항을 충족할 수 없습니다.

```
통합 검사기 → 신뢰할 수 있는 최종 판정 (CT+RGB 논리 결합)
VLM        → 사람이 이해할 수 있는 설명
VLG        → 시각적 근거 제시
```

**3-Way 검사 시스템의 가치:**
- 각 모델의 강점을 결합하여 약점 보완
- 검사자에게 다각적 정보 제공
- 설명 가능한 AI (XAI) 구현
- 오탐지 감소 (교차 검증 효과)

---

## 웹 애플리케이션

### 기술 스택
- **Framework**: Streamlit
- **Port**: 8501
- **Features**: 실시간 추론, 시각화, 리포트 생성

### 화면 구성

#### 1. 메인 대시보드
- 이미지 업로드 (CT/RGB/둘 다)
- 분석 모드 선택
- 실시간 진행 상태 표시

#### 2. 결과 요약 페이지
```
┌─────────────────────────────────────────────┐
│  🔬 통합 검사기    │  🤖 VLM     │  🎯 VLG     │
│  (CNN+AE)      │ (Qwen3-VL) │(GroundingDINO)│
├─────────────────┼─────────────┼─────────────┤
│  CT Grad-CAM   │  CT 분석    │  CT 탐지    │
│  RGB Error Map │  RGB 분석   │  RGB 탐지   │
├─────────────────┴─────────────┴─────────────┤
│  📊 최종 진단 리포트                         │
│  판정: 내부불량 / 외부불량 / 복합불량 / 정상  │
└─────────────────────────────────────────────┘
```

#### 3. 상세 분석 페이지
- AI 소견서 (VLM 생성)
- 결함 위치 하이라이트
- 추론 로그

### 실행 방법
```bash
source venv/bin/activate
streamlit run webapp/app.py --server.port 8501 --server.address 0.0.0.0
```

---

## 학습 및 평가

### TensorBoard 로깅
```bash
# CT CNN
tensorboard --logdir=models/ct_cnn/runs --port=6006

# RGB AutoEncoder
tensorboard --logdir=models/rgb_ae/runs --port=6007
```

### 로깅 항목
- **Scalars**: Loss, Accuracy, ROC-AUC, F1-Score
- **Images**: 원본/재구성/에러맵, Grad-CAM
- **Histograms**: Score 분포 (Normal vs Defect)
- **Figures**: Confusion Matrix, ROC Curve, PR Curve

### 평가 명령어
```bash
# CT CNN 테스트
python models/ct_cnn/test.py --checkpoint models/ct_cnn/checkpoints/ct_unified_best_*.pt

# RGB AE 테스트
python models/rgb_ae/test.py --checkpoint models/rgb_ae/checkpoints/rgb_ae_best_*.pt
```

---

## 프로젝트 구조

```
battery-inspection/
├── models/
│   ├── ct_cnn/                        # CT CNN (다중 아키텍처)
│   │   ├── model.py                   # 모델 디스패치 (create_model)
│   │   ├── model_late_fusion.py       # ★ Late Fusion v2 (최고 F1=0.803)
│   │   ├── model_cbam.py             # CBAM 어텐션
│   │   ├── model_drn_aspp.py         # DRN+ASPP (DeepLabV3+ 스타일)
│   │   ├── model_deeplabv3.py        # DeepLabV3+ 전이학습
│   │   ├── model_timm.py            # ConvNeXt/EfficientNet (timm)
│   │   ├── model_metadata.py         # Early Fusion (메타데이터 MLP)
│   │   ├── model_hdcnn.py            # HD-CNN (계층적 분류)
│   │   ├── model_hierarchical.py     # Hierarchical 분류
│   │   ├── train.py / test.py        # 범용 학습/평가
│   │   ├── train_late_fusion.py      # Late Fusion 전용 학습
│   │   ├── train_metadata.py         # Metadata 전용 학습
│   │   ├── checkpoints/              # 모델 체크포인트
│   │   └── results/                  # 테스트 결과 JSON + Confusion Matrix
│   │
│   ├── rgb_ae/                        # RGB AutoEncoder (이상 탐지)
│   │   ├── model.py                   # ConvAutoEncoder 정의
│   │   ├── train.py / test.py        # 학습/평가
│   │   ├── checkpoints/              # 체크포인트 + threshold.json
│   │   └── results/                  # 테스트 결과
│   │
│   ├── ct_ae/                         # CT AutoEncoder (실험)
│   ├── ct_yolo/                       # CT YOLO (실험)
│   │
│   ├── inspector/                     # 통합 검사 모듈
│   │   ├── inspector.py              # CNN+AE 논리 결합
│   │   ├── ct_ensemble_inspector.py  # CT 앙상블 검사기
│   │   ├── predictor.py              # CT/RGB 개별 예측기
│   │   └── gradcam.py                # Grad-CAM 시각화
│   │
│   ├── vlm/                           # VLM (Qwen3-VL, Gemini)
│   │   ├── inference.py              # Qwen3-VL 추론 (CT/RGB)
│   │   ├── inference_gemini.py       # Gemini API 추론
│   │   ├── prompts.py                # 분석 프롬프트 (CT/RGB)
│   │   ├── test_vlm.py              # 단일 이미지 테스트
│   │   └── test_vlm_eval.py         # 대규모 평가 스크립트
│   │
│   └── vlg/                           # VLG (GroundingDINO)
│       ├── inference.py              # GroundingDINO 추론
│       ├── inference_yoloworld.py    # YOLO-World (비교용)
│       └── prompts.py                # 탐지 프롬프트
│
├── training/
│   ├── configs/                       # 학습 설정 YAML (20+ 실험 config)
│   │   ├── cnn_ct_late_fusion.yaml   # ★ Late Fusion 설정
│   │   ├── cnn_ct_drn_aspp.yaml      # DRN+ASPP 설정
│   │   ├── cnn_ct_cbam.yaml          # CBAM 설정
│   │   ├── vlm_eval.yaml            # VLM CT 평가
│   │   ├── vlm_eval_rgb.yaml        # VLM RGB 평가
│   │   └── config_loader.py          # YAML 설정 로더
│   ├── data/                          # 데이터 파이프라인
│   │   ├── dataset.py                # BatteryDataset
│   │   ├── dataset_metadata.py       # 메타데이터 Dataset
│   │   ├── dataloader.py            # DataLoader + WeightedSampler
│   │   ├── transforms.py            # 데이터 증강 (Albumentations)
│   │   └── splits/                   # train/val/test split 파일
│   ├── evaluation/                    # 평가 메트릭
│   └── visualization/                 # TensorBoard Logger
│
├── scripts/                           # 데이터 전처리/분할 스크립트
│   ├── preprocess.py                 # CT 이미지 전처리 (4000→512)
│   ├── fix_ct_split_by_battery.py    # CT 배터리 ID 기준 split
│   ├── fix_rgb_split_by_battery.py   # RGB 배터리 ID 기준 split
│   └── ...
│
├── webapp/                            # Streamlit 웹 대시보드
│   ├── app.py                        # 메인 앱
│   ├── pages/
│   │   ├── home.py                   # 업로드 페이지
│   │   ├── processing.py            # 추론 처리
│   │   ├── summary.py               # 결과 요약
│   │   └── detail.py                # 상세 분석
│   └── utils/                        # 세션, 스타일, 결함 정보
│
├── docs/                              # 문서
│   ├── MODEL_PERFORMANCE.md          # 전체 모델 성능 비교표
│   └── ...
│
├── config.py                          # 중앙 설정 관리 (.env 로드)
├── PORTFOLIO.md                       # 상세 포트폴리오
├── TASK.md                            # 작업 기록
└── requirements.txt                   # 의존성
```

---

## 핵심 성과

### 1. 검출 성능 (Test 기준, 현재 split 35,529 샘플)

**CT CNN (다중 아키텍처 실험)**:
| 모델 | Test F1 | Accuracy | ROC-AUC | 비고 |
|------|---------|----------|---------|------|
| **Late Fusion v2** | **0.803** | **80.3%** | **0.944** | ★ 최고 성능 |
| DRN+ASPP | 0.794 | 78.2% | 0.943 | 순수 이미지 최고 |
| Metadata v3 | 0.791 | 78.0% | 0.965 | ROC-AUC 최고 |
| CBAM (768) | 0.862 | 86.3% | 0.968 | x축 포함 학습 |

**RGB AutoEncoder**: **98.85% Accuracy**, **99.4% F1**, ROC-AUC 0.909

### 2. 체계적 아키텍처 실험
- ResNet18, ConvNeXt-Tiny, EfficientNet-B0/B4, CBAM, HD-CNN, DRN+ASPP, DeepLabV3+
- Late Fusion, Metadata, Image-Only, Hierarchical 등
- 전처리 전략: Raw Resize, Outline Crop, Patch, Resize 512/768
- 데이터 분할: 배터리 단위 분리 (Data Leakage 방지)

### 3. 멀티모달 통합 분석
- 내부(CT) + 외부(RGB) 동시 검사
- 복합불량 자동 탐지

### 4. 설명 가능한 AI (XAI)
- Grad-CAM으로 결함 위치 시각화
- VLM 자연어 소견서 생성
- VLG 객체 탐지 바운딩 박스

### 5. 실용적인 배포
- Streamlit 웹 인터페이스
- 실시간 추론 및 결과 확인
- Docker 컨테이너화 가능

---

## 향후 개선 방향

1. **과적합 해결**: 더 강한 정규화 (Dropout 0.7, MixUp/CutMix, Label Smoothing 0.15)
2. **Temperature Scaling**: 예측 과신뢰 보정 (오답에도 84~91% 확률 예측하는 문제)
3. **VLM 대규모 평가**: Qwen3-VL 2B 확정 실패 (500샘플, Acc 20.8%), 8B/32B는 GPU 24GB+ 필요
4. **더 많은 배터리 데이터 확보**: 현재 92개 → 최소 200개+ 필요
5. **x축 처리 전략 확정**: 운영 환경에서 x축 제외 가능 여부 결정
6. **Active Learning**: 불확실한 샘플 자동 수집
7. **API 서버**: FastAPI 기반 REST API 구축

---

## 라이선스

이 프로젝트는 연구 및 교육 목적으로 개발되었습니다.

---

## 연락처

프로젝트 관련 문의사항이 있으시면 연락 주세요.

---

*Last Updated: 2026-02-19*
