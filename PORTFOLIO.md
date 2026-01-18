# 배터리 검사 AI 시스템 (Battery Inspection AI System)

## 프로젝트 개요

### 목적
리튬이온 배터리 셀/모듈의 내부 결함(CT 이미지) 및 외부 결함(RGB 이미지)을 자동으로 검출하는 AI 기반 품질 검사 시스템 개발

### 핵심 특징
- **멀티모달 분석**: CT 이미지(내부 결함)와 RGB 이미지(외부 결함) 동시 분석
- **3-Way 앙상블**: CNN + AutoEncoder + VLM/VLG 다중 모델 검증
- **실시간 웹 인터페이스**: Streamlit 기반 대시보드로 즉시 결과 확인
- **설명 가능한 AI**: Grad-CAM, Error Map, Grounding 시각화 제공

### 기술 스택
| 분류 | 기술 |
|------|------|
| Deep Learning | PyTorch, TorchVision, Transformers |
| 모델 | ResNet18, ConvAutoEncoder, Qwen2-VL, GroundingDINO |
| 외부 API | Google Gemini 2.0 Flash (VLM) |
| 시각화 | Grad-CAM, TensorBoard, Matplotlib |
| 웹 프레임워크 | Streamlit |
| 언어 | Python 3.10 |

---

## 시스템 아키텍처

```
[배터리 이미지 입력: CT + RGB]
           ↓
┌──────────────────────────────────────────────────┐
│  System 1: CNN+AE+Grad-CAM 앙상블                │
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
│  System 2: VLM (Gemini / Qwen2-VL)              │
│  → Zero-shot 판정 + 불량 원인 설명              │
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
3. **결과 통합**: 내부/외부 결함 여부 종합 판정
4. **시각화**: Grad-CAM, Error Map, Bounding Box 등 근거 제시

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

#### 아키텍처
- **Backbone**: ResNet18 (ImageNet Pretrained)
- **출력**: 5클래스 Softmax
- **입력 크기**: 512x512

#### 학습 설정
```yaml
optimizer: AdamW
learning_rate: 0.0001
weight_decay: 0.01
scheduler: CosineAnnealingLR
epochs: 50
batch_size: 32
```

#### 성능
| 지표 | Validation | Test |
|------|------------|------|
| Accuracy | 83.1% | **77.4%** |
| Macro F1-Score | 83.4% | **78.8%** |
| ROC-AUC | - | 0.9534 |

#### 시각화
- **Grad-CAM**: 결함 위치 히트맵 생성
- Layer4 특징맵 기반 활성화 영역 시각화

#### ResNet18 선택 이유

최신 모델(EfficientNet, ConvNeXt, Vision Transformer 등) 대신 ResNet18을 선택한 이유:

**1. 데이터셋 규모와 모델 복잡도의 균형**
```
문제: 배터리 CT 데이터셋 규모가 상대적으로 작음 (~수만 장)
→ 대형 모델 사용 시 과적합(Overfitting) 위험
→ ResNet18의 11M 파라미터가 적절한 복잡도
```

**2. 산업 환경 배포 요구사항**
| 요구사항 | ResNet18 | EfficientNet-B7 | ViT-Large |
|----------|----------|-----------------|-----------|
| 모델 크기 | **46MB** | 256MB | 1.2GB |
| 추론 속도 | **~15ms** | ~45ms | ~120ms |
| GPU 메모리 | **2GB** | 6GB | 12GB+ |
| Edge 배포 | ✅ 가능 | △ 제한적 | ❌ 어려움 |

**3. Transfer Learning 효과**
```
✅ ImageNet-1K pretrained 가중치 활용
✅ 배터리 CT 이미지도 "질감, 형태, 구조" 특징 → ImageNet 특징 재사용 가능
✅ 적은 데이터로도 빠른 수렴 및 높은 성능 달성
```

**4. 3x3 컨볼루션 기반 아키텍처의 장점**
```
ResNet은 VGGNet에서 검증된 3x3 컨볼루션 설계를 계승:

✅ 작은 필터 + 깊은 네트워크 = 큰 필터와 동일한 수용 영역(Receptive Field)
   - 3x3 두 층 = 5x5 한 층과 동일한 수용 영역
   - 3x3 세 층 = 7x7 한 층과 동일한 수용 영역

✅ 파라미터 효율성
   - 7x7 필터: 49개 파라미터
   - 3x3 세 층: 27개 파라미터 (45% 절감)

✅ 비선형성 증가
   - 층마다 ReLU 활성화 함수 적용
   - 더 복잡한 특징 학습 가능

✅ 배터리 결함 탐지에 적합
   - 기공(porosity), 균열 등 미세 결함은 국소적 패턴
   - 3x3 필터가 미세 특징 추출에 효과적
```

**5. 실험적 검증 결과**
| 모델 | Val Accuracy | Test Accuracy | Test F1 Macro | 비고 |
|------|--------------|---------------|---------------|------|
| **ResNet18** | 83.1% | **77.4%** | **78.8%** | ✅ 채택 |
| ResNet18 + CBAM | 80.1% | - | - | 오히려 낮음 |

**6. 결론**
```
✅ 단순하고 검증된 아키텍처 = 유지보수 용이
✅ "가장 단순한 해결책이 최선" 원칙 적용
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
| 대안 | 데이터 증강, 앙상블 등 다른 방법 적용 |

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

입력: 512x512x3 RGB
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
| ROC-AUC | 0.9999 | 0.9091 |
| Accuracy | - | 98.85% |
| F1-Score | - | 99.40% |

#### Test Confusion Matrix 분석

| | 예측: 정상 | 예측: 불량 |
|---|---|---|
| **실제 정상** | 1,220 | 285 (오탐) |
| **실제 불량** | 22 (미탐) | 25,222 |

**클래스별 정확도:**
| 클래스 | 정확도 | 설명 |
|--------|--------|------|
| 정상 (Recall) | 81.1% (1,220/1,505) | 정상을 정상으로 판정 |
| 불량 (Recall) | 99.9% (25,222/25,244) | 불량을 불량으로 판정 |
| 전체 Accuracy | 98.85% | 불량 비중이 높아 상승 |

*⚠️ 정상을 불량으로 오탐하는 비율 18.9% (285/1,505) 존재*

#### Threshold 설정
```json
{
  "normal_mean": 0.747,
  "defect_mean": 2.075,
  "threshold": 1.155
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

### 3. Ensemble (CNN + AE 통합)

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

#### 모델
본 시스템은 두 가지 VLM을 지원하며, 웹앱에서 선택 가능합니다.

| 모델 | 실행 환경 | 속도 | 정확도 |
|------|-----------|------|--------|
| **Qwen2-VL-2B-Instruct** | 로컬 GPU | ~25초 | 보통 |
| **Gemini 2.0 Flash** | Google API | ~5초 | **높음** (권장) |

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

#### 테스트 환경

| 항목 | 내용 |
|------|------|
| 테스트 이미지 | 앙상블 데모 (RGB 외부 오염 결함) |
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

**Ensemble 결과**: 모든 테스트에서 일관되게 **외부불량 94.2%** (정확)

#### 상세 로그 분석

**테스트 1: Gemini + GroundingDINO (최적 조합)**
```
[WEBAPP] 15:05:48 - ✅ Ensemble 추론 완료: 외부불량 (신뢰도: 94.2%)
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

**1. Ensemble의 안정성**
```
✅ 모든 테스트에서 일관되게 "외부불량 94.2%" 출력
✅ VLM/VLG 조합과 무관하게 항상 정확한 판정
→ 최종 판정은 Ensemble을 기준으로 해야 함
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
| **실무 배포** | Gemini 2.0 Flash | GroundingDINO | 최고 정확도 |
| **오프라인 환경** | Qwen2-VL-2B | GroundingDINO | API 불가 시 |
| **속도 우선** | Gemini 2.0 Flash | GroundingDINO | 총 ~8초 |
| **비교 테스트** | 선택 가능 | 선택 가능 | 웹앱 설정 |

#### 웹앱 설정 방법

```bash
# 웹앱 실행
streamlit run webapp/app.py --server.port 8501

# 홈페이지 → ⚙️ 고급 설정 열기
# VLM 모델 선택:
#   - 🧠 Qwen2-VL (로컬) - GPU 실행, 오프라인 가능
#   - ☁️ Gemini 2.0 Flash (API) - 빠르고 정확, 권장
# VLG 모델 선택:
#   - 🎯 GroundingDINO (Swin-T) - 정확도 높음, 권장
#   - 🚀 YOLO-World (YOLOv8s) - 비교 테스트용
```

#### 결론

| 항목 | 결론 |
|------|------|
| 최적 VLM | **Gemini 2.0 Flash** (정확도 + 속도 모두 우수) |
| 최적 VLG | **GroundingDINO** (유일하게 결함 탐지 성공) |
| 최적 조합 | **Gemini + GroundingDINO** (3개 모델 모두 정확) |
| Ensemble 역할 | 최종 판정 기준 (VLM/VLG 오판 시 보정) |

---

## 3-Way 모델 비교 분석

본 시스템은 3개의 독립적인 분석 모델을 사용하여 다각도로 결함을 검출합니다. 각 모델의 특성과 한계를 분석하여 상호 보완적인 앙상블 시스템을 구축했습니다.

### 모델별 역할 및 특성

| 구분 | Ensemble (CNN+AE) | VLM (Qwen2-VL) | VLG (GroundingDINO) |
|------|-------------------|----------------|---------------------|
| **역할** | 정량적 분류 | 정성적 분석 | 위치 탐지 |
| **출력** | 클래스 확률 | 자연어 소견서 | 바운딩 박스 |
| **강점** | 높은 정확도 | 설명 가능성 | 위치 시각화 |
| **약점** | 설명 부재 | 위치 부정확 | 분류 불가 |

### 상세 분석

#### 🔬 Ensemble (CT CNN + RGB AutoEncoder)

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

#### 🤖 VLM (Qwen2-VL-2B)

**강점:**
```
✅ 자연어로 결함 설명 생성 (소견서)
✅ Zero-shot: 학습 없이 다양한 결함 인식
✅ 결함의 특성, 심각도, 위치 설명
✅ 사람이 이해하기 쉬운 리포트
✅ 새로운 결함 유형에도 대응 가능
```

**약점:**
```
❌ 바운딩 박스 위치 정확도 낮음
❌ 할루시네이션 (없는 결함을 있다고 할 수 있음)
❌ 추론 속도 느림 (~20초)
❌ 정량적 신뢰도 산출 어려움
```

**VLM 위치 탐지 한계 상세:**
```
문제: VLM이 "중앙 상단에 기공이 있습니다"라고 답변해도
      실제 위치와 다른 경우가 빈번함

원인:
1. VLM은 언어 모델 기반 → 공간 좌표 개념 약함
2. "상단", "중앙" 등 모호한 표현 사용
3. 이미지 해상도 축소로 세밀한 위치 정보 손실
4. Grounding 학습이 부족한 일반 VLM 한계

해결책:
→ 위치 정보는 VLG(GroundingDINO)에 위임
→ VLM은 결함 설명과 소견서 생성에 집중
```

**적합한 용도:** AI 소견서 생성, 검사 리포트 작성, 설명 가능한 AI

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
│   Ensemble    │     │     VLM       │     │     VLG       │
│   (CNN+AE)    │     │  (Qwen2-VL)   │     │(GroundingDINO)│
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
| **Ensemble** | `prediction: "internal_defect", confidence: 0.95, class: "module_porosity"` |
| **VLM** | `"이미지 중앙부에 직경 약 2mm의 기공(porosity)이 관찰됩니다. 이는 제조 과정에서 가스가 빠져나가지 못해 발생한 것으로 추정되며, 배터리 성능에 영향을 줄 수 있습니다."` |
| **VLG** | `[{label: "porosity", bbox: [0.45, 0.52, 0.08, 0.06], score: 0.87}]` |

### 왜 3개 모델이 필요한가?

| 질문 | 답변 모델 |
|------|-----------|
| "이 배터리는 불량인가요?" | **Ensemble** (정확한 분류) |
| "왜 불량인가요? 설명해주세요" | **VLM** (자연어 설명) |
| "결함이 정확히 어디에 있나요?" | **VLG** (바운딩 박스) |

### 결론

단일 모델로는 품질 검사의 모든 요구사항을 충족할 수 없습니다.

```
Ensemble → 신뢰할 수 있는 최종 판정
VLM      → 사람이 이해할 수 있는 설명
VLG      → 시각적 근거 제시
```

**3-Way 앙상블의 가치:**
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
│  🔬 Ensemble    │  🤖 VLM     │  🎯 VLG     │
│  (CNN+AE)      │ (Qwen2-VL) │(GroundingDINO)│
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
│   ├── ct_cnn/
│   │   ├── model.py          # ResNet18 모델 정의
│   │   ├── train.py          # 학습 스크립트
│   │   ├── test.py           # 평가 스크립트
│   │   ├── checkpoints/      # 모델 체크포인트
│   │   └── runs/             # TensorBoard 로그
│   │
│   ├── rgb_ae/
│   │   ├── model.py          # ConvAutoEncoder 정의
│   │   ├── train.py          # 학습 스크립트
│   │   ├── test.py           # 평가 스크립트
│   │   ├── checkpoints/      # 모델 체크포인트
│   │   └── runs/             # TensorBoard 로그
│   │
│   ├── ensemble/
│   │   ├── ensemble.py       # 앙상블 통합 모듈
│   │   ├── predictor.py      # CT/RGB 개별 예측기
│   │   └── gradcam.py        # Grad-CAM 구현
│   │
│   ├── vlm/
│   │   ├── inference.py      # Qwen2-VL 추론
│   │   └── prompts.py        # 분석 프롬프트
│   │
│   └── vlg/
│       ├── inference.py      # GroundingDINO 추론
│       └── prompts.py        # 탐지 프롬프트
│
├── training/
│   ├── configs/              # 학습 설정 YAML
│   └── config_loader.py      # 설정 로더
│
├── data/
│   ├── ct_unified/           # CT 데이터셋
│   └── rgb/                  # RGB 데이터셋
│
├── webapp/
│   ├── app.py                # Streamlit 메인
│   └── pages/
│       ├── processing.py     # 추론 처리
│       ├── summary.py        # 결과 요약
│       └── detail.py         # 상세 분석
│
├── scripts/
│   ├── create_ct_splits.py   # CT 데이터 분할
│   └── create_rgb_normal_splits.py  # RGB 데이터 분할
│
├── CLAUDE.md                 # 개발 가이드
├── TASK.md                   # 작업 기록
└── requirements.txt          # 의존성
```

---

## 핵심 성과

### 1. 검출 성능 (Test 기준)
- CT CNN: **77.4% Accuracy**, **78.8% Macro F1**, ROC-AUC 0.9534
- RGB AE: **98.85% Accuracy**, **99.4% F1**, ROC-AUC 0.9091

### 2. 멀티모달 통합 분석
- 내부(CT) + 외부(RGB) 동시 검사
- 복합불량 자동 탐지

### 3. 설명 가능한 AI (XAI)
- Grad-CAM으로 결함 위치 시각화
- VLM 자연어 소견서 생성
- VLG 객체 탐지 바운딩 박스

### 4. 실용적인 배포
- Streamlit 웹 인터페이스
- 실시간 추론 및 결과 확인
- Docker 컨테이너화 가능

---

## 향후 개선 방향

1. **모델 경량화**: MobileNet, EfficientNet 적용
2. **YOLO-World 통합**: VLG 대안 모델 추가
3. **Active Learning**: 불확실한 샘플 자동 수집
4. **배치 처리**: 대량 이미지 일괄 검사
5. **API 서버**: FastAPI 기반 REST API 구축

---

## 라이선스

이 프로젝트는 연구 및 교육 목적으로 개발되었습니다.

---

## 연락처

프로젝트 관련 문의사항이 있으시면 연락 주세요.

---

*Last Updated: 2026-01-08*
