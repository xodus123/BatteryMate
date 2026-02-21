# TensorBoard 가이드 - 학습 모니터링 및 지표 해석

> **작성일**: 2026-01-04
> **수정일**: 2026-01-29
> **대상**: CT CNN 5클래스 다중분류 학습 (기본 CNN + CNN+Metadata)

---

## 1. 실행 방법

```bash
# ============================================================
# 기본 CT CNN 학습
# ============================================================
# 터미널 1: 학습 시작
python models/ct_cnn/train.py --config cnn_ct_unified

# 터미널 2: TensorBoard 실행
tensorboard --logdir models/ct_cnn/logs --port 6006

# ============================================================
# CNN + Metadata 학습 (2026-01-29 추가)
# ============================================================
# 터미널 1: 학습 시작
python -m models.ct_cnn.train_metadata --config training/configs/cnn_ct_random_crop.yaml

# 터미널 2: TensorBoard 실행
tensorboard --logdir models/ct_cnn/logs --port 6006

# 브라우저 접속
http://localhost:6006
```

---

## 2. 로그 파일 구조

```
models/ct_cnn/
├── logs/
│   ├── resnet18_lr0.0001_bs16_{timestamp}/
│   │   └── events.out.tfevents.*     ← TensorBoard 이벤트
│   └── train_{timestamp}.csv          ← CSV 로그 (epoch별 요약)
│
└── checkpoints/
    ├── ct_unified_best_{timestamp}.pt  ← Best F1 모델
    ├── ct_unified_last_{timestamp}.pt  ← 마지막 epoch 모델
    └── ct_unified_top{k}_{timestamp}.pt ← Top-K 모델 (save_top_k=3)
```

### CSV 로그 형식
```csv
epoch,train_loss,val_loss,val_f1_macro,val_accuracy,lr
1,0.8234,0.6521,0.4532,0.6234,0.0001
2,0.5123,0.4892,0.5678,0.7123,0.0001
```

---

## 3. TensorBoard 탭별 가이드

### 3.1 Scalars (스칼라)

**주요 그래프:**

| 그래프 | 의미 | 정상 패턴 | 이상 패턴 |
|--------|------|-----------|-----------|
| `Loss/train` | 학습 데이터 손실 | ↓ 꾸준히 감소 | 발산, 진동 |
| `Loss/val` | 검증 데이터 손실 | ↓ 감소 후 안정 | train↓ val↑ (과적합) |
| `Metrics/f1_macro` | 5클래스 평균 F1 | ↑ 증가 | 정체, 하락 |
| `Metrics/accuracy` | 전체 정확도 | ↑ 증가 | - |
| `LR` | Learning Rate | 주기적 변화 (Cosine) | - |

**클래스별 에러 분석:**

| 그래프 | 의미 |
|--------|------|
| `Errors/{클래스}/TP` | True Positive (정확히 분류) |
| `Errors/{클래스}/FP` | False Positive (다른 클래스를 이 클래스로 오분류) |
| `Errors/{클래스}/FN` | False Negative (이 클래스를 다른 것으로 오분류) |
| `Errors/{클래스}/Precision` | TP / (TP + FP) |
| `Errors/{클래스}/Recall` | TP / (TP + FN) |

**클래스별 성능:**

| 그래프 | 의미 |
|--------|------|
| `PerClass/F1/{클래스}` | 클래스별 F1 Score |
| `PerClass/Precision/{클래스}` | 클래스별 Precision |
| `PerClass/Recall/{클래스}` | 클래스별 Recall |

---

### 3.2 Images (이미지)

| 이미지 | 설명 | 확인 포인트 |
|--------|------|-------------|
| `ConfusionMatrix/val` | 5x5 혼동 행렬 | 대각선이 진할수록 좋음 |
| `ErrorSummary/table` | TP/FP/FN 요약 테이블 | FP/FN 높은 클래스 확인 |
| `Distribution/val` | 클래스 분포 막대그래프 | 불균형 정도 확인 |
| `GradCAM/{클래스}` | Grad-CAM 시각화 | 모델이 주목하는 영역 확인 |
| `Misclassified/{true}→{pred}` | 오분류 이미지 샘플 | 어떤 케이스에서 실패하는지 확인 |

**Confusion Matrix 읽는 법:**

```
              Predicted
            0   1   2   3   4
         ┌───┬───┬───┬───┬───┐
       0 │███│   │   │   │   │  cell_normal
True   1 │   │███│   │   │   │  cell_porosity
       2 │   │   │███│   │   │  module_normal
       3 │   │   │   │███│   │  module_porosity
       4 │   │   │   │   │███│  module_resin_overflow
         └───┴───┴───┴───┴───┘

███ = 높은 값 (좋음: 대각선에 집중)
대각선 외 셀이 밝으면 = 오분류 발생
```

**주의해야 할 패턴:**
- `cell_normal` ↔ `module_normal` 혼동: 정상 이미지 간 구분 어려움
- `cell_porosity` ↔ `module_porosity` 혼동: porosity 유형 간 혼동
- `module_resin_overflow` FN 높음: 희소 클래스라 놓치기 쉬움

**오분류 이미지 (Misclassified Images):**

학습 중 틀린 예측을 한 이미지를 시각화합니다.

| 항목 | 설명 |
|------|------|
| 태그 형식 | `Misclassified/{정답클래스}→{예측클래스}` |
| 정렬 기준 | 신뢰도가 높은 오분류 샘플 우선 (더 심각한 실패) |
| 클래스당 샘플 수 | 최대 4개 (설정 변경 가능) |

**확인 포인트:**
- 이미지에 실제 결함이 보이는가?
- 이미지 스타일이 다른 클래스와 다른가?
- 특정 배터리 타입/축에서 오분류 집중되는가?

**예시:**
```
Misclassified/cell_porosity→cell_normal
 → cell_porosity인데 cell_normal로 잘못 예측 (신뢰도 95%)
 → 결함이 너무 작거나 이미지에서 안 보일 수 있음
```

---

### 3.3 Histograms (히스토그램)

**예측 확률 분포:**

| 히스토그램 | 의미 | 좋은 형태 |
|------------|------|-----------|
| `Probabilities/{클래스}/all` | 전체 샘플의 해당 클래스 확률 | - |
| `Probabilities/{클래스}/true_samples` | 실제 해당 클래스인 샘플의 확률 | 1.0 근처에 집중 |
| `Probabilities/{클래스}/false_samples` | 다른 클래스 샘플의 확률 | 0.0 근처에 집중 |

**예측 신뢰도 분포:**

| 히스토그램 | 의미 | 좋은 형태 |
|------------|------|-----------|
| `Confidence/correct_predictions` | 정답 예측의 신뢰도 | 1.0 근처에 집중 |
| `Confidence/incorrect_predictions` | 오답 예측의 신뢰도 | 낮은 값에 분포 |
| `Confidence/all_predictions` | 전체 예측 신뢰도 | - |

**해석:**
- `true_samples`가 1.0 근처 + `false_samples`가 0.0 근처 = 모델이 잘 구분함
- `correct`가 높고 `incorrect`가 낮음 = 모델이 확신할 때 정확함
- `incorrect`도 높은 신뢰도 = 과신(overconfident) 문제

---

### 3.4 PR Curves (Precision-Recall 곡선)

| 곡선 | 설명 |
|------|------|
| `PR_Curve/{클래스}` | 클래스별 Precision-Recall 곡선 |

**읽는 법:**
- X축: Recall (재현율)
- Y축: Precision (정밀도)
- 곡선이 오른쪽 위에 가까울수록 좋음
- AUC (곡선 아래 면적)가 1.0에 가까울수록 좋음

**클래스별 주의점:**
- `module_resin_overflow`: 샘플 수가 적어 PR 곡선이 불안정할 수 있음
- Recall이 낮으면 해당 클래스를 놓치는 경우가 많음
- Precision이 낮으면 다른 클래스를 해당 클래스로 잘못 분류

---

### 3.5 Graphs (모델 구조)

ResNet18 모델의 전체 구조를 시각화합니다.

**확인 포인트:**
- 입력: `(1, 3, 1024, 1024)`
- 출력: `(1, 5)` - 5클래스 logits
- 주요 레이어: conv1 → layer1~4 → avgpool → fc

---

## 4. 학습 상태 진단

### 4.1 정상 학습

```
✅ train_loss: 꾸준히 감소
✅ val_loss: train과 비슷하게 감소
✅ f1_macro: 꾸준히 증가
✅ 모든 클래스 Recall > 0.5
```

### 4.2 과적합 (Overfitting) - 상세 판정 기준

#### 과적합이란?
모델이 **Train 데이터를 외워버려서** 새로운 데이터(Val/Test)에서 성능이 떨어지는 현상

#### TensorBoard에서 확인 방법

**Step 1: Scalars 탭 → 검색창에 `Loss` 입력**

**Step 2: `Loss/train`과 `Loss/val` 그래프 비교**

```
정상 학습 패턴:
        Loss
    1.0 │╲
        │ ╲  Val Loss
    0.5 │  ╲____
        │   ╲___  Train Loss
    0.2 │      ╲___________
        └────────────────────── Epoch
        (두 선이 함께 내려감, 격차 일정)


과적합 패턴:
        Loss
    1.2 │        ___╱ Val Loss (↑ 증가)
    0.8 │   ____╱
        │  ╱
    0.4 │ ╱____
        │      ╲____ Train Loss (↓ 계속 감소)
    0.1 │          ╲____________
        └────────────────────────── Epoch
              ↑
           과적합 시작점 (두 선이 갈라지는 지점)
```

#### 수치 기준 (Val Loss / Train Loss 비율)

| 비율 | 상태 | 조치 |
|------|------|------|
| **< 1.5** | ✅ 정상 | 학습 계속 |
| **1.5 ~ 2.0** | ⚠️ 주의 | 모니터링 강화 |
| **2.0 ~ 3.0** | 🟠 경고 | Early Stop 고려 |
| **> 3.0** | 🔴 과적합 | 즉시 중단, 설정 조정 |

#### 실제 예시 (이전 CT CNN 학습)

| Epoch | Train Loss | Val Loss | 비율 | 판정 |
|-------|------------|----------|------|------|
| 1 | 0.82 | 0.91 | 1.1x | ✅ 정상 |
| 5 | 0.45 | 0.58 | 1.3x | ✅ 정상 |
| **8** | **0.19** | **0.75** | **3.9x** | 🟠 Best 모델 |
| 12 | 0.08 | 0.95 | 11.9x | 🔴 과적합 |
| 18 | 0.03 | 1.32 | **44x** | 🔴 심각 |

→ Epoch 8 이후 Val Loss 증가 시작 = 과적합 시작점

#### 과적합 조기 탐지 체크리스트

**매 Epoch 확인:**
- [ ] Val Loss가 이전 epoch보다 증가했는가?
- [ ] Train Loss와 Val Loss 격차가 벌어지고 있는가?
- [ ] Val F1이 정체되거나 하락하는가?

**5 Epoch 연속 확인:**
- [ ] Val Loss가 5 epoch 연속 개선 없음 → Early Stop 트리거
- [ ] Train Loss만 감소하고 Val Loss는 정체 → 과적합 진행 중

#### 과적합 발생 시 조치

| 방법 | 현재 설정 | 조정 방향 | 효과 |
|------|----------|----------|------|
| **Dropout 증가** | 0.3 | → 0.4~0.5 | 뉴런 의존성 감소 |
| **Weight Decay 증가** | 0.01 | → 0.02~0.05 | 가중치 크기 제한 |
| **Data Augmentation 강화** | 6개 | 더 추가 | 데이터 다양성 증가 |
| **Early Stopping patience 감소** | 5 | → 3~4 | 조기 중단 |
| **Label Smoothing 증가** | 0.1 | → 0.15~0.2 | 과신 방지 |
| **Batch Size 증가** | 16 | → 32 | 그래디언트 안정화 |
| **Learning Rate 감소** | 0.0001 | → 0.00005 | 수렴 안정화 |

#### TensorBoard 실시간 모니터링 팁

```bash
# TensorBoard 실행
tensorboard --logdir models/ct_cnn/logs --port 6006

# 브라우저에서 확인
http://localhost:6006
```

1. **Scalars** 탭 → `Loss` 검색
2. 우측 상단 **Smoothing** 슬라이더 0.6으로 조정 (노이즈 제거)
3. 두 그래프가 **갈라지기 시작하는 epoch** 확인
4. 해당 epoch의 체크포인트가 Best 모델일 가능성 높음

### 4.3 과소적합 (Underfitting)

```
⚠️ train_loss: 높은 값에서 정체
⚠️ f1_macro: 낮은 값에서 정체
```

**해결책:**
- 모델 크기 증가 (ResNet18 → ResNet50)
- Learning Rate 조정
- 학습 epoch 증가

### 4.4 클래스 불균형 문제

```
⚠️ module_resin_overflow Recall < 0.3
⚠️ 특정 클래스 FN이 매우 높음
```

**해결책:**
- class_weight 조정 (현재: resin_overflow 25.0, cell_porosity 4.0)
- Focal Loss gamma 증가 (현재 3.0)
- WeightedRandomSampler 사용 (현재 적용됨)

---

## 5. 핵심 평가 지표 상세 설명

### 5.1 기본 지표

| 지표 | 공식 | 의미 | 범위 |
|------|------|------|------|
| **Accuracy** | (TP+TN) / Total | 전체 정확도 | 0~1 |
| **Precision** | TP / (TP+FP) | 예측한 것 중 맞은 비율 | 0~1 |
| **Recall** | TP / (TP+FN) | 실제 정답 중 맞춘 비율 | 0~1 |
| **F1 Score** | 2×(P×R)/(P+R) | Precision과 Recall의 조화평균 | 0~1 |

### 5.2 TP/FP/FN/TN 이해

```
              예측값
            Positive  Negative
         ┌─────────┬─────────┐
Positive │   TP    │   FN    │  ← 실제 Positive 중 놓친 것 = FN
실제값   ├─────────┼─────────┤
Negative │   FP    │   TN    │  ← 실제 Negative를 Positive로 오분류 = FP
         └─────────┴─────────┘
```

- **TP (True Positive)**: 정답을 정답으로 맞춤 ✅
- **FP (False Positive)**: 오답을 정답이라 함 (Type I Error) ⚠️
- **FN (False Negative)**: 정답을 놓침 (Type II Error) ⚠️
- **TN (True Negative)**: 오답을 오답으로 맞춤 ✅

### 5.3 Precision vs Recall 트레이드오프

| 상황 | 우선 지표 | 이유 |
|------|----------|------|
| 불량 탐지 | **Recall** | 불량을 놓치면 안됨 (FN 최소화) |
| 스팸 필터 | **Precision** | 정상 메일을 스팸 처리하면 안됨 (FP 최소화) |
| 균형 필요 | **F1 Score** | 둘 다 중요할 때 |

**배터리 검사에서는 Recall이 더 중요!**
- 불량 배터리를 놓치면 → 안전 문제 발생
- 정상을 불량으로 오분류 → 재검사로 해결 가능

### 5.4 Macro vs Micro vs Weighted 평균

| 평균 방식 | 계산 방법 | 특징 |
|----------|----------|------|
| **Macro** | 클래스별 평균 | 희소 클래스도 동등하게 반영 |
| **Micro** | 전체 TP/FP/FN로 계산 | 다수 클래스에 치우침 |
| **Weighted** | 클래스 샘플 수로 가중 평균 | 실제 분포 반영 |

**현재 설정**: `val_f1_macro` 사용 (희소 클래스 중요시)

### 5.5 Loss 함수 이해

**Focal Loss**: FL(p_t) = -α × (1-p_t)^γ × log(p_t)

| 파라미터 | 현재 값 | 역할 |
|---------|--------|------|
| γ (gamma) | 3.0 | 쉬운 샘플 가중치 감소 (높을수록 어려운 샘플에 집중) |
| α (alpha) | class_weights | 클래스별 가중치 |

**Label Smoothing**: 0.1
- 정답 확률을 1.0 → 0.9로, 오답을 0.0 → 0.1/(K-1)로 부드럽게
- 과신(overconfident) 방지, 일반화 향상

---

## 6. 주요 모니터링 체크리스트

### 6.1 매 Epoch 확인 (필수)

| 체크 항목 | 확인 방법 | 정상 | 이상 |
|----------|----------|------|------|
| Train Loss | `Loss/train` 그래프 | ↓ 꾸준히 감소 | 발산, 정체 |
| Val Loss | `Loss/val` 그래프 | ↓ 감소 또는 안정 | train↓ val↑ |
| Train/Val 격차 | 두 그래프 비교 | 격차 작음 | 격차 커짐 (과적합) |
| F1 Macro | `Metrics/f1_macro` | ↑ 증가 | 정체, 하락 |
| Learning Rate | `LR` 그래프 | Cosine 패턴 | 이상 패턴 |

### 6.2 5 Epoch마다 확인

| 체크 항목 | 확인 방법 | 주의점 |
|----------|----------|--------|
| Confusion Matrix | `Images` 탭 | 대각선 집중 여부 |
| 클래스별 F1 | `PerClass/F1/*` | 특정 클래스만 낮은지 |
| cell_porosity Recall | `PerClass/Recall/cell_porosity` | 목표: > 0.5 |
| resin_overflow Recall | `PerClass/Recall/module_resin_overflow` | 희소 클래스 주의 |

### 6.3 10 Epoch마다 확인 (심층 분석)

| 체크 항목 | 확인 방법 | 분석 포인트 |
|----------|----------|------------|
| 오분류 패턴 | Confusion Matrix 비대각선 | 어떤 클래스끼리 혼동? |
| 예측 신뢰도 | `Confidence/*` 히스토그램 | 과신 문제 여부 |
| Grad-CAM | `GradCAM/*` 이미지 | 모델이 올바른 영역 주목? |
| PR Curve | `PR_Curve/*` | AUC 및 곡선 형태 |

### 6.4 학습 완료 후 체크리스트

- [ ] **Best 모델 저장 확인**: `checkpoints/ct_unified_best_*.pt`
- [ ] **최종 F1 Macro**: 목표 > 0.75
- [ ] **클래스별 성능 균형**:
  - [ ] cell_normal: Recall > 0.8
  - [ ] cell_porosity: Recall > 0.5 (개선 대상)
  - [ ] module_normal: Recall > 0.8
  - [ ] module_porosity: Recall > 0.7
  - [ ] module_resin_overflow: Recall > 0.4 (희소 클래스)
- [ ] **과적합 여부**: train_loss와 val_loss 격차 < 0.2
- [ ] **Early Stopping 트리거 여부**: patience=5 내 개선 없으면 중단

---

## 7. 문제 상황별 대응 가이드

### 7.1 특정 클래스 Recall이 낮을 때

```
증상: cell_porosity Recall < 0.4
```

**진단 순서:**
1. Confusion Matrix에서 FN 확인 → 어떤 클래스로 오분류?
2. Grad-CAM 확인 → 결함 영역을 제대로 보고 있나?
3. 클래스 분포 확인 → 샘플 수 부족?

**해결책:**
| 원인 | 해결 방법 |
|------|----------|
| 샘플 부족 | class_weight 증가, WeightedSampler 확인 |
| 유사 클래스 혼동 | 데이터 증강 강화, gamma 증가 |
| 특징 학습 부족 | 모델 크기 증가, epoch 증가 |

### 7.2 과적합 발생 시

```
증상: train_loss ↓↓ but val_loss ↑
```

**해결책:**
| 방법 | 현재 설정 | 조정 방향 |
|------|----------|----------|
| Dropout 증가 | 0.3 | → 0.4~0.5 |
| Data Augmentation | 6개 transform | 더 강화 |
| Early Stopping | patience=5 | → 3~4 |
| Weight Decay | 0.01 | → 0.02~0.05 |
| Label Smoothing | 0.1 | → 0.15~0.2 |

### 7.3 학습이 정체될 때

```
증상: loss와 F1이 개선되지 않음
```

**해결책:**
| 방법 | 현재 설정 | 조정 방향 |
|------|----------|----------|
| Learning Rate | 0.0001 | → 0.0003 또는 0.00003 |
| Batch Size | 16 | → 32 (메모리 허용 시) |
| 모델 변경 | ResNet18 | → ResNet50 |
| Warmup 추가 | 없음 | Linear Warmup 5 epoch |

---

## 8. 5클래스별 특성

| 클래스 | 비율 | 예상 난이도 | 주의점 |
|--------|------|-------------|--------|
| cell_normal | 28.4% | 쉬움 | module_normal과 혼동 가능 |
| cell_porosity | 9.2% | 보통 | module_porosity와 혼동 가능 |
| module_normal | 28.6% | 쉬움 | cell_normal과 혼동 가능 |
| module_porosity | 32.7% | 보통 | cell_porosity와 혼동 가능 |
| module_resin_overflow | 1.1% | 어려움 | 샘플 부족, Recall 주의 |

---

## 9. 현재 학습 설정 요약

### 9.1 기본 CT CNN 설정

### 모델 설정
| 항목 | 값 |
|------|-----|
| 모델 | ResNet18 (pretrained) |
| Image Size | 1024×1024 |
| Batch Size | 16 |
| Dropout | 0.3 |
| Num Workers | 4 |

### 손실 함수 및 클래스 균형
| 항목 | 값 |
|------|-----|
| 손실 함수 | Focal Loss |
| Focal Gamma | 3.0 |
| Label Smoothing | 0.1 |
| Class Weights | [1.0, 4.0, 1.0, 0.9, 25.0] |
| WeightedRandomSampler | ✅ 적용 |

### 학습 설정
| 항목 | 값 |
|------|-----|
| Optimizer | AdamW |
| Learning Rate | 0.0001 |
| Weight Decay | 0.01 |
| Epochs | 50 |
| Scheduler | CosineAnnealingWarmRestarts (T_0=10, T_mult=2) |
| Gradient Clip | 1.0 |
| AMP | ✅ 적용 |

### Early Stopping
| 항목 | 값 |
|------|-----|
| Monitor | val_f1_macro |
| Mode | max |
| Patience | 5 |
| Min Delta | 0.001 |

### 체크포인트
| 항목 | 값 |
|------|-----|
| Save Best By | val_f1_macro |
| Save Top K | 3 |
| Save Last | ✅ |

### TensorBoard 로깅
| 항목 | 상태 |
|------|------|
| 기본 로깅 (Loss, Metrics, LR) | ✅ |
| Confusion Matrix | ✅ |
| Per-Class Metrics (F1, Precision, Recall) | ✅ |
| Classification Errors (TP/FP/FN) | ✅ |
| Error Summary Table | ✅ |
| PR Curves | ✅ |
| Prediction Confidence | ✅ |
| Misclassified Images | ✅ |
| Grad-CAM | ✅ |
| Model Graph | ✅ |

---

### 9.2 CNN + Metadata 설정 (2026-01-29 추가)

ResNet18에 메타데이터를 결합한 Fusion 모델입니다.

### 모델 설정
| 항목 | 값 |
|------|-----|
| 모델 | ResNet18 + Metadata Fusion |
| Image Size | 512×512 |
| Batch Size | 32 |
| Dropout | 0.3 |
| Metadata Hidden Dim | 64 |
| Metadata Output Dim | 128 |
| Fusion Hidden Dim | 256 |

### 메타데이터 구조
| 인덱스 | 항목 | 값 범위 | 설명 |
|--------|------|---------|------|
| 0 | battery_type | 0 또는 1 | 0=cell, 1=module |
| 1 | axis | 0, 1, 2 | 0=x축, 1=y축, 2=z축 |

**메타데이터 추가 이유:**
- **Axis 상관관계 문제**: x축=99.97% 정상, y/z축=혼재
- **해결책**: 축 정보를 명시적으로 제공하여 이미지 스타일로 축 추론 방지

### 데이터 전처리 (스타일 통일)
| 항목 | 이전 | 현재 |
|------|------|------|
| 정상 이미지 crop | 배터리 전체 또는 큰 영역 | 가늘고 긴 영역 (결함과 동일) |
| 검은 패딩 비율 (정상) | 0% | 71-83% |
| 검은 패딩 비율 (결함) | 78-87% | 78-87% (동일) |

### 손실 함수 및 클래스 균형
| 항목 | 값 |
|------|-----|
| 손실 함수 | Focal Loss |
| Focal Gamma | 2.0 |
| Label Smoothing | 0.07 |
| Class Weights | [2.0, 3.0, 2.5, 0.5, 10.0] |
| WeightedRandomSampler | ✅ 적용 |

### 학습 설정
| 항목 | 값 |
|------|-----|
| Optimizer | AdamW |
| Learning Rate | 0.0001 |
| Weight Decay | 0.03 |
| Epochs | 50 |
| Scheduler | CosineAnnealingWarmRestarts (T_0=10, T_mult=2) |
| Gradient Clip | 1.0 |
| AMP | ✅ 적용 |

### Early Stopping
| 항목 | 값 |
|------|-----|
| Monitor | val_f1_macro |
| Mode | max |
| Patience | 7 |
| Min Delta | 0.001 |

### Config 파일 경로
```bash
training/configs/cnn_ct_random_crop.yaml
```

### Split 파일 경로
```bash
training/data/splits/ct/defect_random/battery_train.txt
training/data/splits/ct/defect_random/battery_val.txt
training/data/splits/ct/defect_random/battery_test.txt
```

---

## 10. AutoEncoder TensorBoard 가이드 (CT AE / RGB AE)

### 10.1 실행 방법

```bash
# ============================================================
# CT AutoEncoder
# ============================================================
# 학습
python models/ct_ae/train.py --config autoencoder_ct

# TensorBoard
tensorboard --logdir models/ct_ae/logs --port 6008

# ============================================================
# RGB AutoEncoder
# ============================================================
# 학습
python models/rgb_ae/train.py --config autoencoder_rgb

# 테스트 (TensorBoard 로깅 포함)
python models/rgb_ae/test.py --checkpoint <path> --visualize

# TensorBoard
tensorboard --logdir models/rgb_ae/logs --port 6009
```

### 10.2 CT AE 학습 로그

#### Scalars (스칼라)

| 그래프 | 의미 | 정상 패턴 |
|--------|------|-----------|
| `Loss/train` | 학습 재구성 손실 | ↓ 꾸준히 감소 |
| `Loss/val` | 검증 재구성 손실 (정상 이미지만) | ↓ 감소 후 안정 |
| `LR` | Learning Rate | ReduceLROnPlateau 패턴 |
| `Metrics/normal_score_mean` | 정상 이미지 평균 점수 | 낮을수록 좋음 |
| `Metrics/normal_score_std` | 정상 이미지 점수 표준편차 | 작을수록 일관적 |
| `Metrics/defect_score_mean` | 결함 이미지 평균 점수 | 높을수록 좋음 |
| `Metrics/defect_score_std` | 결함 이미지 점수 표준편차 | - |
| `Metrics/roc_auc` | ROC-AUC 점수 | 1.0에 가까울수록 좋음 |

#### 핵심 지표: Normal vs Defect Score

```
좋은 분리:
Normal Score: 0.15 ± 0.05  (낮고 일관됨)
Defect Score: 0.35 ± 0.10  (Normal보다 확실히 높음)
→ Threshold 0.25 근처에서 잘 분리

나쁜 분리 (현재 Cell 상태):
Normal Score: 0.15 ± 0.09
Defect Score: 0.15 ± 0.08  (거의 동일!)
→ 분리 불가능
```

#### CSV 로그 형식

```csv
epoch,train_loss,val_loss,normal_score_mean,normal_score_std,defect_score_mean,defect_score_std,roc_auc,lr
1,0.0523,0.0412,0.1523,0.0821,0.1687,0.0912,0.5823,0.001
```

### 10.3 RGB AE 테스트 로그 (상세)

RGB AE 테스트 스크립트는 더 상세한 TensorBoard 로깅을 제공합니다.

#### Scalars

| 그래프 | 의미 |
|--------|------|
| `Test/ROC_AUC` | ROC-AUC 점수 |
| `Test/Accuracy` | 정확도 |
| `Test/F1_Score` | F1 Score |
| `Test/Threshold` | 사용된 Threshold |
| `Test/Optimal_Threshold` | ROC 기반 최적 Threshold |
| `Test/Normal_Score_Mean` | 정상 이미지 평균 점수 |
| `Test/Normal_Score_Std` | 정상 이미지 점수 표준편차 |
| `Test/Defect_Score_Mean` | 결함 이미지 평균 점수 |
| `Test/Defect_Score_Std` | 결함 이미지 점수 표준편차 |

#### Histograms

| 히스토그램 | 의미 | 좋은 형태 |
|------------|------|-----------|
| `Test/Normal_Scores` | 정상 이미지 점수 분포 | 낮은 값에 집중 |
| `Test/Defect_Scores` | 결함 이미지 점수 분포 | Normal과 분리됨 |

**분포 해석:**
```
좋은 케이스:
Normal: [====    ]  (0.0~0.2 집중)
Defect: [    ====]  (0.3~0.5 집중)
→ 겹치는 영역 적음

나쁜 케이스:
Normal: [  ====  ]  (0.1~0.3)
Defect: [  ====  ]  (0.1~0.3)
→ 완전히 겹침, 분리 불가
```

#### Figures (그래프 이미지)

| 그래프 | 설명 | 확인 포인트 |
|--------|------|-------------|
| `Test/Confusion_Matrix` | 2x2 혼동 행렬 | 대각선 집중 여부 |
| `Test/ROC_Curve` | ROC 곡선 | 좌상단에 가까울수록 좋음 |
| `Test/PR_Curve` | Precision-Recall 곡선 | 우상단에 가까울수록 좋음 |
| `Test/Score_Distribution` | 점수 분포 히스토그램 | 두 분포 분리 여부 |

#### Images (재구성 이미지)

| 이미지 | 설명 |
|--------|------|
| `Test/Original` | 원본 입력 이미지 |
| `Test/Reconstructed` | AE 재구성 이미지 |
| `Test/Difference` | 원본과 재구성의 차이 (에러 맵) |

**에러 맵 해석:**
- 정상 이미지: 차이 적음 (어두운 이미지)
- 결함 이미지: 결함 영역에서 큰 차이 (밝은 영역)

### 10.4 CT AE 설정 요약

| 항목 | 값 |
|------|-----|
| 모델 | ConvAutoEncoder |
| Image Size | 1024×1024 |
| Latent Dim | 512 |
| 손실 함수 | MSE Loss |
| 학습 데이터 | 정상 이미지만 (label 0, 2) |

### Cell/Module 별도 Threshold

CT AE에서는 Cell과 Module의 점수 분포가 다르므로 별도 threshold 사용:

| 타입 | Threshold | 전략 |
|------|-----------|------|
| Cell | 0.12 | Recall 우선 (결함 70% 탐지) |
| Module | 0.28 | 균형 (68%/68%) |

**threshold.json 형식:**
```json
{
  "threshold": 0.186,
  "cell_threshold": 0.12,
  "module_threshold": 0.28,
  "method": "separate_cell_module"
}
```

### 10.5 AE 학습 진단

#### 정상 학습

```
✅ train_loss: 꾸준히 감소
✅ val_loss: train과 비슷하게 감소
✅ normal_score_mean < defect_score_mean
✅ ROC-AUC > 0.7
```

#### 분리 실패

```
⚠️ normal_score ≈ defect_score (거의 동일)
⚠️ ROC-AUC ≈ 0.5 (랜덤 수준)
```

**원인 및 해결책:**
| 원인 | 해결책 |
|------|--------|
| 정상/결함 스타일 유사 | 전처리 개선, 데이터 증강 |
| 모델 용량 부족 | Latent dim 증가, 레이어 추가 |
| 학습 데이터 부족 | 더 많은 정상 데이터 |
| Cell/Module 혼재 | 별도 Threshold 적용 |

---

## 11. Hierarchical CNN TensorBoard 가이드

### 11.1 개요

2단계 분류 모델로, Coarse(대분류)와 Fine(세분류)를 동시에 학습합니다.

```
┌─────────────────────────────────────────────────────┐
│  1단계 (Coarse): Normal vs Defect                   │
│  ├─ Normal (0): cell_normal, module_normal          │
│  └─ Defect (1): cell_porosity, module_porosity,     │
│                 module_resin_overflow               │
├─────────────────────────────────────────────────────┤
│  2단계 (Fine): 5클래스 세부 분류                      │
│  └─ Defect인 경우만 Fine 분류 사용                   │
└─────────────────────────────────────────────────────┘
```

### 11.2 실행 방법

```bash
# 학습
python models/ct_cnn/train_hierarchical.py --config cnn_ct_hierarchical

# TensorBoard
tensorboard --logdir models/ct_cnn/logs --port 6006
```

### 11.3 Scalars (Loss)

| 그래프 | 의미 | 설명 |
|--------|------|------|
| `Loss/train_coarse` | Coarse 분류 손실 | Normal vs Defect 2분류 |
| `Loss/train_fine` | Fine 분류 손실 | 5클래스 세부 분류 |
| `Loss/train_total` | 결합 손실 | coarse_weight × coarse + fine_weight × fine |
| `Loss/val` | 검증 손실 | 전체 검증 손실 |

### 11.4 Scalars (Metrics)

| 그래프 | 의미 |
|--------|------|
| `Metrics/val_coarse_acc` | Coarse 분류 정확도 (Normal vs Defect) |
| `Metrics/val_fine_f1` | Fine 분류 F1 Score (5클래스) |
| `Metrics/val_final_f1` | 최종 F1 Score (Coarse→Fine 결합) |

### 11.5 Loss 가중치 설정

```yaml
criteria:
  coarse_weight: 1.0   # Coarse loss 가중치
  fine_weight: 1.0     # Fine loss 가중치
```

**튜닝 가이드:**
- Coarse 정확도가 낮으면 → `coarse_weight` 증가
- Fine 분류가 잘 안되면 → `fine_weight` 증가
- 보통 1:1 또는 0.5:1로 시작

### 11.6 CSV 로그 형식

```csv
epoch,train_loss,train_coarse_loss,train_fine_loss,val_loss,val_coarse_acc,val_fine_f1,val_final_f1,lr
1,1.234,0.456,0.778,1.123,0.85,0.42,0.45,0.0001
```

### 11.7 언제 사용하나?

| 상황 | 추천 모델 |
|------|----------|
| 5클래스 직접 분류 | 기본 CNN 또는 CNN+Metadata |
| Normal/Defect 먼저 분류 후 세부 분류 | **Hierarchical CNN** |
| Normal 분류 정확도가 중요 | **Hierarchical CNN** |

**장점:**
- 1단계에서 Normal을 잘 걸러내면 2단계 부담 감소
- Normal vs Defect 경계 명확하게 학습

**단점:**
- 1단계 오류가 2단계로 전파 (Coarse 실패 → Fine도 실패)
- 학습 복잡도 증가

---

## 12. 빠른 참조

### TensorBoard 단축키

| 단축키 | 기능 |
|--------|------|
| `R` | 새로고침 |
| `F` | 그래프 영역 맞춤 |
| 마우스 휠 | 확대/축소 |
| 드래그 | 영역 선택 확대 |

### 유용한 필터

Scalars 탭에서 검색창 활용:
- `Loss` - Loss 관련 그래프만
- `f1` - F1 관련 그래프만
- `resin` - resin_overflow 클래스만

---

**문서 끝**
