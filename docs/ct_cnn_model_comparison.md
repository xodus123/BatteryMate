# CT CNN 모델 비교 분석

> 작성일: 2026-01-26
> 목적: 배터리 CT 이미지 5클래스 분류 모델 성능 비교

---

## 1. 실험 개요

### 1.1 클래스 정의
| 클래스 ID | 클래스명 | 설명 |
|-----------|----------|------|
| 0 | cell_normal | Cell 정상 |
| 1 | cell_porosity | Cell 기공 결함 |
| 2 | module_normal | Module 정상 |
| 3 | module_porosity | Module 기공 결함 |
| 4 | module_resin_overflow | Module 레진 넘침 |

### 1.2 데이터 분포 (Test Set 기준)
| 클래스 | 샘플 수 | 비율 |
|--------|---------|------|
| cell_normal | 8,461 | 23.2% |
| cell_porosity | 6,095 | 16.7% |
| module_normal | 11,191 | 30.7% |
| module_porosity | 10,650 | 29.2% |
| module_resin_overflow | 27 | 0.07% |
| **합계** | **36,424** | 100% |

### 1.3 전처리 방식

| 전처리 | 설명 | 출력 크기 |
|--------|------|-----------|
| **1차 전처리** | battery_outline bbox 기준 crop (워터마크/프레임 제거) | 1024x1024 |
| **2차 전처리** | defect bbox 기준 crop (결함 영역 확대) | 512x512 |

---

## 2. 모델별 성능 비교

### 2.1 Validation 성능 요약

| 모델 | 전처리 | Best Val F1 (macro) | Best Val Acc | Early Stop Epoch | 비고 |
|------|--------|---------------------|--------------|------------------|------|
| **Basic CNN (ResNet18)** | 2차 | **0.8369** | 0.8684 | 2 | Resin 워터마크 문제 |
| **Basic CNN (ResNet18)** | 1차 | 0.7979 | 0.9116 | 2 | 증강 수정 후 |
| **CBAM CNN** | 2차 | 0.6221 | 0.8225 | 10 | 심한 과적합 |
| **Hierarchical CNN** | 1차 | 0.6303 | - | 1 | Coarse/Fine 2단계 |
| **Metadata Fusion** | 1차 | *(학습 중)* | - | - | bbox 정보 결합 |

### 2.2 학습 곡선 분석

#### Basic CNN (1차 전처리) - 최신 실험
```
Epoch  Train Loss  Val Loss  Val F1    Val Acc   LR
1      0.3090      0.2060    0.7962    0.9116    0.000050
2      0.2255      0.4439    0.7979    0.8822    0.000049  ← Best F1
3      0.2045      0.4031    0.7834    0.8755    0.000045
4      0.1957      0.6420    0.7579    0.8623    0.000040
5      0.1853      0.8927    0.7579    0.8356    0.000033
6      0.1774      0.5161    0.7523    0.8679    0.000026  ← Early Stop
```
- **특징**: Val loss 증가하며 과적합 시작, Early stopping 작동

#### Basic CNN (2차 전처리) - defect crop
```
Epoch  Train Loss  Val Loss  Val F1    Val Acc   LR
1      0.3276      0.2607    0.8159    0.8389    0.000050
2      0.2363      0.2682    0.8369    0.8684    0.000049  ← Best F1
3      0.2140      0.3502    0.8080    0.8371    0.000045
4      0.2030      0.4088    0.8038    0.8504    0.000040
5      0.1965      0.4842    0.7894    0.8297    0.000033
6      0.1841      0.6024    0.8161    0.8526    0.000026  ← Early Stop
```
- **특징**: F1 높으나 Resin 워터마크로 인한 spurious correlation 의심

#### CBAM CNN (2차 전처리)
```
Epoch  Train Loss  Val Loss  Val F1    Val Acc
1      0.2072      1.4229    0.5476    0.6659
...
10     0.0350      1.6517    0.6221    0.8225  ← Best F1
14     0.0504      1.5858    0.6186    0.8304  ← Early Stop
```
- **특징**: Train loss 매우 낮으나 Val loss 높음 → 심각한 과적합

#### Hierarchical CNN
```
Epoch  Train Loss  Coarse Acc  Fine F1   Final F1
1      0.4916      0.8371      1.0000    0.6303  ← Best
8      0.3024      0.8544      0.9655    0.5692  ← Early Stop
```
- **특징**: Fine F1 ~100%지만 Coarse 분류에서 Normal/Defect 혼동

---

## 3. 클래스별 상세 분석 (Test Set)

### 3.1 Basic CNN (1차 전처리) 클래스별 성능

| 클래스 | Precision | Recall | F1 Score | Support |
|--------|-----------|--------|----------|---------|
| cell_normal | 0.600 | 0.744 | 0.665 | 8,461 |
| cell_porosity | 0.471 | 0.316 | **0.378** | 6,095 |
| module_normal | 0.957 | 0.843 | 0.896 | 11,191 |
| module_porosity | 0.857 | 0.960 | 0.906 | 10,650 |
| module_resin_overflow | 0.466 | 1.000 | 0.635 | 27 |
| **Macro Avg** | **0.670** | **0.773** | **0.696** | 36,424 |

### 3.2 Confusion Matrix 분석

```
                  Predicted
              cell_n  cell_p  mod_n  mod_p  resin
Actual
cell_normal    6294    2167      0      0      0
cell_porosity  4169    1926      0      0      0    ← 68% FN!
module_normal    19       0   9434   1707     31
module_porosity   0       0    426  10224      0
module_resin      0       0      0      0     27
```

### 3.3 주요 문제점

1. **cell_porosity 저성능 (F1: 0.378)**
   - 68%가 cell_normal로 오분류
   - Cell 내 기공과 정상의 시각적 차이 미미

2. **cell_normal ↔ cell_porosity 혼동**
   - cell_normal의 26%가 cell_porosity로 오분류
   - 클래스 경계가 모호함

3. **module_resin_overflow 극소 샘플**
   - 27개 샘플로 학습 불안정
   - Recall 100%지만 FP 많음 (Precision 47%)

---

## 4. 전처리별 비교

| 항목 | 1차 전처리 | 2차 전처리 |
|------|------------|------------|
| 이미지 크기 | 1024x1024 | 512x512 |
| 내용 | 배터리 전체 영역 | 결함 bbox 영역만 |
| 장점 | 배터리 유형(cell/module) 구분 용이 | 결함 특징에 집중 |
| 단점 | 결함 영역이 상대적으로 작음 | 정상 이미지 처리 불가 |
| Val F1 | 0.7979 | 0.8369 |
| **Resin 문제** | 없음 | 워터마크/프레임 학습 |

### 4.1 Resin 워터마크 문제

2차 전처리에서 Resin 클래스가 F1 0.99를 기록했으나, 이는 **워터마크/프레임 패턴**을 학습한 것으로 분석됨:
- Resin 이미지에만 특정 워터마크가 존재
- 모델이 결함이 아닌 워터마크로 분류 학습
- 1차 전처리로 워터마크 영역 제거 후 정상 학습

---

## 5. 증강(Augmentation) 설정

### 5.1 CT 이미지 특성
- **밝기 = 물질 밀도**: ColorJitter 부적합
- **미세 결함**: GaussianBlur 부적합 (특징 흐려짐)

### 5.2 최종 증강 설정

| 모델 | Flip | Rotation | Affine | ColorJitter | Blur |
|------|------|----------|--------|-------------|------|
| Basic CNN | O | O | O | X | X |
| CBAM CNN | O | O | O | X | X |
| Hierarchical | O | O | O | X | X |
| **Metadata** | **X** | **X** | **X** | **X** | **X** |

- Metadata 모델: bbox 좌표가 메타데이터로 입력되므로 기하 변환도 제외

---

## 6. 모델 아키텍처 비교

| 모델 | Base | 추가 구성 | 파라미터 수 |
|------|------|-----------|-------------|
| Basic CNN | ResNet18 | Dropout FC | ~11.2M |
| CBAM CNN | ResNet18 | CBAM (layer3,4) | ~11.4M |
| Hierarchical | ResNet18 | Coarse + Fine Head | ~11.4M |
| Metadata Fusion | ResNet18 | Metadata Encoder + Fusion | ~11.8M |

---

## 7. 결론 및 권장사항

### 7.1 현재 Best 모델
- **1차 전처리 + Basic CNN (ResNet18)**
- Val F1 Macro: 0.7979
- Test F1 Macro: 0.696

### 7.2 주요 개선 필요 사항
1. **cell_porosity 성능 개선** (F1 0.378 → 목표 0.6+)
   - 더 세밀한 특징 학습 필요
   - Attention 메커니즘 또는 더 높은 해상도 고려

2. **module_resin_overflow 데이터 증강**
   - 27개 → 최소 500개 이상 필요
   - 또는 Few-shot learning 접근

3. **Metadata Fusion 실험 완료**
   - bbox 정보가 분류에 도움되는지 확인

### 7.3 다음 실험 계획
- [ ] Metadata Fusion 모델 학습 완료
- [ ] cell_porosity 전용 세부 분류기 검토
- [ ] 더 높은 해상도 (1024 → 1536) 실험
- [ ] Test set 상세 오류 분석

---

## 부록: 실험 환경

| 항목 | 값 |
|------|-----|
| GPU | NVIDIA (CUDA) |
| Framework | PyTorch 2.x |
| Optimizer | AdamW |
| Learning Rate | 5e-5 |
| Weight Decay | 0.03 |
| Batch Size | 16 (1024px) / 32 (512px) |
| Scheduler | CosineAnnealingWarmRestarts |
| Early Stopping | patience=4~5 |
| Loss | Focal Loss (gamma=1.5) + Label Smoothing (0.07) |
