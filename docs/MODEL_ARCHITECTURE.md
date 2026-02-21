# ResNet18 모델 구조

> **작성일**: 2024-12-29
> **모델**: ResNet18 (배터리 불량 검사용)
> **입력 크기**: 512x512x3
> **출력**: 2 classes (정상/불량)

---

## 📊 전체 흐름도

```
입력 이미지 (3, 512, 512)
    ↓
[Conv1] 7x7 conv, 64 filters, stride=2
    → (64, 256, 256)
    ↓
[BatchNorm + ReLU + MaxPool]
    → (64, 128, 128)
    ↓
[Layer1] 2개 BasicBlock (64 channels)
    → (64, 128, 128)
    ↓
[Layer2] 2개 BasicBlock (128 channels, stride=2)
    → (128, 64, 64)
    ↓
[Layer3] 2개 BasicBlock (256 channels, stride=2)
    → (256, 32, 32)
    ↓
[Layer4] 2개 BasicBlock (512 channels, stride=2)
    → (512, 16, 16)
    ↓
[Global Average Pooling]
    → (512, 1, 1)
    ↓
[Flatten]
    → (512)
    ↓
[Fully Connected] 512 → 2
    → (2)  [정상, 불량]
```

---

## 🏗️ 레이어별 상세 구조

### 1. 초기 레이어

| 레이어 | 파라미터 | 입력 크기 | 출력 크기 |
|--------|---------|-----------|-----------|
| **Conv1** | kernel=7x7, filters=64, stride=2, padding=3 | (3, 512, 512) | (64, 256, 256) |
| **BatchNorm1** | - | (64, 256, 256) | (64, 256, 256) |
| **ReLU** | - | (64, 256, 256) | (64, 256, 256) |
| **MaxPool** | kernel=3x3, stride=2, padding=1 | (64, 256, 256) | (64, 128, 128) |

### 2. Residual Layers

#### Layer1 (64 channels)
- **BasicBlock x 2**
- 입력: (64, 128, 128)
- 출력: (64, 128, 128)
- 파라미터: ~147K

#### Layer2 (128 channels)
- **BasicBlock x 2**
- 입력: (64, 128, 128)
- 출력: (128, 64, 64)
- Stride=2 (첫 번째 블록)
- 파라미터: ~525K

#### Layer3 (256 channels)
- **BasicBlock x 2**
- 입력: (128, 64, 64)
- 출력: (256, 32, 32)
- Stride=2 (첫 번째 블록)
- 파라미터: ~2.1M

#### Layer4 (512 channels)
- **BasicBlock x 2**
- 입력: (256, 32, 32)
- 출력: (512, 16, 16)
- Stride=2 (첫 번째 블록)
- 파라미터: ~8.4M

### 3. 출력 레이어

| 레이어 | 파라미터 | 입력 크기 | 출력 크기 |
|--------|---------|-----------|-----------|
| **AdaptiveAvgPool** | output_size=(1, 1) | (512, 16, 16) | (512, 1, 1) |
| **Flatten** | - | (512, 1, 1) | (512) |
| **FC (Fully Connected)** | in=512, out=2 | (512) | (2) |

---

## 🔍 BasicBlock 구조

ResNet18의 기본 빌딩 블록:

```python
class BasicBlock:
    def forward(x):
        identity = x  # 입력 저장 (Residual Connection)

        # 첫 번째 Conv Block
        out = Conv2d(x)         # 3x3 convolution
        out = BatchNorm2d(out)
        out = ReLU(out)

        # 두 번째 Conv Block
        out = Conv2d(out)       # 3x3 convolution
        out = BatchNorm2d(out)

        # Residual Connection
        out = out + identity    # Skip connection
        out = ReLU(out)

        return out
```

**핵심 원리:**
- 입력을 출력에 직접 더함 (Shortcut/Skip Connection)
- 기울기 소실(Gradient Vanishing) 문제 해결
- 깊은 네트워크 학습 가능

---

## 📈 모델 통계

### 전체 파라미터

| 항목 | 값 |
|------|-----|
| **총 레이어 수** | 18개 (Convolutional 레이어 기준) |
| **총 파라미터** | 11,177,538개 (약 1,120만개) |
| **학습 가능 파라미터** | 11,177,538개 (전체) |
| **모델 크기 (float32)** | ~43 MB |
| **모델 크기 (체크포인트)** | ~129 MB (optimizer state 포함) |

### 레이어별 파라미터 분포

```
Conv1 + BN1:        ~10K   (0.1%)
Layer1:            ~147K   (1.3%)
Layer2:            ~525K   (4.7%)
Layer3:           ~2.1M    (18.8%)
Layer4:           ~8.4M    (75.2%)
FC:                 ~1K    (0.01%)
─────────────────────────────────
Total:           ~11.2M   (100%)
```

**특징:**
- Layer4가 전체 파라미터의 75%를 차지
- 깊은 레이어일수록 파라미터 수가 많음
- 고수준 특징(high-level features) 추출에 집중

---

## 🎯 현재 설정

### 학습 설정 (`training/configs/cnn.yaml`)

```yaml
model:
  name: resnet18
  pretrained: true          # ImageNet-1K pretrained
  num_classes: 2            # 정상/불량

data:
  image_size: 512           # 512x512 입력
  batch_size: 32
  num_workers: 4

training:
  optimizer: Adam
  lr: 0.0001
  weight_decay: 0.0001
  epochs: 30

criteria:
  early_stopping:
    patience: 5
    monitor: val_loss
```

---

## ✅ 장점

### 1. ImageNet Pretrained 사용
- 1,400만 장의 이미지로 사전 학습된 가중치
- Transfer Learning으로 빠른 수렴
- 일반적인 특징(에지, 텍스처 등) 이미 학습됨

### 2. 전체 레이어 학습 (Fine-tuning)
- 모든 11M 파라미터가 학습됨
- 배터리 CT 데이터에 완전히 최적화 가능
- 저수준 특징부터 고수준 특징까지 조정

### 3. 적절한 모델 크기
- ResNet18: 너무 깊지도 얕지도 않음
- CT 데이터 ~5,000장에 적합
- 과적합 위험 낮음

### 4. Residual Connection
- 깊은 네트워크에서도 안정적 학습
- 기울기 소실 문제 해결
- 더 복잡한 패턴 학습 가능

---

## 🔄 각 레이어의 역할

### Conv1 + Layer1 (초기 레이어)
**학습하는 것:**
- 에지 (edge)
- 코너 (corner)
- 기본 텍스처 (texture)

**배터리 검사에서:**
- 표면 거칠기
- 경계선
- 기본 패턴

### Layer2 + Layer3 (중간 레이어)
**학습하는 것:**
- 복합 패턴
- 형태 (shape)
- 부분적 객체

**배터리 검사에서:**
- 크랙 패턴
- 불규칙한 영역
- 이물질 형태

### Layer4 + FC (깊은 레이어)
**학습하는 것:**
- 고수준 의미 (semantic)
- 전체적인 맥락
- 클래스 구분 특징

**배터리 검사에서:**
- "이게 정상인가 불량인가"
- 결함의 심각도
- 종합적 판단

---

## 📊 입력 → 출력 변환 과정

### 공간 해상도 변화

```
512x512  (입력 이미지)
   ↓ Conv1 (stride=2)
256x256
   ↓ MaxPool (stride=2)
128x128
   ↓ Layer1 (stride=1)
128x128
   ↓ Layer2 (stride=2)
64x64
   ↓ Layer3 (stride=2)
32x32
   ↓ Layer4 (stride=2)
16x16
   ↓ Global AvgPool
1x1
```

### 채널 수 변화

```
3      (RGB 입력)
  ↓
64     (Conv1)
  ↓
64     (Layer1)
  ↓
128    (Layer2)
  ↓
256    (Layer3)
  ↓
512    (Layer4)
  ↓
2      (FC - 정상/불량)
```

**특징:**
- 공간 해상도 ↓ (512 → 1)
- 채널 수 ↑ (3 → 512)
- "상세한 위치 정보" → "추상적인 의미 정보"

---

## 🎨 TensorBoard에서 확인하기

학습 시작 후 TensorBoard에 모델 구조 그래프가 자동으로 추가됩니다.

**확인 방법:**
1. 학습 시작: `python models/ct_cnn/train.py`
2. TensorBoard 접속: `http://localhost:6006`
3. **GRAPHS** 탭 클릭
4. 시각적으로 모델 구조 확인 가능

---

## 📚 참고 자료

### ResNet 논문
- **제목**: "Deep Residual Learning for Image Recognition"
- **저자**: Kaiming He et al. (Microsoft Research)
- **발표**: CVPR 2016
- **핵심 아이디어**: Residual Connection (Skip Connection)

### PyTorch 공식 구현
```python
import torchvision.models as models
model = models.resnet18(pretrained=True)
```

### 코드 위치
- 모델 정의: `models/ct_cnn/model.py`
- 학습 스크립트: `models/ct_cnn/train.py`
- 설정 파일: `training/configs/cnn.yaml`

---

## 🔧 모델 수정 가이드

### ResNet50으로 변경하려면:

**1. Config 수정** (`training/configs/cnn.yaml`):
```yaml
model:
  name: resnet50  # resnet18 → resnet50
```

**2. 모델 파일 수정** (`models/ct_cnn/model.py`):
```python
self.model = models.resnet50(pretrained=pretrained)  # resnet18 → resnet50
```

**차이점:**
- ResNet18: 11M 파라미터, 18 레이어
- ResNet50: 25M 파라미터, 50 레이어
- ResNet50이 더 강력하지만 학습 시간 증가

---

## 🚀 성능 개선 전략

### 📌 기본 원칙

**⚠️ 중요: 먼저 현재 구조로 베이스라인 성능을 확보하세요!**

성능 개선은 다음 순서로 진행:
1. ✅ **현재 ResNet18로 학습** → 베이스라인 성능 측정
2. 📊 **결과 분석** → 어떤 문제가 있는지 파악
3. 🔧 **타겟 개선** → 문제에 맞는 해결책 적용

---

### 1️⃣ 모델 크기 증가 (성능 ↑, 속도 ↓)

#### Option A: ResNet50
**언제 사용:**
- 베이스라인 성능이 부족할 때
- 데이터가 충분할 때 (5,000장 이상 ✅)
- GPU 메모리가 충분할 때

**변경 방법:**

```yaml
# training/configs/cnn.yaml
model:
  name: resnet50  # resnet18 → resnet50
  pretrained: true
  num_classes: 2
```

```python
# models/ct_cnn/model.py (해당 라인 수정)
self.model = models.resnet50(pretrained=pretrained)
```

**비교:**

| 모델 | 파라미터 | 레이어 | 학습 시간 | 성능 |
|------|---------|--------|----------|------|
| ResNet18 | 11M | 18 | 기준 | 기준 |
| ResNet50 | 25M | 50 | ~2배 | +2~5% |
| ResNet101 | 44M | 101 | ~3배 | +3~7% |

**주의사항:**
- Batch size를 줄여야 할 수 있음 (32 → 16)
- 학습 시간 증가
- 과적합 위험 (데이터 부족 시)

---

#### Option B: EfficientNet (효율적)
**장점:**
- ResNet50보다 적은 파라미터로 높은 성능
- 메모리 효율적

**변경 방법:**

```python
# models/ct_cnn/model.py
import timm

class EfficientNetClassifier(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        self.model = timm.create_model(
            'efficientnet_b0',
            pretrained=pretrained,
            num_classes=num_classes
        )
```

**비교:**

| 모델 | 파라미터 | 성능 | 추론 속도 |
|------|---------|------|----------|
| ResNet18 | 11M | 기준 | 빠름 |
| EfficientNet-B0 | 5.3M | +3~5% | 보통 |
| EfficientNet-B1 | 7.8M | +4~7% | 보통 |

---

### 2️⃣ Feature Map vs Vector 이해

#### 현재 구조 (GAP 사용 - 올바름!)

```python
Layer4 출력: (512, 16, 16)  ← Feature Map (공간 정보 O)
    ↓
GAP (Global Average Pooling)
    ↓
Vector: (512)               ← 공간 정보 X
    ↓
FC: (512) → (2)
```

**Feature Map (2D):**
- 크기: (채널, 높이, 너비) = (512, 16, 16)
- 공간 정보 유지: "왼쪽 위에 크랙이 있다"
- 용도: Object Detection, Segmentation

**Vector (1D):**
- 크기: (채널) = (512)
- 공간 정보 없음: "크랙이 있다" (위치 모름)
- 용도: Classification (현재 작업!)

**왜 GAP가 좋은가:**

```python
# GAP 없이 Flatten만 사용하면:
Layer4: (512, 16, 16)
    ↓
Flatten: (512 × 16 × 16) = (131,072)  ← 너무 큼!
    ↓
FC: (131,072) → (2)  ← 파라미터 262,144개 (비효율!)

# GAP 사용 (현재):
Layer4: (512, 16, 16)
    ↓
GAP: 각 채널의 16×16 값을 평균 → (512)
    ↓
FC: (512) → (2)  ← 파라미터 1,024개 (효율적!)
```

**결론:**
- ✅ 현재 ResNet18은 이미 GAP 사용 중
- ✅ Classification에는 GAP가 최적
- ✅ 변경 불필요!

---

### 3️⃣ Attention 메커니즘 추가 (고급)

**언제 사용:**
- 베이스라인 성능이 80% 이상일 때
- 더 정교한 특징 추출이 필요할 때
- "어디를 봐야 하는지" 학습시키고 싶을 때

#### SE (Squeeze-and-Excitation) Block

```python
# models/ct_cnn/model.py
class SEBlock(nn.Module):
    """채널 간 중요도를 학습"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        # x: (B, C, H, W)
        b, c, _, _ = x.size()

        # Global Average Pooling
        y = x.view(b, c, -1).mean(dim=2)  # (B, C)

        # Channel attention
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))

        # Re-weight channels
        y = y.view(b, c, 1, 1)
        return x * y.expand_as(x)


class ResNetWithSE(nn.Module):
    """ResNet + SE Block"""
    def __init__(self, num_classes=2):
        super().__init__()
        self.resnet = models.resnet18(pretrained=True)

        # SE Block을 Layer4 뒤에 추가
        self.se = SEBlock(512)

        # FC layer 교체
        self.resnet.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        # SE Block 적용
        x = self.se(x)  # ← 중요한 채널에 가중치

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.resnet.fc(x)

        return x
```

**효과:**
- 중요한 특징에 집중
- 약 +1~3% 성능 향상
- 파라미터 증가 거의 없음 (~0.1M)

---

### 4️⃣ Multi-Scale Features (고급)

**개념:**
- Layer2, Layer3, Layer4 출력을 모두 사용
- 저수준 + 중수준 + 고수준 특징 결합

```python
class MultiScaleResNet(nn.Module):
    """여러 레이어의 특징을 결합"""
    def __init__(self, num_classes=2):
        super().__init__()
        resnet = models.resnet18(pretrained=True)

        # Backbone
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1  # (64, H/4, W/4)
        self.layer2 = resnet.layer2  # (128, H/8, W/8)
        self.layer3 = resnet.layer3  # (256, H/16, W/16)
        self.layer4 = resnet.layer4  # (512, H/32, W/32)

        # 각 레이어별 GAP
        self.gap = nn.AdaptiveAvgPool2d(1)

        # 결합 후 FC
        self.fc = nn.Linear(128 + 256 + 512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)

        # Layer2, 3, 4 특징 추출
        x2 = self.layer2(x)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        # 각각 GAP
        f2 = self.gap(x2).flatten(1)  # (B, 128)
        f3 = self.gap(x3).flatten(1)  # (B, 256)
        f4 = self.gap(x4).flatten(1)  # (B, 512)

        # Concatenate
        features = torch.cat([f2, f3, f4], dim=1)  # (B, 896)

        # Classification
        out = self.fc(features)
        return out
```

**효과:**
- 다양한 스케일의 결함 감지
- 미세한 크랙 + 큰 변형 동시 탐지
- 약 +2~4% 성능 향상

---

### 5️⃣ Data Augmentation (가장 먼저 시도!)

**코스트 제로 성능 향상!**

```python
# training/data/transforms.py
from torchvision import transforms

def get_train_transforms(image_size=512):
    """학습용 Augmentation"""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),

        # 기본 Augmentation
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),

        # 고급 Augmentation
        transforms.ColorJitter(
            brightness=0.2,  # 밝기 변화
            contrast=0.2,    # 대비 변화
            saturation=0.1   # 채도 변화
        ),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),  # 이동
            scale=(0.9, 1.1)       # 크기 변화
        ),

        # Normalize
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
```

**효과:**
- 데이터 다양성 증가
- 과적합 방지
- 약 +3~7% 성능 향상
- **추가 비용 없음!**

---

### 6️⃣ Learning Rate Scheduling

**현재 문제:**
- 고정 Learning Rate (0.0001)
- 학습 후반부에 미세 조정 어려움

**해결책: Cosine Annealing**

```python
# models/ct_cnn/train.py
from torch.optim.lr_scheduler import CosineAnnealingLR

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)  # 초기 LR 높임

# Scheduler
scheduler = CosineAnnealingLR(
    optimizer,
    T_max=config['training']['epochs'],
    eta_min=1e-6  # 최소 LR
)

# 학습 루프에서
for epoch in range(epochs):
    train_loss = train_epoch()
    val_loss, val_metrics = validate()

    # Scheduler step
    scheduler.step()
```

**효과:**
- 초반: 빠르게 수렴
- 후반: 미세 조정
- 약 +1~3% 성능 향상

---

### 📊 개선 전략 우선순위

#### 🥇 1순위 (먼저 시도)
1. **Data Augmentation** - 비용 없이 성능 ↑
2. **Learning Rate Scheduling** - 간단한 코드 추가
3. **Batch Size / Image Size 조정** - Config만 변경

#### 🥈 2순위 (베이스라인 70% 이상일 때)
4. **ResNet50으로 변경** - 모델 크기 증가
5. **EfficientNet 시도** - 효율적인 모델

#### 🥉 3순위 (베이스라인 80% 이상일 때)
6. **SE Block 추가** - Attention 메커니즘
7. **Multi-Scale Features** - 복잡한 구조 변경

---

### ⚠️ 주의사항

**하지 말아야 할 것:**
- ❌ 한 번에 여러 개선 동시 적용 → 어떤 게 효과적인지 모름
- ❌ 베이스라인 없이 복잡한 모델부터 시작
- ❌ 데이터 분석 없이 무작정 모델만 변경

**해야 할 것:**
- ✅ 한 번에 하나씩 변경
- ✅ 각 변경의 효과 측정
- ✅ 실험 결과 기록 (Config + 성능)

---

### 📈 실험 로그 예시

```markdown
## 실험 기록

### Baseline (ResNet18)
- Config: image_size=512, lr=0.0001, bs=32
- 결과: F1=0.78, Acc=0.80
- 문제: Recall이 낮음 (불량 미탐 많음)

### Experiment 1: Data Augmentation
- 변경: RandomFlip, Rotation, ColorJitter 추가
- 결과: F1=0.82 (+4%), Acc=0.83
- 효과: ✅ 과적합 감소, Recall 개선

### Experiment 2: ResNet50
- 변경: ResNet18 → ResNet50
- 결과: F1=0.84 (+2%), Acc=0.85
- 효과: ✅ 미세한 결함 탐지 개선
- 비용: 학습 시간 2배 증가

### Experiment 3: SE Block
- 변경: Layer4 뒤에 SE Block 추가
- 결과: F1=0.85 (+1%), Acc=0.86
- 효과: ✅ 중요한 특징에 집중
- 비용: 파라미터 거의 증가 없음
```

---

**문서 작성일**: 2024-12-29
**최종 수정일**: 2024-12-29
**모델 버전**: ResNet18 (ImageNet Pretrained)
**프로젝트**: 배터리 불량 검사 시스템
