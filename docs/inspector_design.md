# 통합 검사 구조 설계 (CNN + AutoEncoder)

> **작성일**: 2026-01-02
> **목적**: CT CNN과 RGB AutoEncoder를 결합하여 최종 불량 판정 1개 도출

---

## 🎯 통합 검사 목표

**CNN 예측 확률 + AE 이상 점수 → 가중 평균/투표 → 최종 결정 (defect or normal)**

### 핵심 전략
- CT CNN: 내부 구조 분석 (X-ray)
- RGB AutoEncoder: 외관 이상 탐지
- **두 모델의 장점을 결합하여 정확도 향상**

---

## 📊 통합 검사 파이프라인

```
[배터리 이미지]
    ↓
┌─────────────────────────────────┐
│  데이터 분리                     │
│  - CT 이미지                     │
│  - RGB 이미지                    │
└─────────────────────────────────┘
    ↓
┌─────────────────┬───────────────────┐
│  CT CNN         │  RGB AutoEncoder  │
│  (ResNet18)     │  (CAE)            │
└─────────────────┴───────────────────┘
    ↓               ↓
  확률: 0.85      이상 점수: 0.72
    ↓               ↓
┌─────────────────────────────────┐
│  점수 정규화                     │
│  - CNN: 0~1 (이미 정규화됨)     │
│  - AE: 0~1 범위로 변환          │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│  통합 검사 레이어                   │
│  ┌─────────────────────────┐   │
│  │ 방법 1: 가중 평균        │   │
│  │ 방법 2: 투표             │   │
│  │ 방법 3: 규칙 기반        │   │
│  └─────────────────────────┘   │
└─────────────────────────────────┘
    ↓
**최종 판정**: defect (0.78) or normal (0.22)
```

---

## 🔢 통합 검사 방법

### 방법 1: 가중 평균 (Weighted Average) ⭐ 기본

```python
final_score = w_cnn * cnn_prob + w_ae * ae_score

# 기본 가중치
w_cnn = 0.6  # CT CNN 가중치
w_ae = 0.4   # AutoEncoder 가중치

# 최종 판정
if final_score >= threshold:
    prediction = "defect"
else:
    prediction = "normal"
```

**장점**:
- 간단하고 해석 가능
- 가중치 조정으로 모델 기여도 제어

**가중치 설정 전략**:
- 초기값: 0.6 (CNN) / 0.4 (AE)
- Validation Set에서 Grid Search로 최적화
- 예: `[(0.5, 0.5), (0.6, 0.4), (0.7, 0.3)]`

---

### 방법 2: 투표 (Voting)

```python
# Hard Voting
cnn_pred = "defect" if cnn_prob >= 0.5 else "normal"
ae_pred = "defect" if ae_score >= ae_threshold else "normal"

if cnn_pred == ae_pred:
    final_pred = cnn_pred
else:
    # 불일치 시 CNN 우선 (또는 AE 우선, 설정 가능)
    final_pred = cnn_pred
```

**장점**:
- 단순 명확
- 모델 간 합의 확인 가능

**단점**:
- 확률 정보 손실
- 불일치 시 처리 규칙 필요

---

### 방법 3: 규칙 기반 (Rule-Based)

```python
# 예시: 두 모델 모두 높은 확신도일 때만 불량 판정
if cnn_prob >= 0.8 and ae_score >= ae_threshold * 1.2:
    final_pred = "defect"
elif cnn_prob <= 0.3 and ae_score <= ae_threshold * 0.8:
    final_pred = "normal"
else:
    # 중간 영역: 가중 평균 사용
    final_score = w_cnn * cnn_prob + w_ae * ae_score
    final_pred = "defect" if final_score >= 0.5 else "normal"
```

**장점**:
- 도메인 지식 반영 가능
- 확신도 낮은 경우 별도 처리

**단점**:
- 복잡도 증가
- 규칙 설계 필요

---

## 🎛️ 설정 파일 구조

```yaml
# training/configs/inspector.yaml

inspector:
  method: "weighted_average"  # weighted_average | voting | rule_based

  weighted_average:
    w_cnn: 0.6                # CNN 가중치
    w_ae: 0.4                 # AutoEncoder 가중치
    threshold: 0.5            # 최종 판정 임계값

  voting:
    cnn_threshold: 0.5
    ae_threshold_multiplier: 1.0  # ae_threshold * multiplier
    tie_breaker: "cnn"        # cnn | ae | uncertain

  rule_based:
    high_confidence_cnn: 0.8
    low_confidence_cnn: 0.3
    ae_multiplier_high: 1.2
    ae_multiplier_low: 0.8
    fallback_method: "weighted_average"

models:
  cnn:
    checkpoint: "experiments/checkpoints/cnn/resnet18_best.pt"
    config: "training/configs/cnn.yaml"

  autoencoder:
    checkpoint: "experiments/checkpoints/autoencoder/ae_rgb.pt"
    threshold_file: "experiments/checkpoints/autoencoder/ae_rgb_threshold.json"
    config: "training/configs/autoencoder_rgb.yaml"
```

---

## 📦 코드 구조

```
backend/app/
├── models/
│   ├── cnn/
│   │   └── predictor.py         # CNN 추론
│   ├── autoencoder/
│   │   └── predictor.py         # AE 추론
│   └── inspector/                # ⭐ 새로 추가
│       ├── __init__.py
│       ├── inspector.py          # 통합 검사 메인 로직
│       ├── weighted_avg.py      # 가중 평균 구현
│       ├── voting.py            # 투표 구현
│       └── rule_based.py        # 규칙 기반 구현
│
├── core/
│   └── pipeline.py              # 수정: 통합 검사 통합
│
└── schemas/
    └── response.py              # 수정: 통합 검사 결과 스키마

training/
├── configs/
│   └── inspector.yaml            # ⭐ 통합 검사 설정
│
└── evaluation/
    └── inspector_optimizer.py    # ⭐ 가중치 최적화 스크립트
```

---

## 🔧 구현 예시

### Inspector 클래스

```python
# backend/app/models/inspector/inspector.py

from typing import Dict, Tuple
from app.models.cnn.predictor import CNNPredictor
from app.models.autoencoder.predictor import AEPredictor
import yaml

class InspectorPredictor:
    """CNN + AutoEncoder 통합 검사"""

    def __init__(self, config_path: str):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        # 모델 초기화
        self.cnn = CNNPredictor(
            checkpoint=self.config['models']['cnn']['checkpoint'],
            config=self.config['models']['cnn']['config']
        )
        self.ae = AEPredictor(
            checkpoint=self.config['models']['autoencoder']['checkpoint'],
            threshold_file=self.config['models']['autoencoder']['threshold_file']
        )

        # 통합 검사 설정
        self.method = self.config['inspector']['method']
        self.inspector_config = self.config['inspector'][self.method]

    def predict(self, ct_image: str, rgb_image: str) -> Dict:
        """
        통합 검사 예측

        Args:
            ct_image: CT 이미지 경로
            rgb_image: RGB 이미지 경로

        Returns:
            {
                "prediction": "defect" or "normal",
                "confidence": 0.78,
                "cnn": {"prob": 0.85, "pred": "defect"},
                "ae": {"score": 0.72, "pred": "defect"},
                "method": "weighted_average"
            }
        """
        # 1. 개별 모델 추론
        cnn_result = self.cnn.predict(ct_image)
        ae_result = self.ae.predict(rgb_image)

        # 2. 점수 추출
        cnn_prob = cnn_result['probability']  # 0~1
        ae_score = ae_result['normalized_score']  # 0~1로 정규화됨

        # 3. 통합 검사 결합
        if self.method == "weighted_average":
            final_pred, final_conf = self._weighted_average(cnn_prob, ae_score)
        elif self.method == "voting":
            final_pred, final_conf = self._voting(cnn_prob, ae_score, ae_result['threshold'])
        elif self.method == "rule_based":
            final_pred, final_conf = self._rule_based(cnn_prob, ae_score, ae_result['threshold'])
        else:
            raise ValueError(f"Unknown inspector method: {self.method}")

        return {
            "prediction": final_pred,
            "confidence": final_conf,
            "cnn": {
                "probability": cnn_prob,
                "prediction": "defect" if cnn_prob >= 0.5 else "normal"
            },
            "ae": {
                "score": ae_score,
                "threshold": ae_result['threshold'],
                "prediction": ae_result['prediction']
            },
            "method": self.method
        }

    def _weighted_average(self, cnn_prob: float, ae_score: float) -> Tuple[str, float]:
        """가중 평균"""
        w_cnn = self.inspector_config['w_cnn']
        w_ae = self.inspector_config['w_ae']
        threshold = self.inspector_config['threshold']

        final_score = w_cnn * cnn_prob + w_ae * ae_score
        prediction = "defect" if final_score >= threshold else "normal"

        return prediction, final_score

    def _voting(self, cnn_prob: float, ae_score: float, ae_threshold: float) -> Tuple[str, float]:
        """투표"""
        cnn_pred = "defect" if cnn_prob >= self.inspector_config['cnn_threshold'] else "normal"
        ae_pred = "defect" if ae_score >= ae_threshold * self.inspector_config['ae_threshold_multiplier'] else "normal"

        if cnn_pred == ae_pred:
            confidence = (cnn_prob + ae_score) / 2
            return cnn_pred, confidence
        else:
            # Tie breaker
            if self.inspector_config['tie_breaker'] == "cnn":
                return cnn_pred, cnn_prob
            elif self.inspector_config['tie_breaker'] == "ae":
                return ae_pred, ae_score
            else:
                return "uncertain", 0.5

    def _rule_based(self, cnn_prob: float, ae_score: float, ae_threshold: float) -> Tuple[str, float]:
        """규칙 기반"""
        high_cnn = self.inspector_config['high_confidence_cnn']
        low_cnn = self.inspector_config['low_confidence_cnn']
        ae_high = ae_threshold * self.inspector_config['ae_multiplier_high']
        ae_low = ae_threshold * self.inspector_config['ae_multiplier_low']

        # 높은 확신도: 둘 다 불량
        if cnn_prob >= high_cnn and ae_score >= ae_high:
            return "defect", max(cnn_prob, ae_score)

        # 높은 확신도: 둘 다 정상
        elif cnn_prob <= low_cnn and ae_score <= ae_low:
            return "normal", 1 - max(cnn_prob, ae_score)

        # 중간 영역: Fallback
        else:
            return self._weighted_average(cnn_prob, ae_score)
```

---

## 📈 가중치 최적화

### Grid Search 스크립트

```python
# training/evaluation/inspector_optimizer.py

import yaml
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from app.models.inspector.inspector import InspectorPredictor

def optimize_weights(val_dataset, config_path):
    """
    Validation Set에서 최적 가중치 탐색

    Args:
        val_dataset: (ct_images, rgb_images, labels)
        config_path: 통합 검사 설정 파일

    Returns:
        best_weights: (w_cnn, w_ae)
        best_threshold: float
        best_f1: float
    """
    ct_images, rgb_images, labels = val_dataset

    # 탐색 범위
    weight_candidates = [
        (0.5, 0.5),
        (0.6, 0.4),
        (0.7, 0.3),
        (0.8, 0.2),
        (0.4, 0.6),
    ]
    threshold_candidates = np.linspace(0.3, 0.7, 9)

    best_f1 = 0
    best_config = None

    for w_cnn, w_ae in weight_candidates:
        for threshold in threshold_candidates:
            # Config 임시 수정
            with open(config_path) as f:
                config = yaml.safe_load(f)
            config['inspector']['weighted_average']['w_cnn'] = w_cnn
            config['inspector']['weighted_average']['w_ae'] = w_ae
            config['inspector']['weighted_average']['threshold'] = threshold

            # 저장
            temp_config = "/tmp/inspector_temp.yaml"
            with open(temp_config, 'w') as f:
                yaml.dump(config, f)

            # 예측
            inspector = InspectorPredictor(temp_config)
            predictions = []

            for ct_img, rgb_img in zip(ct_images, rgb_images):
                result = inspector.predict(ct_img, rgb_img)
                pred_label = 1 if result['prediction'] == "defect" else 0
                predictions.append(pred_label)

            # 평가
            f1 = f1_score(labels, predictions)

            if f1 > best_f1:
                best_f1 = f1
                best_config = {
                    'w_cnn': w_cnn,
                    'w_ae': w_ae,
                    'threshold': threshold,
                    'f1': f1
                }

    print(f"Best Config: {best_config}")
    return best_config

# 실행 예시
if __name__ == "__main__":
    from training.data.dataloader import load_val_dataset

    val_data = load_val_dataset()
    best = optimize_weights(val_data, "training/configs/inspector.yaml")

    # 최적 설정 저장
    with open("experiments/inspector_best_config.json", 'w') as f:
        json.dump(best, f, indent=2)
```

---

## 🎯 실행 흐름

### 1. 학습 단계
```bash
# CT CNN 학습
python models/ct_cnn/train.py

# RGB AutoEncoder 학습
python models/rgb_ae/train.py

# 통합 검사 가중치 최적화
python training/evaluation/inspector_optimizer.py
```

### 2. 추론 단계
```bash
# FastAPI 서버 실행
uvicorn backend.app.main:app --reload

# API 호출 예시
curl -X POST "http://localhost:8000/infer" \
  -F "ct_image=@battery_ct_001.jpg" \
  -F "rgb_image=@battery_rgb_001.jpg"
```

### 3. 결과 예시
```json
{
  "prediction": "defect",
  "confidence": 0.78,
  "cnn": {
    "probability": 0.85,
    "prediction": "defect"
  },
  "ae": {
    "score": 0.72,
    "threshold": 0.65,
    "prediction": "defect"
  },
  "method": "weighted_average",
  "weights": {
    "w_cnn": 0.6,
    "w_ae": 0.4
  }
}
```

---

## ✅ 장점

1. **정확도 향상**: 두 모델의 장점 결합
2. **해석 가능성**: 개별 모델 결과도 함께 제공
3. **유연성**: 가중치/방법 조정 가능
4. **안정성**: 한 모델이 틀려도 다른 모델이 보완

---

## ⚠️ 주의사항

1. **데이터 매칭**: CT와 RGB 이미지가 같은 배터리인지 확인 필요
2. **점수 정규화**: AE 이상 점수를 0~1 범위로 정규화 필수
3. **Threshold 관리**: AE Threshold 파일을 체크포인트와 함께 저장
4. **가중치 최적화**: Validation Set에서 Grid Search 수행

---

## 📝 다음 단계

1. ✅ 통합 검사 설계 완료
2. ⏳ `inspector.yaml` 작성
3. ⏳ `InspectorPredictor` 구현
4. ⏳ FastAPI에 통합 검사 엔드포인트 추가
5. ⏳ 가중치 최적화 스크립트 실행
6. ⏳ Web UI에 통합 검사 결과 표시

---

**작성일**: 2026-01-02
**상태**: 설계 완료, 구현 대기
