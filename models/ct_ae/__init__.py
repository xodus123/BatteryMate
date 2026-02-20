"""CT AutoEncoder 모듈

CT 이미지 기반 Anomaly Detection
- 정상 패턴 학습 후 결함 이미지 입력 시 높은 재구성 오류 발생
- CNN+Metadata와 앙상블하여 사용
"""
from models.rgb_ae.model import ConvAutoEncoder, create_model

__all__ = ['ConvAutoEncoder', 'create_model']
