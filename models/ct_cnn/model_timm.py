"""timm 기반 CNN 모델 (EfficientNet, ConvNeXt 등)

timm 라이브러리를 사용하여 다양한 최신 CNN 아키텍처 지원
- EfficientNet-B0/B1/B2
- ConvNeXt-Tiny/Small
- EfficientNetV2-S
- 등등
"""
import torch
import torch.nn as nn

try:
    import timm
except ImportError:
    raise ImportError("timm 라이브러리가 필요합니다: pip install timm")


class TimmClassifier(nn.Module):
    """
    timm 라이브러리 기반 분류 모델

    지원 모델:
    - efficientnet_b0, efficientnet_b1, efficientnet_b2
    - convnext_tiny, convnext_small
    - efficientnetv2_s
    - resnet18, resnet34, resnet50 (timm 버전)
    """

    def __init__(
        self,
        model_name: str = 'efficientnet_b0',
        num_classes: int = 5,
        pretrained: bool = True,
        dropout: float = 0.3,
        drop_path_rate: float = 0.0
    ):
        """
        Args:
            model_name: timm 모델 이름
            num_classes: 출력 클래스 수
            pretrained: ImageNet pretrained 사용 여부
            dropout: Classifier dropout 비율
            drop_path_rate: Stochastic depth (ConvNeXt 등에서 사용)
        """
        super(TimmClassifier, self).__init__()

        # timm 모델 생성
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=dropout,
            drop_path_rate=drop_path_rate
        )

        self.model_name = model_name
        self.num_classes = num_classes
        self.dropout_rate = dropout

        # 파라미터 수 계산
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print(f"✅ {model_name} 모델 생성 완료")
        print(f"   - Pretrained: {pretrained}")
        print(f"   - Num classes: {num_classes}")
        print(f"   - Dropout: {dropout}")
        print(f"   - Drop path rate: {drop_path_rate}")
        print(f"   - Total params: {total_params:,}")
        print(f"   - Trainable params: {trainable_params:,}")

    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W)

        Returns:
            logits: (B, num_classes)
        """
        return self.model(x)

    def get_features(self, x):
        """
        특징 추출 (Grad-CAM 등에 사용)

        Args:
            x: (B, 3, H, W)

        Returns:
            features: 마지막 conv layer 출력
        """
        return self.model.forward_features(x)


class EfficientNetClassifier(TimmClassifier):
    """EfficientNet 분류 모델"""

    def __init__(
        self,
        variant: str = 'b0',
        num_classes: int = 5,
        pretrained: bool = True,
        dropout: float = 0.3
    ):
        """
        Args:
            variant: EfficientNet 변형 (b0, b1, b2, v2_s 등)
            num_classes: 출력 클래스 수
            pretrained: ImageNet pretrained 사용 여부
            dropout: Dropout 비율
        """
        if variant.startswith('v2'):
            model_name = f'efficientnetv2_{variant[3:]}'
        else:
            model_name = f'efficientnet_{variant}'

        super().__init__(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout
        )


class ConvNeXtClassifier(TimmClassifier):
    """ConvNeXt 분류 모델"""

    def __init__(
        self,
        variant: str = 'tiny',
        num_classes: int = 5,
        pretrained: bool = True,
        dropout: float = 0.3,
        drop_path_rate: float = 0.1
    ):
        """
        Args:
            variant: ConvNeXt 변형 (tiny, small, base, large)
            num_classes: 출력 클래스 수
            pretrained: ImageNet pretrained 사용 여부
            dropout: Dropout 비율
            drop_path_rate: Stochastic depth 비율
        """
        model_name = f'convnext_{variant}'

        super().__init__(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout,
            drop_path_rate=drop_path_rate
        )


def create_timm_model(config: dict) -> nn.Module:
    """
    Config 기반 timm 모델 생성

    Args:
        config: YAML config dict
            model:
                name: 모델 이름 (efficientnet_b0, convnext_tiny 등)
                num_classes: 클래스 수
                pretrained: pretrained 사용 여부
                dropout: dropout 비율
                drop_path_rate: (optional) stochastic depth

    Returns:
        모델 인스턴스
    """
    model_config = config['model']
    model_name = model_config.get('name', 'efficientnet_b0')
    num_classes = model_config.get('num_classes', 5)
    pretrained = model_config.get('pretrained', True)
    dropout = model_config.get('dropout', 0.3)
    drop_path_rate = model_config.get('drop_path_rate', 0.0)

    return TimmClassifier(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
        drop_path_rate=drop_path_rate
    )


def list_available_models():
    """사용 가능한 모델 목록 출력"""
    recommended = [
        # EfficientNet
        'efficientnet_b0',      # 5.3M params
        'efficientnet_b1',      # 7.8M params
        'efficientnet_b2',      # 9.2M params
        'efficientnetv2_s',     # 21M params
        # ConvNeXt
        'convnext_tiny',        # 28M params
        'convnext_small',       # 50M params
        # 기타
        'resnet18',             # 11M params (timm 버전)
        'resnet34',             # 21M params
        'mobilenetv3_large_100', # 5.4M params
    ]

    print("권장 모델 목록:")
    for model in recommended:
        try:
            m = timm.create_model(model, pretrained=False)
            params = sum(p.numel() for p in m.parameters())
            print(f"  - {model}: {params/1e6:.1f}M params")
            del m
        except Exception as e:
            print(f"  - {model}: (로드 실패: {e})")


# 사용 예시
if __name__ == "__main__":
    print("=" * 60)
    list_available_models()
    print("=" * 60)

    # EfficientNet-B0 테스트
    print("\nEfficientNet-B0 테스트:")
    model = EfficientNetClassifier(variant='b0', num_classes=5)
    dummy_input = torch.randn(2, 3, 1024, 1024)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")

    # ConvNeXt-Tiny 테스트
    print("\nConvNeXt-Tiny 테스트:")
    model = ConvNeXtClassifier(variant='tiny', num_classes=5)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
