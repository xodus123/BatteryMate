"""ResNet18 모델 정의 - CT 데이터 분류 (5클래스)

클래스: cell_normal, cell_porosity, module_normal, module_porosity, module_resin_overflow
"""
import torch
import torch.nn as nn
import torchvision.models as models


class ResNet18Classifier(nn.Module):
    """
    ResNet18 기반 분류 모델
    - ImageNet-1K pretrained 사용
    - 마지막 FC layer만 교체
    - Dropout 지원 (과적합 방지)
    """

    def __init__(self, num_classes: int = 5, pretrained: bool = True, dropout: float = 0.3):
        """
        Args:
            num_classes: 출력 클래스 수 (기본 5: cell_normal, cell_porosity, module_normal, module_porosity, module_resin_overflow)
            pretrained: ImageNet pretrained 사용 여부
            dropout: Dropout 비율 (0.0 ~ 1.0)
        """
        super(ResNet18Classifier, self).__init__()

        # ResNet18 로드
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.model = models.resnet18(weights=weights)

        # 마지막 FC layer를 Dropout + Linear로 교체
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes)
        )

        self.dropout_rate = dropout

        print(f"✅ ResNet18 모델 생성 완료")
        print(f"   - Pretrained: {pretrained}")
        print(f"   - Num classes: {num_classes}")
        print(f"   - Dropout: {dropout}")

    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W)

        Returns:
            logits: (B, num_classes)
        """
        return self.model(x)


def create_model(config: dict) -> nn.Module:
    """
    Config 기반 모델 생성

    Args:
        config: YAML config dict
            model:
                name: 모델 이름 (resnet18, efficientnet_b0, convnext_tiny 등)
                backbone: (optional) 'timm' 지정 시 timm 라이브러리 사용
                num_classes: 클래스 수
                pretrained: pretrained 사용 여부
                dropout: dropout 비율
                drop_path_rate: (optional) stochastic depth

    Returns:
        모델 인스턴스
    """
    model_name = config['model'].get('name', 'resnet18')
    backbone = config['model'].get('backbone', None)
    use_cbam = config['model'].get('use_cbam', False)

    # timm 라이브러리 사용 (EfficientNet, ConvNeXt 등)
    if backbone == 'timm' or model_name.startswith(('efficientnet', 'convnext', 'mobilenet')):
        from models.ct_cnn.model_timm import create_timm_model
        model = create_timm_model(config)

    # DRN+ASPP 모델
    elif model_name == 'drn_aspp':
        from models.ct_cnn.model_drn_aspp import create_drn_aspp_model
        model = create_drn_aspp_model(config)

    # DeepLabV3+ 사전학습 가중치 활용 모델
    elif model_name == 'deeplabv3':
        from models.ct_cnn.model_deeplabv3 import create_deeplabv3_model
        model = create_deeplabv3_model(config)

    # CBAM 모델 사용 시 별도 파일에서 import
    elif use_cbam or model_name == 'resnet18_cbam':
        from models.ct_cnn.model_cbam import ResNet18CBAM
        model = ResNet18CBAM(
            num_classes=config['model']['num_classes'],
            pretrained=config['model']['pretrained'],
            dropout=config['model'].get('dropout', 0.3)
        )

    # 기본 ResNet18
    else:
        model = ResNet18Classifier(
            num_classes=config['model']['num_classes'],
            pretrained=config['model']['pretrained'],
            dropout=config['model'].get('dropout', 0.3)
        )

    return model


# 사용 예시
if __name__ == "__main__":
    # 모델 생성
    model = ResNet18Classifier(num_classes=5, pretrained=True)

    # 테스트
    dummy_input = torch.randn(4, 3, 512, 512)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")  # (4, 5)
