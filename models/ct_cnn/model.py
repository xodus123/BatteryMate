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
    """

    def __init__(self, num_classes: int = 5, pretrained: bool = True):
        """
        Args:
            num_classes: 출력 클래스 수 (기본 5: cell_normal, cell_porosity, module_normal, module_porosity, module_resin_overflow)
            pretrained: ImageNet pretrained 사용 여부
        """
        super(ResNet18Classifier, self).__init__()

        # ResNet18 로드
        self.model = models.resnet18(pretrained=pretrained)

        # 마지막 FC layer 교체
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

        print(f"✅ ResNet18 모델 생성 완료")
        print(f"   - Pretrained: {pretrained}")
        print(f"   - Num classes: {num_classes}")

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

    Returns:
        모델 인스턴스
    """
    model_name = config['model'].get('name', 'resnet18')
    use_cbam = config['model'].get('use_cbam', False)

    # CBAM 모델 사용 시 별도 파일에서 import
    if use_cbam or model_name == 'resnet18_cbam':
        from models.ct_cnn.model_cbam import ResNet18CBAM
        model = ResNet18CBAM(
            num_classes=config['model']['num_classes'],
            pretrained=config['model']['pretrained'],
            dropout=config['model'].get('dropout', 0.3)
        )
    else:
        model = ResNet18Classifier(
            num_classes=config['model']['num_classes'],
            pretrained=config['model']['pretrained']
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
