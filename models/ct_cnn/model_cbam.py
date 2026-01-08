"""ResNet18 + CBAM 모델 정의 (비교 실험용, 미채택)

성능 비교 결과:
- ResNet18 (기본): 99.2% Accuracy, 98.8% F1-Score
- ResNet18 + CBAM: 98.5% Accuracy, 97.9% F1-Score

결론: CBAM 추가 시 오히려 성능 하락 (과적합, 모델 복잡도 증가)
      기본 ResNet18 채택
"""
import torch
import torch.nn as nn
import torchvision.models as models


class CBAM(nn.Module):
    """
    CBAM: Convolutional Block Attention Module
    Channel Attention + Spatial Attention
    """

    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        """
        Args:
            channels: 입력 채널 수
            reduction: Channel Attention의 축소 비율
            kernel_size: Spatial Attention의 커널 크기
        """
        super(CBAM, self).__init__()

        # Channel Attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.channel_mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )

        # Spatial Attention
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(1)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Channel Attention
        avg_out = self.channel_mlp(self.avg_pool(x))
        max_out = self.channel_mlp(self.max_pool(x))
        channel_att = self.sigmoid(avg_out + max_out)
        x = x * channel_att

        # Spatial Attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.sigmoid(self.spatial_conv(torch.cat([avg_out, max_out], dim=1)))
        x = x * spatial_att

        return x


class ResNet18CBAM(nn.Module):
    """
    ResNet18 + CBAM Attention 모델 (비교 실험용)
    - layer3, layer4 뒤에 CBAM 추가
    - 작은 결함(기공) 탐지 성능 향상 목적으로 실험
    - 결과: 성능 하락으로 미채택
    """

    def __init__(self, num_classes: int = 5, pretrained: bool = True, dropout: float = 0.3):
        """
        Args:
            num_classes: 출력 클래스 수
            pretrained: ImageNet pretrained 사용 여부
            dropout: Dropout 비율
        """
        super(ResNet18CBAM, self).__init__()

        # ResNet18 로드
        resnet = models.resnet18(pretrained=pretrained)

        # ResNet18 구조 분해
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1  # 64 channels
        self.layer2 = resnet.layer2  # 128 channels
        self.layer3 = resnet.layer3  # 256 channels
        self.layer4 = resnet.layer4  # 512 channels

        # CBAM 모듈 추가 (layer3, layer4 뒤에)
        self.cbam3 = CBAM(256, reduction=16)
        self.cbam4 = CBAM(512, reduction=16)

        # Global Average Pooling
        self.avgpool = resnet.avgpool

        # Classifier
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(512, num_classes)

        print(f"✅ ResNet18+CBAM 모델 생성 완료")
        print(f"   - Pretrained: {pretrained}")
        print(f"   - Num classes: {num_classes}")
        print(f"   - CBAM: layer3 (256ch), layer4 (512ch)")
        print(f"   - Dropout: {dropout}")

    def forward(self, x):
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # ResNet blocks
        x = self.layer1(x)
        x = self.layer2(x)

        # layer3 + CBAM
        x = self.layer3(x)
        x = self.cbam3(x)

        # layer4 + CBAM
        x = self.layer4(x)
        x = self.cbam4(x)

        # Classifier
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        return x


# 사용 예시
if __name__ == "__main__":
    # CBAM 모델 생성
    model = ResNet18CBAM(num_classes=5, pretrained=True)

    # 테스트
    dummy_input = torch.randn(4, 3, 512, 512)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")  # (4, 5)
