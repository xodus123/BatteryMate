"""DRN + ASPP 분류 모델 - DeepLabV3+ 스타일 다중 스케일 특징 추출

CT 이미지에서 다양한 스케일의 결함 패턴(기공, 수지 오버플로우 등)을 포착하기 위해
dilated convolution + ASPP로 넓은 receptive field를 확보하되,
분류 태스크에 맞게 GAP → FC로 마무리.

아키텍처:
  ResNet50 backbone (dilated, output_stride=16)
  → Low-level features (layer1, 256ch → 48ch)
  → ASPP (layer4, 2048ch → 256ch, rates=[6,12,18])
  → Feature combination (ASPP + low-level → 256ch)
  → GAP → Dropout → FC(256→5)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class DepthwiseSeparableConv(nn.Module):
    """Depthwise Separable Convolution: depthwise 3x3 + pointwise 1x1 + BN + ReLU"""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, padding: int = 1, dilation: int = 1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size,
            padding=padding, dilation=dilation, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)


class ASPPModule(nn.Module):
    """ASPP의 단일 atrous conv 브랜치 (depthwise separable)"""

    def __init__(self, in_channels: int, out_channels: int, rate: int):
        super().__init__()
        self.conv = DepthwiseSeparableConv(
            in_channels, out_channels,
            kernel_size=3, padding=rate, dilation=rate
        )

    def forward(self, x):
        return self.conv(x)


class ASPPPooling(nn.Module):
    """ASPP의 image-level pooling 브랜치"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        # BN 제외: pooling 후 1x1 spatial에서 BN은 batch_size=1일 때 실패
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        size = x.shape[2:]
        x = self.pool(x)
        x = self.conv(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling - 5개 브랜치 통합

    브랜치:
      1. 1x1 conv
      2~4. 3x3 depthwise sep conv (rate=6, 12, 18)
      5. Image-level pooling (GAP → 1x1 conv → upsample)
    → Concat → 1x1 projection
    """

    def __init__(self, in_channels: int, out_channels: int = 256,
                 rates: list = None):
        super().__init__()
        if rates is None:
            rates = [6, 12, 18]

        # 1x1 conv 브랜치
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # Atrous conv 브랜치들
        self.atrous_convs = nn.ModuleList([
            ASPPModule(in_channels, out_channels, rate) for rate in rates
        ])

        # Image-level pooling 브랜치
        self.image_pooling = ASPPPooling(in_channels, out_channels)

        # Projection: concat → 1x1 conv
        concat_channels = out_channels * (1 + len(rates) + 1)  # 1x1 + atrous + pooling
        self.project = nn.Sequential(
            nn.Conv2d(concat_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        features = [self.conv1x1(x)]
        for atrous_conv in self.atrous_convs:
            features.append(atrous_conv(x))
        features.append(self.image_pooling(x))

        x = torch.cat(features, dim=1)
        return self.project(x)


class DRNASPPClassifier(nn.Module):
    """DRN + ASPP 기반 분류 모델

    ResNet50 backbone (dilated, output_stride=16)
    → Low-level features (layer1) + ASPP (layer4)
    → Feature combination → GAP → FC
    """

    def __init__(self, num_classes: int = 5, pretrained: bool = True,
                 dropout: float = 0.5, aspp_rates: list = None,
                 freeze_stem: bool = True):
        super().__init__()

        if aspp_rates is None:
            aspp_rates = [6, 12, 18]

        # ResNet50 backbone (output_stride=16: layer4에서 dilation 적용)
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        resnet = models.resnet50(
            weights=weights,
            replace_stride_with_dilation=[False, False, True]
        )

        # Backbone 분해
        self.stem = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
        )
        self.layer1 = resnet.layer1  # 256ch, stride 4 (128x128 for 512 input)
        self.layer2 = resnet.layer2  # 512ch, stride 8
        self.layer3 = resnet.layer3  # 1024ch, stride 16
        self.layer4 = resnet.layer4  # 2048ch, stride 16 (dilated)

        # stem + layer1 freeze (과적합 방지)
        if freeze_stem:
            for param in self.stem.parameters():
                param.requires_grad = False
            for param in self.layer1.parameters():
                param.requires_grad = False

        # Low-level feature 축소 (layer1: 256ch → 48ch)
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(256, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )

        # ASPP (layer4 출력: 2048ch → 256ch)
        self.aspp = ASPP(2048, 256, rates=aspp_rates)

        # Feature combination: ASPP(256) + low-level(48) = 304ch → 256ch
        self.combine_conv = nn.Sequential(
            DepthwiseSeparableConv(304, 256, kernel_size=3, padding=1),
            DepthwiseSeparableConv(256, 256, kernel_size=3, padding=1),
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes)
        )

        self._print_info(num_classes, pretrained, dropout, aspp_rates, freeze_stem)

    def _print_info(self, num_classes, pretrained, dropout, aspp_rates, freeze_stem):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"✅ DRN+ASPP 모델 생성 완료")
        print(f"   - Backbone: ResNet50 (output_stride=16)")
        print(f"   - Pretrained: {pretrained}")
        print(f"   - Num classes: {num_classes}")
        print(f"   - ASPP rates: {aspp_rates}")
        print(f"   - Dropout: {dropout}")
        print(f"   - Freeze stem+layer1: {freeze_stem}")
        print(f"   - Parameters: {trainable / 1e6:.1f}M trainable / {total / 1e6:.1f}M total")

    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W)
        Returns:
            logits: (B, num_classes)
        """
        # Backbone
        x = self.stem(x)
        low_level = self.layer1(x)  # (B, 256, H/4, W/4)
        x = self.layer2(low_level)
        x = self.layer3(x)
        x = self.layer4(x)  # (B, 2048, H/16, W/16)

        # ASPP
        aspp_out = self.aspp(x)  # (B, 256, H/16, W/16)

        # Low-level feature 처리
        low_level = self.low_level_conv(low_level)  # (B, 48, H/4, W/4)

        # ASPP 출력을 low-level 크기로 upsample 후 concat
        aspp_up = F.interpolate(
            aspp_out, size=low_level.shape[2:],
            mode='bilinear', align_corners=False
        )  # (B, 256, H/4, W/4)
        combined = torch.cat([aspp_up, low_level], dim=1)  # (B, 304, H/4, W/4)

        # Feature combination
        combined = self.combine_conv(combined)  # (B, 256, H/4, W/4)

        # Classification
        return self.classifier(combined)


def create_drn_aspp_model(config: dict) -> nn.Module:
    """Config 기반 DRN+ASPP 모델 생성"""
    model_cfg = config['model']
    return DRNASPPClassifier(
        num_classes=model_cfg.get('num_classes', 5),
        pretrained=model_cfg.get('pretrained', True),
        dropout=model_cfg.get('dropout', 0.5),
        aspp_rates=model_cfg.get('aspp_rates', [6, 12, 18]),
        freeze_stem=model_cfg.get('freeze_stem', True),
    )


if __name__ == "__main__":
    model = DRNASPPClassifier(num_classes=5, pretrained=True)

    dummy_input = torch.randn(2, 3, 512, 512)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")  # (2, 5)
