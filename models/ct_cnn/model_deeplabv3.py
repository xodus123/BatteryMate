"""DeepLabV3+ 사전학습 가중치 활용 분류 모델

원본 DeepLabV3+ 세그멘테이션 모델(DRN-D-54 + ASPP, 4클래스)의
backbone + ASPP 사전학습 가중치를 활용하여 분류 태스크에 적용.

포팅 출처: D:\모델\1.모델소스코드\모델1_DeepLabv3\pytorch-deeplab-xception-eval\modeling\
  - backbone/drn.py → Bottleneck, DRN 클래스
  - aspp.py → _ASPPModule, ASPP 클래스
  - SynchronizedBatchNorm2d → nn.BatchNorm2d 교체 (가중치 형식 동일)

아키텍처:
  DRN-D-54 backbone (output_stride=8)
  → ASPP (512ch → 256ch, rates=[1,12,24,36])
  → Classification head: GAP → Dropout → FC(256→5)

핵심 가치: 배터리 CT 이미지로 학습된 DRN-D-54 backbone을 활용하므로,
ImageNet pretrained보다 나은 feature extraction 기대.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# DRN-D-54 Backbone (원본 drn.py 포팅)
# ============================================================

def conv3x3(in_planes, out_planes, stride=1, padding=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=padding, bias=False, dilation=dilation)


class Bottleneck(nn.Module):
    """원본 drn.py Bottleneck 그대로 포팅 (state_dict 키 호환)"""
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 dilation=(1, 1), residual=True, BatchNorm=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation[1], bias=False,
                               dilation=dilation[1])
        self.bn2 = BatchNorm(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class DRN(nn.Module):
    """원본 drn.py DRN 클래스 (arch='D' 전용으로 간소화)

    DRN-D-54 구조:
      layer0: Conv7x7 stride=1 → 16ch
      layer1: Conv3x3 → 16ch
      layer2: Conv3x3 stride=2 → 32ch
      layer3: 3×Bottleneck stride=2 → 256ch  ← low_level_feat
      layer4: 4×Bottleneck stride=2 → 512ch
      layer5: 6×Bottleneck dilation=2 → 1024ch
      layer6: 3×Bottleneck dilation=4 → 2048ch
      layer7: Conv3x3 dilation=2 → 512ch
      layer8: Conv3x3 dilation=1 → 512ch
    """

    def __init__(self, block, layers,
                 channels=(16, 32, 64, 128, 256, 512, 512, 512),
                 BatchNorm=None):
        super().__init__()
        self.inplanes = channels[0]
        self.out_dim = channels[-1]

        self.layer0 = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=7, stride=1, padding=3, bias=False),
            BatchNorm(channels[0]),
            nn.ReLU(inplace=True)
        )

        self.layer1 = self._make_conv_layers(
            channels[0], layers[0], stride=1, BatchNorm=BatchNorm)
        self.layer2 = self._make_conv_layers(
            channels[1], layers[1], stride=2, BatchNorm=BatchNorm)

        self.layer3 = self._make_layer(block, channels[2], layers[2], stride=2, BatchNorm=BatchNorm)
        self.layer4 = self._make_layer(block, channels[3], layers[3], stride=2, BatchNorm=BatchNorm)
        self.layer5 = self._make_layer(block, channels[4], layers[4],
                                       dilation=2, new_level=False, BatchNorm=BatchNorm)
        self.layer6 = None if layers[5] == 0 else \
            self._make_layer(block, channels[5], layers[5], dilation=4,
                             new_level=False, BatchNorm=BatchNorm)

        self.layer7 = None if layers[6] == 0 else \
            self._make_conv_layers(channels[6], layers[6], dilation=2, BatchNorm=BatchNorm)
        self.layer8 = None if layers[7] == 0 else \
            self._make_conv_layers(channels[7], layers[7], dilation=1, BatchNorm=BatchNorm)

        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1,
                    new_level=True, residual=True, BatchNorm=None):
        assert dilation == 1 or dilation % 2 == 0
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm(planes * block.expansion),
            )

        layers = list()
        layers.append(block(
            self.inplanes, planes, stride, downsample,
            dilation=(1, 1) if dilation == 1 else (
                dilation // 2 if new_level else dilation, dilation),
            residual=residual, BatchNorm=BatchNorm))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, residual=residual,
                                dilation=(dilation, dilation), BatchNorm=BatchNorm))

        return nn.Sequential(*layers)

    def _make_conv_layers(self, channels, convs, stride=1, dilation=1, BatchNorm=None):
        modules = []
        for i in range(convs):
            modules.extend([
                nn.Conv2d(self.inplanes, channels, kernel_size=3,
                          stride=stride if i == 0 else 1,
                          padding=dilation, bias=False, dilation=dilation),
                BatchNorm(channels),
                nn.ReLU(inplace=True)])
            self.inplanes = channels
        return nn.Sequential(*modules)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)

        x = self.layer3(x)
        low_level_feat = x

        x = self.layer4(x)
        x = self.layer5(x)

        if self.layer6 is not None:
            x = self.layer6(x)

        if self.layer7 is not None:
            x = self.layer7(x)

        if self.layer8 is not None:
            x = self.layer8(x)

        return x, low_level_feat


# ============================================================
# ASPP 모듈 (원본 aspp.py 포팅)
# ============================================================

class _ASPPModule(nn.Module):
    """원본 aspp.py _ASPPModule 그대로 포팅 (일반 Conv2d, depthwise 아님)"""

    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
        super().__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                     stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)
        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASPP(nn.Module):
    """원본 aspp.py ASPP 구조 그대로 포팅

    DRN backbone용 (inplanes=512, output_stride=8):
      aspp1: 1x1 conv (dilation=1) → 256ch
      aspp2: 3x3 conv (dilation=12) → 256ch
      aspp3: 3x3 conv (dilation=24) → 256ch
      aspp4: 3x3 conv (dilation=36) → 256ch
      global_avg_pool: GAP → 1x1 conv → 256ch
      → concat 1280ch → 1x1 conv → 256ch
    """

    def __init__(self, backbone, output_stride, BatchNorm):
        super().__init__()
        if backbone == 'drn':
            inplanes = 512
        else:
            inplanes = 2048

        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = _ASPPModule(inplanes, 256, 1, padding=0, dilation=dilations[0], BatchNorm=BatchNorm)
        self.aspp2 = _ASPPModule(inplanes, 256, 3, padding=dilations[1], dilation=dilations[1], BatchNorm=BatchNorm)
        self.aspp3 = _ASPPModule(inplanes, 256, 3, padding=dilations[2], dilation=dilations[2], BatchNorm=BatchNorm)
        self.aspp4 = _ASPPModule(inplanes, 256, 3, padding=dilations[3], dilation=dilations[3], BatchNorm=BatchNorm)

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(inplanes, 256, 1, stride=1, bias=False),
            BatchNorm(256),
            nn.ReLU()
        )
        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = BatchNorm(256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


# ============================================================
# DeepLabV3 분류 모델
# ============================================================

class DeepLabV3Classifier(nn.Module):
    """DeepLabV3+ 사전학습 가중치 활용 분류 모델

    원본 세그멘테이션 모델의 backbone(DRN-D-54) + ASPP 가중치를 로드하고,
    decoder를 GAP + FC 분류 헤드로 교체.

    state_dict 키 매핑:
      self.backbone → 원본 'backbone.*' 키
      self.aspp → 원본 'aspp.*' 키
      self.classifier → 새로 학습 (decoder 키 스킵)
    """

    def __init__(self, num_classes: int = 5, dropout: float = 0.5,
                 freeze_backbone: bool = True,
                 pretrained_segmentation: str = None):
        super().__init__()

        BatchNorm = nn.BatchNorm2d

        # DRN-D-54 backbone (원본과 동일 구조)
        self.backbone = DRN(
            block=Bottleneck,
            layers=[1, 1, 3, 4, 6, 3, 1, 1],
            channels=(16, 32, 64, 128, 256, 512, 512, 512),
            BatchNorm=BatchNorm
        )

        # ASPP (DRN backbone, output_stride=8)
        self.aspp = ASPP(backbone='drn', output_stride=8, BatchNorm=BatchNorm)

        # 분류 헤드 (원본 decoder 대체)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes)
        )

        # 사전학습 가중치 로드
        if pretrained_segmentation:
            self.load_pretrained_segmentation(pretrained_segmentation)

        # backbone + ASPP freeze
        if freeze_backbone:
            self._freeze_backbone_aspp()

        self._print_info(num_classes, dropout, freeze_backbone, pretrained_segmentation)

    def load_pretrained_segmentation(self, checkpoint_path: str):
        """원본 세그멘테이션 모델에서 backbone + ASPP 가중치만 로드 (decoder 스킵)"""
        print(f"📦 사전학습 가중치 로드 중: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # backbone + aspp 키만 필터링 (decoder 키 제외)
        loaded_keys = []
        skipped_keys = []
        for key, value in state_dict.items():
            if key.startswith('backbone.') or key.startswith('aspp.'):
                loaded_keys.append(key)
            else:
                skipped_keys.append(key)

        filtered_state_dict = {k: v for k, v in state_dict.items() if k in loaded_keys}

        # strict=False: classifier 키는 체크포인트에 없으므로
        missing, unexpected = self.load_state_dict(filtered_state_dict, strict=False)

        # classifier 키만 missing이어야 정상
        classifier_missing = [k for k in missing if k.startswith('classifier.')]
        other_missing = [k for k in missing if not k.startswith('classifier.')]

        print(f"  ✅ 로드된 키: {len(loaded_keys)}개 (backbone + aspp)")
        print(f"  ⏭️  스킵된 키: {len(skipped_keys)}개 (decoder)")
        if other_missing:
            print(f"  ⚠️  매칭 실패: {other_missing}")
        print(f"  🆕 새로 학습할 키: {len(classifier_missing)}개 (classifier)")

    def _freeze_backbone_aspp(self):
        """backbone + ASPP 파라미터 freeze + eval 모드 고정"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.aspp.parameters():
            param.requires_grad = False
        self.backbone.eval()
        self.aspp.eval()
        self._frozen = True

    def train(self, mode=True):
        """frozen 모듈은 항상 eval 모드 유지 (BatchNorm 안정성)"""
        super().train(mode)
        if getattr(self, '_frozen', False) and mode:
            self.backbone.eval()
            self.aspp.eval()
        return self

    def _print_info(self, num_classes, dropout, freeze_backbone, pretrained_segmentation):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"✅ DeepLabV3 분류 모델 생성 완료")
        print(f"   - Backbone: DRN-D-54 (output_stride=8)")
        print(f"   - ASPP: rates=[1,12,24,36]")
        print(f"   - Num classes: {num_classes}")
        print(f"   - Dropout: {dropout}")
        print(f"   - Freeze backbone+ASPP: {freeze_backbone}")
        print(f"   - Pretrained: {'세그멘테이션 가중치' if pretrained_segmentation else '없음'}")
        print(f"   - Parameters: {trainable / 1e6:.2f}M trainable / {total / 1e6:.2f}M total")

    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W)
        Returns:
            logits: (B, num_classes)
        """
        x, _ = self.backbone(x)  # low_level_feat 사용 안 함
        x = self.aspp(x)         # (B, 256, H/8, W/8)
        return self.classifier(x)


def create_deeplabv3_model(config: dict) -> nn.Module:
    """Config 기반 DeepLabV3 분류 모델 생성"""
    model_cfg = config['model']
    return DeepLabV3Classifier(
        num_classes=model_cfg.get('num_classes', 5),
        dropout=model_cfg.get('dropout', 0.5),
        freeze_backbone=model_cfg.get('freeze_backbone', True),
        pretrained_segmentation=model_cfg.get('pretrained_segmentation', None),
    )


# 단독 실행 테스트
if __name__ == "__main__":
    import os

    print("=" * 60)
    print("DeepLabV3 분류 모델 테스트")
    print("=" * 60)

    # 1. 사전학습 가중치 없이 생성
    print("\n[1] 기본 모델 생성 (가중치 없음)")
    model = DeepLabV3Classifier(num_classes=5, freeze_backbone=False)

    dummy = torch.randn(2, 3, 512, 512)
    out = model(dummy)
    print(f"  Output shape: {out.shape}")  # (2, 5)
    assert out.shape == (2, 5), f"Expected (2, 5), got {out.shape}"

    # 2. 사전학습 가중치 로드 테스트
    ckpt_path = "models/ct_cnn/checkpoints/deeplabv3_drn_ct.pt"
    if os.path.exists(ckpt_path):
        print(f"\n[2] 사전학습 가중치 로드 테스트")
        model2 = DeepLabV3Classifier(
            num_classes=5,
            freeze_backbone=True,
            pretrained_segmentation=ckpt_path
        )
        out2 = model2(dummy)
        print(f"  Output shape: {out2.shape}")
        assert out2.shape == (2, 5)
        print("\n✅ 모든 테스트 통과!")
    else:
        print(f"\n⚠️  체크포인트 미발견: {ckpt_path}")
        print("  기본 모델 테스트만 통과")
