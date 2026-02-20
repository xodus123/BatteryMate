"""ResNet + Late Fusion Model

이미지 특징 추출 후 메타데이터를 마지막에 결합
- 1단계: 이미지만으로 특징 학습 (메타데이터 영향 없음)
- 2단계: 분류 직전에 메타데이터 concat (학습 안 됨, 힌트로만 사용)

vs Early Fusion (기존):
- 메타데이터가 인코더를 통해 학습됨 → 메타데이터 의존도 높음
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional


class ResNetLateFusion(nn.Module):
    """ResNet + Late Fusion Model

    메타데이터를 인코더 없이 raw로 마지막에 concat
    - 이미지: ResNet18 → 512차원
    - 메타데이터: raw 2차원 (battery_type, axis)
    - 최종: [512 + 2] = 514차원 → 분류
    """

    METADATA_DIM = 2  # battery_type + axis

    def __init__(
        self,
        num_classes: int = 5,
        pretrained: bool = True,
        dropout: float = 0.5,
        freeze_backbone: bool = False
    ):
        """
        Args:
            num_classes: 출력 클래스 수
            pretrained: ImageNet pretrained 사용 여부
            dropout: Dropout 비율
            freeze_backbone: 백본 가중치 고정 여부 (2단계 학습용)
        """
        super().__init__()

        self.num_classes = num_classes
        self.freeze_backbone = freeze_backbone

        # ResNet18 백본
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.resnet18(weights=weights)

        # ResNet fc 제거 (특징만 추출)
        self.image_feature_dim = self.backbone.fc.in_features  # 512
        self.backbone.fc = nn.Identity()

        # Backbone freeze 옵션
        if freeze_backbone:
            self._freeze_backbone()

        # 분류기 (이미지 512 + 메타데이터 2 = 514)
        classifier_input_dim = self.image_feature_dim + self.METADATA_DIM

        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, num_classes)
        )

        # 가중치 초기화
        self._init_classifier_weights()

        print(f"ResNet18 Late Fusion 모델 생성")
        print(f"  - Pretrained: {pretrained}")
        print(f"  - Backbone Frozen: {freeze_backbone}")
        print(f"  - Image features: {self.image_feature_dim}")
        print(f"  - Metadata (raw): {self.METADATA_DIM}")
        print(f"  - Classifier input: {classifier_input_dim}")
        print(f"  - Num classes: {num_classes}")

    def _freeze_backbone(self):
        """백본 가중치 고정"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("  - Backbone weights frozen")

    def unfreeze_backbone(self):
        """백본 가중치 해제 (Fine-tuning용)"""
        for param in self.backbone.parameters():
            param.requires_grad = True
        self.freeze_backbone = False
        print("Backbone weights unfrozen for fine-tuning")

    def _init_classifier_weights(self):
        """분류기 가중치 초기화"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(
        self,
        image: torch.Tensor,
        metadata: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            image: (B, 3, H, W) 이미지 텐서
            metadata: (B, 2) 메타데이터 텐서 [battery_type, axis]
                      - battery_type: 0 (cell) or 1 (module)
                      - axis: 0 (x), 1 (y), 2 (z)

        Returns:
            (B, num_classes) 로짓
        """
        # 이미지 특징 추출
        image_features = self.backbone(image)  # (B, 512)

        # 메타데이터 처리
        if metadata is not None:
            # 메타데이터 정규화 (0~1 또는 0~2 범위를 유지)
            # 이미 0/1 또는 0/1/2 값이므로 그대로 사용
            meta_normalized = metadata
        else:
            # 메타데이터 없으면 기본값 (cell, x축)
            batch_size = image_features.size(0)
            device = image_features.device
            meta_normalized = torch.zeros(batch_size, self.METADATA_DIM, device=device)

        # Late Fusion: 마지막에 concat
        fused = torch.cat([image_features, meta_normalized], dim=1)  # (B, 514)

        # 분류
        logits = self.classifier(fused)

        return logits

    def forward_image_only(self, image: torch.Tensor) -> torch.Tensor:
        """이미지만으로 특징 추출 (디버깅/분석용)"""
        return self.backbone(image)

    def get_feature_dim(self) -> int:
        """전체 특징 차원 반환"""
        return self.image_feature_dim + self.METADATA_DIM


class ResNetImageOnly(nn.Module):
    """이미지만 사용하는 모델 (1단계 학습용)

    Late Fusion의 1단계: 순수 이미지만으로 학습
    나중에 가중치를 Late Fusion 모델로 전이
    """

    def __init__(
        self,
        num_classes: int = 5,
        pretrained: bool = True,
        dropout: float = 0.5
    ):
        super().__init__()

        self.num_classes = num_classes

        # ResNet18 백본
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.resnet18(weights=weights)

        # fc 교체
        self.image_feature_dim = self.backbone.fc.in_features  # 512
        self.backbone.fc = nn.Sequential(
            nn.Linear(self.image_feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, num_classes)
        )

        print(f"ResNet18 Image-Only 모델 생성")
        print(f"  - Pretrained: {pretrained}")
        print(f"  - Num classes: {num_classes}")

    def forward(self, image: torch.Tensor, metadata: Optional[torch.Tensor] = None) -> torch.Tensor:
        """metadata 인자는 호환성을 위해 받지만 무시"""
        return self.backbone(image)

    def get_backbone_state_dict(self):
        """백본 가중치만 추출 (Late Fusion으로 전이용)"""
        # fc 레이어 제외하고 백본만 추출
        state_dict = {}
        for name, param in self.backbone.named_parameters():
            if not name.startswith('fc'):
                state_dict[name] = param.data.clone()
        return state_dict


def create_late_fusion_model(
    num_classes: int = 5,
    pretrained: bool = True,
    dropout: float = 0.5,
    freeze_backbone: bool = False
) -> ResNetLateFusion:
    """Late Fusion 모델 생성 헬퍼"""
    return ResNetLateFusion(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
        freeze_backbone=freeze_backbone
    )


def create_image_only_model(
    num_classes: int = 5,
    pretrained: bool = True,
    dropout: float = 0.5
) -> ResNetImageOnly:
    """Image-Only 모델 생성 헬퍼"""
    return ResNetImageOnly(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout
    )


if __name__ == "__main__":
    print("=" * 60)
    print("Late Fusion Model Test")
    print("=" * 60)

    # Late Fusion 모델 테스트
    model = create_late_fusion_model(num_classes=5, pretrained=True)

    # 파라미터 수
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")

    # Forward 테스트
    batch_size = 4
    image = torch.randn(batch_size, 3, 512, 512)
    metadata = torch.tensor([
        [0.0, 0.0],  # cell, x
        [0.0, 1.0],  # cell, y
        [1.0, 0.0],  # module, x
        [1.0, 2.0],  # module, z
    ])

    output = model(image, metadata)
    print(f"\nInput image shape: {image.shape}")
    print(f"Input metadata shape: {metadata.shape}")
    print(f"Output shape: {output.shape}")

    # 메타데이터 없이 테스트
    output_no_meta = model(image, None)
    print(f"Output (no metadata) shape: {output_no_meta.shape}")

    print("\n" + "=" * 60)
    print("Image-Only Model Test")
    print("=" * 60)

    # Image-Only 모델 테스트
    model_img = create_image_only_model(num_classes=5, pretrained=True)
    output_img = model_img(image)
    print(f"Output shape: {output_img.shape}")
