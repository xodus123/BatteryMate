"""ResNet + Metadata Fusion Model

이미지 특징과 JSON 메타데이터를 결합하여 분류
- 이미지: ResNet18로 특징 추출
- 메타데이터:
  - battery_type (0=cell, 1=module)
  - axis (0=x, 1=y, 2=z)
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional, Dict


class MetadataEncoder(nn.Module):
    """메타데이터를 인코딩하는 네트워크"""

    def __init__(self, input_dim: int, hidden_dim: int = 32, output_dim: int = 32, dropout: float = 0.5):
        super().__init__()
        self.output_dim = output_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),  # 메타데이터 의존도 감소
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)  # 추가 dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class ResNetMetadataFusion(nn.Module):
    """ResNet + Metadata Fusion Model

    메타데이터 입력:
        - battery_type: 0 (cell) or 1 (module)
        - axis: 0 (x), 1 (y), 2 (z)
    """

    METADATA_DIM = 2  # 메타데이터 특징 수 (battery_type + axis)

    def __init__(
        self,
        num_classes: int = 5,
        pretrained: bool = True,
        dropout: float = 0.5,
        metadata_hidden_dim: int = 32,
        metadata_output_dim: int = 32,
        metadata_dropout: float = 0.5,
        fusion_hidden_dim: int = 256
    ):
        super().__init__()

        self.num_classes = num_classes
        self.metadata_output_dim = metadata_output_dim  # zero padding용

        # ResNet18 백본 (이미지 특징 추출)
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.resnet18(weights=weights)

        # ResNet의 fc 레이어 제거 (특징만 추출)
        self.image_feature_dim = self.backbone.fc.in_features  # 512
        self.backbone.fc = nn.Identity()

        # 메타데이터 인코더 (의존도 감소: 작은 dim + 높은 dropout)
        self.metadata_encoder = MetadataEncoder(
            input_dim=self.METADATA_DIM,
            hidden_dim=metadata_hidden_dim,
            output_dim=metadata_output_dim,
            dropout=metadata_dropout
        )

        # Fusion 레이어 (이미지 특징 + 메타데이터 특징)
        fusion_input_dim = self.image_feature_dim + metadata_output_dim  # 512 + 32 = 544

        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_hidden_dim),
            nn.BatchNorm1d(fusion_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim // 2),
            nn.BatchNorm1d(fusion_hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5)
        )

        # 최종 분류기
        self.classifier = nn.Linear(fusion_hidden_dim // 2, num_classes)

        # 가중치 초기화
        self._init_weights()

    def _init_weights(self):
        """커스텀 레이어 가중치 초기화"""
        for module in [self.metadata_encoder, self.fusion, self.classifier]:
            for m in module.modules():
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

        Returns:
            (B, num_classes) 로짓
        """
        # 이미지 특징 추출
        image_features = self.backbone(image)  # (B, 512)

        if metadata is not None:
            # 메타데이터 인코딩
            metadata_features = self.metadata_encoder(metadata)  # (B, metadata_output_dim)

            # 특징 결합
            fused = torch.cat([image_features, metadata_features], dim=1)
        else:
            # 메타데이터 없으면 zero padding
            batch_size = image_features.size(0)
            device = image_features.device
            zero_metadata = torch.zeros(batch_size, self.metadata_output_dim, device=device)
            fused = torch.cat([image_features, zero_metadata], dim=1)

        # Fusion 및 분류
        fused = self.fusion(fused)
        logits = self.classifier(fused)

        return logits

    def get_image_features(self, image: torch.Tensor) -> torch.Tensor:
        """이미지 특징만 추출 (Grad-CAM용)"""
        return self.backbone(image)


def create_metadata_model(
    num_classes: int = 5,
    pretrained: bool = True,
    dropout: float = 0.5,
    **kwargs
) -> ResNetMetadataFusion:
    """모델 생성 헬퍼 함수"""
    return ResNetMetadataFusion(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
        **kwargs
    )


# 메타데이터 추출 유틸리티 함수
def extract_axis_from_filename(filename: str) -> float:
    """파일명에서 axis 추출

    파일명 형식: CT_{type}_{package}_{battery_id}_{axis}_{slice_id}.jpg
    예: CT_cell_pouch_102_x_141.jpg → axis = 0 (x)

    Returns:
        0.0 (x), 1.0 (y), 2.0 (z)
    """
    import os
    base_name = os.path.splitext(os.path.basename(filename))[0]
    parts = base_name.split('_')

    axis_map = {'x': 0.0, 'y': 1.0, 'z': 2.0}

    if len(parts) >= 2:
        axis_char = parts[-2].lower()
        if axis_char in axis_map:
            return axis_map[axis_char]

    return 0.0


def extract_metadata_from_json(label_data: dict, filename: str = None) -> torch.Tensor:
    """JSON 라벨 데이터에서 메타데이터 추출

    Args:
        label_data: JSON 라벨 딕셔너리
        filename: 이미지 파일명 (axis 추출용)

    Returns:
        (2,) 메타데이터 텐서 [battery_type, axis]
    """
    # 기본값
    battery_type = 0.0  # 0: cell, 1: module
    axis = 0.0  # 0: x, 1: y, 2: z

    # battery_type 추출
    data_info = label_data.get('data_info', {})
    if data_info.get('type', '').lower() == 'module':
        battery_type = 1.0

    # axis 추출 (파일명에서)
    if filename:
        axis = extract_axis_from_filename(filename)

    return torch.tensor([battery_type, axis], dtype=torch.float32)


if __name__ == "__main__":
    # 테스트
    model = create_metadata_model(num_classes=5, pretrained=True)
    print(f"Model created: {model.__class__.__name__}")

    # 파라미터 수
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")

    # Forward 테스트
    batch_size = 4
    image = torch.randn(batch_size, 3, 512, 512)
    metadata = torch.randn(batch_size, 2)  # [battery_type, axis]

    output = model(image, metadata)
    print(f"Input image shape: {image.shape}")
    print(f"Input metadata shape: {metadata.shape}")
    print(f"Output shape: {output.shape}")

    # 메타데이터 없이 테스트
    output_no_meta = model(image, None)
    print(f"Output (no metadata) shape: {output_no_meta.shape}")
