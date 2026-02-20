"""Hierarchical ResNet18 모델 정의 - 2단계 분류

1단계 (Coarse): Normal vs Defect
2단계 (Fine): Defect 종류 (cell_porosity, module_porosity, module_resin)

학습: Coarse는 전체 샘플, Fine은 Defect 샘플만
추론: Coarse → Defect인 경우만 Fine 사용
"""
import torch
import torch.nn as nn
import torchvision.models as models


class HierarchicalResNet18(nn.Module):
    """
    Hierarchical ResNet18 - 2단계 분류 모델

    Coarse (2 classes):
        0: Normal (cell_normal + module_normal)
        1: Defect (cell_porosity + module_porosity + module_resin)

    Fine (5 classes, Defect 세부 분류):
        0: cell_normal (Normal일 때)
        1: cell_porosity
        2: module_normal (Normal일 때)
        3: module_porosity
        4: module_resin
    """

    def __init__(self, num_fine_classes: int = 5, pretrained: bool = True, dropout: float = 0.3):
        """
        Args:
            num_fine_classes: Fine 분류 클래스 수 (기본 5)
            pretrained: ImageNet pretrained 사용 여부
            dropout: Dropout 비율
        """
        super(HierarchicalResNet18, self).__init__()

        # ResNet18 backbone
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = models.resnet18(weights=weights)

        # Feature extractor (FC layer 제외)
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool
        )

        self.feature_dim = 512
        self.dropout = nn.Dropout(p=dropout)

        # Coarse head: Normal(0) vs Defect(1)
        self.head_coarse = nn.Linear(self.feature_dim, 2)

        # Fine head: 5 classes (전체 분류)
        self.head_fine = nn.Linear(self.feature_dim, num_fine_classes)

        self.num_fine_classes = num_fine_classes

        print(f"✅ Hierarchical ResNet18 모델 생성 완료")
        print(f"   - Pretrained: {pretrained}")
        print(f"   - Coarse classes: 2 (Normal vs Defect)")
        print(f"   - Fine classes: {num_fine_classes}")
        print(f"   - Dropout: {dropout}")

    def forward(self, x, return_features: bool = False):
        """
        Args:
            x: (B, 3, H, W) 입력 이미지
            return_features: True면 features도 반환

        Returns:
            coarse_logits: (B, 2) Normal vs Defect
            fine_logits: (B, num_fine_classes) 세부 분류
            features: (B, 512) - return_features=True일 때만
        """
        # Feature extraction
        features = self.backbone(x)
        features = torch.flatten(features, 1)  # (B, 512)
        features = self.dropout(features)

        # Coarse prediction
        coarse_logits = self.head_coarse(features)

        # Fine prediction
        fine_logits = self.head_fine(features)

        if return_features:
            return coarse_logits, fine_logits, features
        return coarse_logits, fine_logits

    def predict(self, x):
        """
        순차 추론: Coarse → Fine (조건부)

        Args:
            x: (B, 3, H, W) 입력 이미지

        Returns:
            final_preds: (B,) 최종 예측 (0~4)
            coarse_preds: (B,) Coarse 예측 (0: Normal, 1: Defect)
            fine_preds: (B,) Fine 예측 (0~4)
            coarse_probs: (B, 2) Coarse 확률
            fine_probs: (B, num_fine_classes) Fine 확률
        """
        self.eval()
        with torch.no_grad():
            coarse_logits, fine_logits = self.forward(x)

            coarse_probs = torch.softmax(coarse_logits, dim=1)
            fine_probs = torch.softmax(fine_logits, dim=1)

            coarse_preds = torch.argmax(coarse_probs, dim=1)  # 0: Normal, 1: Defect
            fine_preds = torch.argmax(fine_probs, dim=1)  # 0~4

            # 순차 로직: Coarse가 Normal이면 Normal 클래스 사용
            # Normal: cell_normal(0) 또는 module_normal(2) 중 Fine 확률 높은 것
            # Defect: Fine prediction 그대로 사용

            final_preds = fine_preds.clone()

            # Normal인 경우: cell_normal(0) vs module_normal(2) 중 선택
            normal_mask = (coarse_preds == 0)
            if normal_mask.sum() > 0:
                # Normal 클래스들의 확률만 비교
                normal_probs = fine_probs[normal_mask][:, [0, 2]]  # cell_normal, module_normal
                normal_choice = torch.argmax(normal_probs, dim=1)
                # 0 → cell_normal(0), 1 → module_normal(2)
                final_preds[normal_mask] = torch.where(normal_choice == 0,
                                                        torch.tensor(0, device=x.device),
                                                        torch.tensor(2, device=x.device))

        return final_preds, coarse_preds, fine_preds, coarse_probs, fine_probs


class HierarchicalLoss(nn.Module):
    """
    Hierarchical Loss - Coarse + Conditional Fine Loss

    Coarse Loss: 전체 샘플에 대해 계산
    Fine Loss: Defect 샘플에 대해서만 계산
    """

    def __init__(
        self,
        coarse_weight: float = 1.0,
        fine_weight: float = 1.0,
        fine_class_weights: torch.Tensor = None,
        label_smoothing: float = 0.0
    ):
        """
        Args:
            coarse_weight: Coarse loss 가중치
            fine_weight: Fine loss 가중치
            fine_class_weights: Fine 클래스별 가중치 [num_classes]
            label_smoothing: Label smoothing 값
        """
        super().__init__()
        self.coarse_weight = coarse_weight
        self.fine_weight = fine_weight
        self.label_smoothing = label_smoothing

        # Coarse loss (Normal vs Defect)
        self.coarse_criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        # Fine loss (5 classes)
        self.fine_criterion = nn.CrossEntropyLoss(
            weight=fine_class_weights,
            label_smoothing=label_smoothing
        )

        # Label mapping: fine_label → coarse_label
        # 0(cell_normal), 2(module_normal) → 0(Normal)
        # 1(cell_porosity), 3(module_porosity), 4(resin) → 1(Defect)
        self.normal_classes = [0, 2]  # cell_normal, module_normal
        self.defect_classes = [1, 3, 4]  # cell_porosity, module_porosity, resin

    def get_coarse_labels(self, fine_labels: torch.Tensor) -> torch.Tensor:
        """Fine labels를 Coarse labels로 변환"""
        coarse_labels = torch.zeros_like(fine_labels)
        for defect_class in self.defect_classes:
            coarse_labels[fine_labels == defect_class] = 1
        return coarse_labels

    def forward(
        self,
        coarse_logits: torch.Tensor,
        fine_logits: torch.Tensor,
        fine_labels: torch.Tensor
    ) -> tuple:
        """
        Args:
            coarse_logits: (B, 2) Coarse 예측
            fine_logits: (B, num_classes) Fine 예측
            fine_labels: (B,) 정답 라벨 (0~4)

        Returns:
            total_loss: 총 loss
            coarse_loss: Coarse loss
            fine_loss: Fine loss (Defect만)
        """
        # Coarse labels 생성
        coarse_labels = self.get_coarse_labels(fine_labels)

        # Coarse loss (전체)
        coarse_loss = self.coarse_criterion(coarse_logits, coarse_labels)

        # Fine loss (Defect만)
        defect_mask = (coarse_labels == 1)

        if defect_mask.sum() > 0:
            fine_loss = self.fine_criterion(
                fine_logits[defect_mask],
                fine_labels[defect_mask]
            )
        else:
            fine_loss = torch.tensor(0.0, device=coarse_logits.device)

        # Total loss
        total_loss = self.coarse_weight * coarse_loss + self.fine_weight * fine_loss

        return total_loss, coarse_loss, fine_loss


def create_hierarchical_model(config: dict) -> nn.Module:
    """
    Config 기반 Hierarchical 모델 생성

    Args:
        config: YAML config dict

    Returns:
        모델 인스턴스
    """
    model = HierarchicalResNet18(
        num_fine_classes=config['classes']['num_classes'],
        pretrained=config['model'].get('pretrained', True),
        dropout=config['model'].get('dropout', 0.3)
    )
    return model


# 테스트
if __name__ == "__main__":
    # 모델 생성
    model = HierarchicalResNet18(num_fine_classes=5, pretrained=True)

    # 테스트 입력
    dummy_input = torch.randn(4, 3, 512, 512)

    # Forward
    coarse, fine = model(dummy_input)
    print(f"Coarse shape: {coarse.shape}")  # (4, 2)
    print(f"Fine shape: {fine.shape}")  # (4, 5)

    # Predict (순차 추론)
    final, coarse_pred, fine_pred, _, _ = model.predict(dummy_input)
    print(f"Final predictions: {final}")
    print(f"Coarse predictions: {coarse_pred}")
    print(f"Fine predictions: {fine_pred}")

    # Loss 테스트
    criterion = HierarchicalLoss()
    labels = torch.tensor([0, 1, 3, 4])  # cell_normal, cell_porosity, module_porosity, resin
    total, coarse_loss, fine_loss = criterion(coarse, fine, labels)
    print(f"Total loss: {total:.4f}")
    print(f"Coarse loss: {coarse_loss:.4f}")
    print(f"Fine loss: {fine_loss:.4f}")
