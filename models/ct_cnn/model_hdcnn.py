"""HD-CNN (Hierarchical Deep CNN) 구현

논문: "HD-CNN: Hierarchical Deep Convolutional Neural Networks for Large Scale Visual Recognition"

핵심 구조:
- Shared Layers: 저수준 특징 추출 (class-agnostic) → conv1 ~ layer2
- Coarse Branch: 독립적인 rear layers → layer3_c, layer4_c, fc_coarse
- Fine Branch: 독립적인 rear layers → layer3_f, layer4_f, fc_fine

학습:
- Coarse: 전체 샘플로 Normal vs Defect 학습
- Fine: Defect 샘플만으로 세부 분류 학습 (독립적 가중치)

추론:
- Coarse 먼저 실행 → Normal이면 cell/module_normal 중 선택
- Defect이면 Fine branch로 세부 분류
"""
import torch
import torch.nn as nn
import torchvision.models as models
from typing import Tuple, Optional
import copy


class HDCNN(nn.Module):
    """HD-CNN: Hierarchical Deep CNN

    Coarse (2 classes):
        0: Normal (cell_normal + module_normal)
        1: Defect (cell_porosity + module_porosity + module_resin)

    Fine (5 classes):
        0: cell_normal
        1: cell_porosity
        2: module_normal
        3: module_porosity
        4: module_resin
    """

    # 클래스 매핑
    NORMAL_CLASSES = [0, 2]  # cell_normal, module_normal
    DEFECT_CLASSES = [1, 3, 4]  # cell_porosity, module_porosity, module_resin

    def __init__(
        self,
        num_fine_classes: int = 5,
        pretrained: bool = True,
        dropout: float = 0.5
    ):
        """
        Args:
            num_fine_classes: Fine 분류 클래스 수 (기본 5)
            pretrained: ImageNet pretrained 사용 여부
            dropout: Dropout 비율
        """
        super().__init__()

        self.num_fine_classes = num_fine_classes
        self.dropout_rate = dropout

        # Pretrained ResNet18 로드
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = models.resnet18(weights=weights)

        # ============================================================
        # Shared Layers (저수준 특징 - class agnostic)
        # conv1 → bn1 → relu → maxpool → layer1 → layer2
        # ============================================================
        self.shared_layers = nn.Sequential(
            resnet.conv1,      # 3 → 64
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,    # 64 → 64
            resnet.layer1,     # 64 → 64
            resnet.layer2,     # 64 → 128
        )

        # ============================================================
        # Coarse Branch (독립적인 rear layers)
        # layer3_coarse → layer4_coarse → avgpool → fc_coarse
        # ============================================================
        self.coarse_layer3 = copy.deepcopy(resnet.layer3)  # 128 → 256
        self.coarse_layer4 = copy.deepcopy(resnet.layer4)  # 256 → 512
        self.coarse_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.coarse_dropout = nn.Dropout(p=dropout)
        self.coarse_fc = nn.Linear(512, 2)  # Normal vs Defect

        # ============================================================
        # Fine Branch (독립적인 rear layers)
        # layer3_fine → layer4_fine → avgpool → fc_fine
        # ============================================================
        self.fine_layer3 = copy.deepcopy(resnet.layer3)  # 128 → 256
        self.fine_layer4 = copy.deepcopy(resnet.layer4)  # 256 → 512
        self.fine_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fine_dropout = nn.Dropout(p=dropout)
        self.fine_fc = nn.Linear(512, num_fine_classes)  # 5 classes

        # 가중치 초기화 (fc layers)
        self._init_fc_weights()

        print(f"✅ HD-CNN 모델 생성 완료")
        print(f"   - Pretrained: {pretrained}")
        print(f"   - Shared Layers: conv1 → layer2")
        print(f"   - Coarse Branch: layer3 → layer4 → fc(2) [독립 가중치]")
        print(f"   - Fine Branch: layer3 → layer4 → fc({num_fine_classes}) [독립 가중치]")
        print(f"   - Dropout: {dropout}")

        # 파라미터 수 출력
        self._print_param_counts()

    def _init_fc_weights(self):
        """FC layer 가중치 초기화"""
        for fc in [self.coarse_fc, self.fine_fc]:
            nn.init.kaiming_normal_(fc.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(fc.bias, 0)

    def _print_param_counts(self):
        """파라미터 수 출력"""
        shared_params = sum(p.numel() for p in self.shared_layers.parameters())
        coarse_params = sum(p.numel() for p in self.coarse_layer3.parameters()) + \
                       sum(p.numel() for p in self.coarse_layer4.parameters()) + \
                       sum(p.numel() for p in self.coarse_fc.parameters())
        fine_params = sum(p.numel() for p in self.fine_layer3.parameters()) + \
                     sum(p.numel() for p in self.fine_layer4.parameters()) + \
                     sum(p.numel() for p in self.fine_fc.parameters())
        total_params = sum(p.numel() for p in self.parameters())

        print(f"   - Shared params: {shared_params:,}")
        print(f"   - Coarse branch params: {coarse_params:,}")
        print(f"   - Fine branch params: {fine_params:,}")
        print(f"   - Total params: {total_params:,}")

    def forward_shared(self, x: torch.Tensor) -> torch.Tensor:
        """Shared layers forward"""
        return self.shared_layers(x)

    def forward_coarse(self, shared_features: torch.Tensor) -> torch.Tensor:
        """Coarse branch forward"""
        x = self.coarse_layer3(shared_features)
        x = self.coarse_layer4(x)
        x = self.coarse_avgpool(x)
        x = torch.flatten(x, 1)
        x = self.coarse_dropout(x)
        return self.coarse_fc(x)

    def forward_fine(self, shared_features: torch.Tensor) -> torch.Tensor:
        """Fine branch forward"""
        x = self.fine_layer3(shared_features)
        x = self.fine_layer4(x)
        x = self.fine_avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fine_dropout(x)
        return self.fine_fc(x)

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass - 두 branch 모두 실행

        Args:
            x: (B, 3, H, W) 입력 이미지
            return_features: shared features도 반환할지

        Returns:
            coarse_logits: (B, 2) Normal vs Defect
            fine_logits: (B, 5) 세부 분류
            shared_features: (optional) shared layer 출력
        """
        # Shared layers
        shared_features = self.forward_shared(x)

        # Coarse branch (독립 가중치)
        coarse_logits = self.forward_coarse(shared_features)

        # Fine branch (독립 가중치)
        fine_logits = self.forward_fine(shared_features)

        if return_features:
            return coarse_logits, fine_logits, shared_features
        return coarse_logits, fine_logits

    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        계층적 추론: Coarse → Fine (조건부)

        Args:
            x: (B, 3, H, W) 입력 이미지

        Returns:
            final_preds: (B,) 최종 예측 (0~4)
            coarse_preds: (B,) Coarse 예측 (0: Normal, 1: Defect)
            fine_preds: (B,) Fine 예측 (0~4)
            coarse_probs: (B, 2) Coarse 확률
            fine_probs: (B, 5) Fine 확률
        """
        self.eval()
        with torch.no_grad():
            coarse_logits, fine_logits = self.forward(x)

            coarse_probs = torch.softmax(coarse_logits, dim=1)
            fine_probs = torch.softmax(fine_logits, dim=1)

            coarse_preds = torch.argmax(coarse_probs, dim=1)  # 0: Normal, 1: Defect
            fine_preds = torch.argmax(fine_probs, dim=1)      # 0~4

            # 계층적 결정
            final_preds = fine_preds.clone()

            # Normal로 판정된 경우: cell_normal(0) vs module_normal(2) 중 선택
            normal_mask = (coarse_preds == 0)
            if normal_mask.sum() > 0:
                # Normal 클래스들의 확률만 비교
                normal_probs = fine_probs[normal_mask][:, [0, 2]]
                normal_choice = torch.argmax(normal_probs, dim=1)
                # 0 → cell_normal(0), 1 → module_normal(2)
                final_preds[normal_mask] = torch.where(
                    normal_choice == 0,
                    torch.tensor(0, device=x.device),
                    torch.tensor(2, device=x.device)
                )

            # Defect로 판정된 경우: fine_preds 그대로 사용
            # (단, normal 클래스가 선택되지 않도록 마스킹 가능)
            defect_mask = (coarse_preds == 1)
            if defect_mask.sum() > 0:
                # Defect 클래스들의 확률만 비교
                defect_probs = fine_probs[defect_mask][:, [1, 3, 4]]
                defect_choice = torch.argmax(defect_probs, dim=1)
                # 0 → cell_porosity(1), 1 → module_porosity(3), 2 → module_resin(4)
                defect_mapping = torch.tensor([1, 3, 4], device=x.device)
                final_preds[defect_mask] = defect_mapping[defect_choice]

        return final_preds, coarse_preds, fine_preds, coarse_probs, fine_probs


class HDCNNLoss(nn.Module):
    """HD-CNN 학습용 Loss

    Coarse Loss: 전체 샘플에 대해 계산
    Fine Loss: Defect 샘플에 대해서만 계산 (조건부)
    """

    def __init__(
        self,
        coarse_weight: float = 1.0,
        fine_weight: float = 1.0,
        fine_class_weights: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0
    ):
        """
        Args:
            coarse_weight: Coarse loss 가중치
            fine_weight: Fine loss 가중치
            fine_class_weights: Fine 클래스별 가중치
            label_smoothing: Label smoothing
        """
        super().__init__()

        self.coarse_weight = coarse_weight
        self.fine_weight = fine_weight

        self.coarse_criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.fine_criterion = nn.CrossEntropyLoss(
            weight=fine_class_weights,
            label_smoothing=label_smoothing
        )

        self.normal_classes = HDCNN.NORMAL_CLASSES
        self.defect_classes = HDCNN.DEFECT_CLASSES

    def get_coarse_labels(self, fine_labels: torch.Tensor) -> torch.Tensor:
        """Fine labels → Coarse labels 변환"""
        coarse_labels = torch.zeros_like(fine_labels)
        for defect_class in self.defect_classes:
            coarse_labels[fine_labels == defect_class] = 1
        return coarse_labels

    def forward(
        self,
        coarse_logits: torch.Tensor,
        fine_logits: torch.Tensor,
        fine_labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            coarse_logits: (B, 2) Coarse 예측
            fine_logits: (B, 5) Fine 예측
            fine_labels: (B,) 정답 라벨 (0~4)

        Returns:
            total_loss: 총 loss
            coarse_loss: Coarse loss
            fine_loss: Fine loss
        """
        # Coarse labels 생성
        coarse_labels = self.get_coarse_labels(fine_labels)

        # Coarse loss (전체 샘플)
        coarse_loss = self.coarse_criterion(coarse_logits, coarse_labels)

        # Fine loss (Defect 샘플만)
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


def create_hdcnn_model(
    num_fine_classes: int = 5,
    pretrained: bool = True,
    dropout: float = 0.5
) -> HDCNN:
    """HD-CNN 모델 생성 헬퍼"""
    return HDCNN(
        num_fine_classes=num_fine_classes,
        pretrained=pretrained,
        dropout=dropout
    )


# ============================================================
# 테스트
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("HD-CNN 모델 테스트")
    print("=" * 60)

    # 모델 생성
    model = create_hdcnn_model(num_fine_classes=5, pretrained=True, dropout=0.5)

    # 테스트 입력
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 512, 512)

    # Forward
    coarse_logits, fine_logits = model(dummy_input)
    print(f"\nForward 결과:")
    print(f"  Coarse logits shape: {coarse_logits.shape}")  # (4, 2)
    print(f"  Fine logits shape: {fine_logits.shape}")      # (4, 5)

    # Predict (계층적 추론)
    final, coarse_pred, fine_pred, coarse_probs, fine_probs = model.predict(dummy_input)
    print(f"\nPredict 결과:")
    print(f"  Final predictions: {final}")
    print(f"  Coarse predictions: {coarse_pred} (0=Normal, 1=Defect)")
    print(f"  Fine predictions: {fine_pred}")

    # Loss 테스트
    criterion = HDCNNLoss(coarse_weight=1.0, fine_weight=1.0)
    labels = torch.tensor([0, 1, 3, 4])  # cell_normal, cell_porosity, module_porosity, resin

    total_loss, coarse_loss, fine_loss = criterion(coarse_logits, fine_logits, labels)
    print(f"\nLoss 결과:")
    print(f"  Total loss: {total_loss:.4f}")
    print(f"  Coarse loss: {coarse_loss:.4f}")
    print(f"  Fine loss: {fine_loss:.4f}")

    # 가중치 독립성 확인
    print(f"\n가중치 독립성 확인:")
    coarse_l3_weight = model.coarse_layer3[0].conv1.weight.data.mean().item()
    fine_l3_weight = model.fine_layer3[0].conv1.weight.data.mean().item()
    print(f"  Coarse layer3 weight mean: {coarse_l3_weight:.6f}")
    print(f"  Fine layer3 weight mean: {fine_l3_weight:.6f}")
    print(f"  동일 여부 (초기화 직후): {abs(coarse_l3_weight - fine_l3_weight) < 1e-6}")

    # 학습 시뮬레이션 (가중치 분리 확인)
    print(f"\n학습 후 가중치 분리 확인:")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 가짜 학습 1 step
    coarse_logits, fine_logits = model(dummy_input)
    loss, _, _ = criterion(coarse_logits, fine_logits, labels)
    loss.backward()
    optimizer.step()

    # 학습 후 가중치 비교
    coarse_l3_weight_after = model.coarse_layer3[0].conv1.weight.data.mean().item()
    fine_l3_weight_after = model.fine_layer3[0].conv1.weight.data.mean().item()
    print(f"  Coarse layer3 weight mean (after): {coarse_l3_weight_after:.6f}")
    print(f"  Fine layer3 weight mean (after): {fine_l3_weight_after:.6f}")
    print(f"  가중치 분리 확인: {abs(coarse_l3_weight_after - fine_l3_weight_after) > 1e-6}")

    print("\n" + "=" * 60)
    print("✅ HD-CNN 테스트 완료")
    print("=" * 60)
