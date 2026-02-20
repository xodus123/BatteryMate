"""Hierarchical 모델용 Grad-CAM 시각화 모듈

2단계 분류 모델에서:
- Coarse head (Normal vs Defect) 판정 근거
- Fine head (세부 결함 종류) 판정 근거
각각 시각화 가능
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Tuple, Optional, List


class HierarchicalGradCAM:
    """
    Hierarchical 모델용 Grad-CAM

    Coarse/Fine 두 head에 대해 각각 Grad-CAM 생성 가능
    """

    def __init__(self, model: torch.nn.Module, target_layer: str = 'layer4'):
        """
        Args:
            model: HierarchicalResNet18 모델
            target_layer: Grad-CAM을 적용할 레이어 이름 (backbone 내)
        """
        self.model = model
        self.target_layer = target_layer
        self.device = next(model.parameters()).device

        # Hook 저장용
        self.gradients = None
        self.activations = None

        # Hook 등록
        self._register_hooks()

    def _register_hooks(self):
        """Forward/Backward hook 등록"""

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        # Hierarchical 모델의 backbone에서 타겟 레이어 찾기
        target = None

        # backbone은 Sequential이므로 내부에서 찾기
        if hasattr(self.model, 'backbone'):
            # backbone 내의 각 모듈 순회
            for idx, module in enumerate(self.model.backbone):
                module_name = module.__class__.__name__

                # layer4를 찾기 (Sequential의 7번째가 layer4)
                # ResNet18 구조: conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4, avgpool
                if self.target_layer == 'layer4' and idx == 7:
                    target = module
                    break

                # 이름으로 찾기 (named_modules)
                for name, submodule in module.named_modules():
                    if name == self.target_layer or f'{idx}.{name}' == self.target_layer:
                        target = submodule
                        break

                if target is not None:
                    break

        # 직접 이름으로 찾기
        if target is None:
            for name, module in self.model.named_modules():
                if self.target_layer in name:
                    target = module
                    break

        # 마지막 Conv2d fallback
        if target is None:
            last_conv = None
            for name, module in self.model.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    last_conv = module
            if last_conv is not None:
                target = last_conv
                print(f"[HierarchicalGradCAM] 타겟 레이어를 찾지 못해 마지막 Conv2d 사용")

        if target is None:
            raise ValueError(f"타겟 레이어 '{self.target_layer}'를 찾을 수 없습니다.")

        target.register_forward_hook(forward_hook)
        target.register_full_backward_hook(backward_hook)

    def generate_coarse(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> Tuple[np.ndarray, int, float]:
        """
        Coarse head (Normal vs Defect)에 대한 Grad-CAM

        Args:
            input_tensor: (1, C, H, W) 입력 이미지
            target_class: 0(Normal) or 1(Defect), None이면 예측 사용

        Returns:
            (heatmap, predicted_class, confidence)
        """
        self.model.eval()
        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad = True

        # Forward
        coarse_logits, fine_logits = self.model(input_tensor)

        # Softmax
        coarse_probs = F.softmax(coarse_logits, dim=1)

        # 타겟 클래스
        if target_class is None:
            target_class = coarse_logits.argmax(dim=1).item()

        confidence = coarse_probs[0, target_class].item()

        # Backward (Coarse head에 대해)
        self.model.zero_grad()
        one_hot = torch.zeros_like(coarse_logits)
        one_hot[0, target_class] = 1
        coarse_logits.backward(gradient=one_hot, retain_graph=True)

        # Grad-CAM 계산
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        heatmap = cam.squeeze().cpu().numpy()

        return heatmap, target_class, confidence

    def generate_fine(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> Tuple[np.ndarray, int, float]:
        """
        Fine head (세부 분류)에 대한 Grad-CAM

        Args:
            input_tensor: (1, C, H, W) 입력 이미지
            target_class: 0~4 클래스, None이면 예측 사용

        Returns:
            (heatmap, predicted_class, confidence)
        """
        self.model.eval()
        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad = True

        # Forward
        coarse_logits, fine_logits = self.model(input_tensor)

        # Softmax
        fine_probs = F.softmax(fine_logits, dim=1)

        # 타겟 클래스
        if target_class is None:
            target_class = fine_logits.argmax(dim=1).item()

        confidence = fine_probs[0, target_class].item()

        # Backward (Fine head에 대해)
        self.model.zero_grad()
        one_hot = torch.zeros_like(fine_logits)
        one_hot[0, target_class] = 1
        fine_logits.backward(gradient=one_hot, retain_graph=True)

        # Grad-CAM 계산
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        heatmap = cam.squeeze().cpu().numpy()

        return heatmap, target_class, confidence

    def generate_both(
        self,
        input_tensor: torch.Tensor
    ) -> dict:
        """
        Coarse와 Fine 모두에 대한 Grad-CAM 생성

        Args:
            input_tensor: (1, C, H, W) 입력 이미지

        Returns:
            {
                'coarse': (heatmap, pred, conf),
                'fine': (heatmap, pred, conf),
                'final_pred': 순차 추론 결과
            }
        """
        # Coarse Grad-CAM
        coarse_heatmap, coarse_pred, coarse_conf = self.generate_coarse(input_tensor)

        # Fine Grad-CAM
        fine_heatmap, fine_pred, fine_conf = self.generate_fine(input_tensor)

        # 순차 추론 결과
        if coarse_pred == 0:  # Normal
            # cell_normal(0) vs module_normal(2) 중 선택
            self.model.eval()
            with torch.no_grad():
                _, fine_logits = self.model(input_tensor.to(self.device))
                fine_probs = F.softmax(fine_logits, dim=1)
                normal_probs = fine_probs[0, [0, 2]]
                final_pred = 0 if normal_probs[0] > normal_probs[1] else 2
        else:  # Defect
            final_pred = fine_pred

        return {
            'coarse': (coarse_heatmap, coarse_pred, coarse_conf),
            'fine': (fine_heatmap, fine_pred, fine_conf),
            'final_pred': final_pred
        }

    def __call__(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        head: str = 'fine'
    ) -> Tuple[np.ndarray, int]:
        """
        기존 GradCAM 인터페이스 호환

        Args:
            input_tensor: 입력 이미지
            target_class: 타겟 클래스
            head: 'coarse' or 'fine'

        Returns:
            (heatmap, predicted_class)
        """
        if head == 'coarse':
            heatmap, pred, _ = self.generate_coarse(input_tensor, target_class)
        else:
            heatmap, pred, _ = self.generate_fine(input_tensor, target_class)

        return heatmap, pred


def visualize_hierarchical_gradcam(
    coarse_heatmap: np.ndarray,
    fine_heatmap: np.ndarray,
    original_image: np.ndarray,
    coarse_pred: int,
    fine_pred: int,
    final_pred: int,
    coarse_conf: float,
    fine_conf: float,
    class_names: List[str],
    save_path: Optional[str] = None
) -> Optional[np.ndarray]:
    """
    Hierarchical Grad-CAM 시각화

    Coarse와 Fine 히트맵을 나란히 표시

    Args:
        coarse_heatmap: Coarse Grad-CAM (H, W)
        fine_heatmap: Fine Grad-CAM (H, W)
        original_image: 원본 이미지 (H, W, 3)
        coarse_pred: Coarse 예측 (0: Normal, 1: Defect)
        fine_pred: Fine 예측 (0~4)
        final_pred: 최종 예측 (순차 추론)
        coarse_conf: Coarse 신뢰도
        fine_conf: Fine 신뢰도
        class_names: 클래스 이름 리스트
        save_path: 저장 경로

    Returns:
        시각화 이미지 (저장 안 할 경우)
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    coarse_names = ['Normal', 'Defect']

    # 원본 이미지
    if original_image.max() <= 1.0:
        axes[0].imshow(original_image)
    else:
        axes[0].imshow(original_image.astype(np.uint8))
    axes[0].set_title(f'Original\nFinal: {class_names[final_pred]}')
    axes[0].axis('off')

    # Coarse Grad-CAM
    h, w = original_image.shape[:2]
    coarse_resized = cv2.resize(coarse_heatmap, (w, h))
    axes[1].imshow(coarse_resized, cmap='jet')
    axes[1].set_title(f'Coarse Grad-CAM\n{coarse_names[coarse_pred]}: {coarse_conf:.1%}')
    axes[1].axis('off')

    # Fine Grad-CAM
    fine_resized = cv2.resize(fine_heatmap, (w, h))
    axes[2].imshow(fine_resized, cmap='jet')
    axes[2].set_title(f'Fine Grad-CAM\n{class_names[fine_pred]}: {fine_conf:.1%}')
    axes[2].axis('off')

    # Overlay (Fine 기준)
    if original_image.max() <= 1.0:
        overlay_base = (original_image * 255).astype(np.uint8)
    else:
        overlay_base = original_image.astype(np.uint8)

    heatmap_uint8 = (fine_resized * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(overlay_base, 0.6, heatmap_colored, 0.4, 0)
    axes[3].imshow(overlay)
    axes[3].set_title('Overlay (Fine)')
    axes[3].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        return None
    else:
        fig.canvas.draw()
        img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()
        return img_array


# 테스트
if __name__ == "__main__":
    import sys
    from pathlib import Path
    _project_root = Path(__file__).parent.parent.parent.absolute()
    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))

    print("Hierarchical Grad-CAM 테스트")

    from models.ct_cnn.model_hierarchical import HierarchicalResNet18

    # 모델 생성
    model = HierarchicalResNet18(num_fine_classes=5, pretrained=True)
    model.eval()

    # Grad-CAM 생성
    gradcam = HierarchicalGradCAM(model, target_layer='layer4')

    # 더미 입력
    dummy_input = torch.randn(1, 3, 512, 512)

    # 테스트
    result = gradcam.generate_both(dummy_input)

    print(f"Coarse heatmap shape: {result['coarse'][0].shape}")
    print(f"Coarse pred: {result['coarse'][1]} ({['Normal', 'Defect'][result['coarse'][1]]})")
    print(f"Coarse conf: {result['coarse'][2]:.4f}")

    print(f"Fine heatmap shape: {result['fine'][0].shape}")
    print(f"Fine pred: {result['fine'][1]}")
    print(f"Fine conf: {result['fine'][2]:.4f}")

    print(f"Final pred: {result['final_pred']}")

    print("\n✅ Hierarchical Grad-CAM 테스트 완료!")
