"""Grad-CAM 시각화 모듈"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Tuple, Optional, List
from PIL import Image


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM)

    CNN 모델의 판정 근거 영역을 시각화
    """

    def __init__(self, model: torch.nn.Module, target_layer: str = 'layer4'):
        """
        Args:
            model: CNN 모델 (ResNet18 등)
            target_layer: Grad-CAM을 적용할 레이어 이름
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

        # 타겟 레이어 찾기
        target = None

        # 1. 직접 찾기 (ResNet18CBAM 등)
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                target = module
                break

        # 2. 내부 model 속성에서 찾기 (ResNet18Classifier: self.model.layer4)
        if target is None and hasattr(self.model, 'model'):
            for name, module in self.model.model.named_modules():
                if name == self.target_layer:
                    target = module
                    break

        # 3. backbone 속성에서 찾기
        if target is None and hasattr(self.model, 'backbone'):
            for name, module in self.model.backbone.named_modules():
                if name == self.target_layer:
                    target = module
                    break

        # 4. 마지막 conv layer 자동 찾기 (fallback)
        if target is None:
            # 마지막으로 발견된 Conv2d 레이어 사용
            last_conv = None
            for name, module in self.model.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    last_conv = module
            if last_conv is not None:
                target = last_conv
                print(f"[GradCAM] 타겟 레이어를 찾지 못해 마지막 Conv2d 레이어 사용")

        if target is None:
            raise ValueError(f"타겟 레이어 '{self.target_layer}'를 찾을 수 없습니다.")

        target.register_forward_hook(forward_hook)
        target.register_full_backward_hook(backward_hook)

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> Tuple[np.ndarray, int, float]:
        """
        Grad-CAM 히트맵 생성

        Args:
            input_tensor: 입력 이미지 텐서 (1, C, H, W)
            target_class: 타겟 클래스 인덱스 (None이면 예측 클래스 사용)

        Returns:
            (heatmap, predicted_class, confidence)
            - heatmap: (H, W) numpy array, 0~1 범위
            - predicted_class: 예측된 클래스 인덱스
            - confidence: 예측 확률
        """
        self.model.eval()
        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad = True

        # Forward pass
        output = self.model(input_tensor)

        # Softmax로 확률 변환
        probs = F.softmax(output, dim=1)

        # 타겟 클래스 결정
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        confidence = probs[0, target_class].item()

        # Backward pass
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        # Grad-CAM 계산
        # gradients: (1, C, H, W)
        # activations: (1, C, H, W)

        # Global Average Pooling of gradients
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted combination of activation maps
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)

        # ReLU
        cam = F.relu(cam)

        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        # numpy 변환
        heatmap = cam.squeeze().cpu().numpy()

        return heatmap, target_class, confidence

    def generate_with_overlay(
        self,
        input_tensor: torch.Tensor,
        original_image: np.ndarray,
        target_class: Optional[int] = None,
        alpha: float = 0.4,
        colormap: int = cv2.COLORMAP_JET
    ) -> Tuple[np.ndarray, np.ndarray, int, float]:
        """
        Grad-CAM 히트맵 + 원본 이미지 오버레이

        Args:
            input_tensor: 입력 이미지 텐서 (1, C, H, W)
            original_image: 원본 이미지 numpy array (H, W, 3), 0~255 범위
            target_class: 타겟 클래스 인덱스
            alpha: 오버레이 투명도
            colormap: OpenCV colormap

        Returns:
            (heatmap, overlay, predicted_class, confidence)
            - heatmap: 컬러 히트맵 (H, W, 3)
            - overlay: 오버레이 이미지 (H, W, 3)
            - predicted_class: 예측 클래스
            - confidence: 예측 확률
        """
        # Grad-CAM 생성
        cam, pred_class, conf = self.generate(input_tensor, target_class)

        # 원본 이미지 크기로 리사이즈
        h, w = original_image.shape[:2]
        cam_resized = cv2.resize(cam, (w, h))

        # 컬러맵 적용
        cam_uint8 = (cam_resized * 255).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(cam_uint8, colormap)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        # 오버레이
        if original_image.max() <= 1.0:
            original_image = (original_image * 255).astype(np.uint8)

        overlay = cv2.addWeighted(original_image, 1 - alpha, heatmap_colored, alpha, 0)

        return heatmap_colored, overlay, pred_class, conf


class GradCAMPlusPlus(GradCAM):
    """
    Grad-CAM++ - 더 정교한 시각화

    여러 객체가 있을 때 더 잘 동작
    """

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> Tuple[np.ndarray, int, float]:
        """Grad-CAM++ 히트맵 생성"""
        self.model.eval()
        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad = True

        # Forward pass
        output = self.model(input_tensor)
        probs = F.softmax(output, dim=1)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        confidence = probs[0, target_class].item()

        # Backward pass
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        # Grad-CAM++ 가중치 계산
        gradients = self.gradients  # (1, C, H, W)
        activations = self.activations  # (1, C, H, W)

        # Second derivative approximation
        grad_2 = gradients ** 2
        grad_3 = grad_2 * gradients

        # Alpha 계산
        sum_activations = activations.sum(dim=(2, 3), keepdim=True)
        alpha_num = grad_2
        alpha_denom = 2 * grad_2 + sum_activations * grad_3 + 1e-8
        alpha = alpha_num / alpha_denom

        # 가중치 계산
        weights = (alpha * F.relu(gradients)).sum(dim=(2, 3), keepdim=True)

        # CAM 계산
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        heatmap = cam.squeeze().cpu().numpy()

        return heatmap, target_class, confidence


def visualize_gradcam(
    heatmap: np.ndarray,
    original_image: np.ndarray,
    class_name: str,
    confidence: float,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 4)
) -> Optional[np.ndarray]:
    """
    Grad-CAM 결과 시각화 (matplotlib)

    Args:
        heatmap: Grad-CAM 히트맵 (H, W)
        original_image: 원본 이미지 (H, W, 3)
        class_name: 예측 클래스 이름
        confidence: 예측 확률
        save_path: 저장 경로 (None이면 저장 안함)
        figsize: 그림 크기

    Returns:
        결합된 이미지 (저장하지 않을 경우)
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # 원본 이미지
    if original_image.max() <= 1.0:
        axes[0].imshow(original_image)
    else:
        axes[0].imshow(original_image.astype(np.uint8))
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Grad-CAM 히트맵
    axes[1].imshow(heatmap, cmap='jet')
    axes[1].set_title('Grad-CAM Heatmap')
    axes[1].axis('off')

    # 오버레이
    h, w = original_image.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))

    if original_image.max() <= 1.0:
        overlay_base = (original_image * 255).astype(np.uint8)
    else:
        overlay_base = original_image.astype(np.uint8)

    heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(overlay_base, 0.6, heatmap_colored, 0.4, 0)

    axes[2].imshow(overlay)
    axes[2].set_title(f'Overlay\n{class_name}: {confidence:.2%}')
    axes[2].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        return None
    else:
        # Figure를 numpy array로 변환
        fig.canvas.draw()
        img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()
        return img_array


# 테스트
if __name__ == "__main__":
    print("Grad-CAM 모듈 테스트")

    # 간단한 테스트
    import torchvision.models as models

    # ResNet18 로드
    model = models.resnet18(pretrained=True)
    model.eval()

    # Grad-CAM 생성
    gradcam = GradCAM(model, target_layer='layer4')

    # 더미 입력
    dummy_input = torch.randn(1, 3, 224, 224)

    # 히트맵 생성
    heatmap, pred_class, conf = gradcam.generate(dummy_input)

    print(f"Heatmap shape: {heatmap.shape}")
    print(f"Predicted class: {pred_class}")
    print(f"Confidence: {conf:.4f}")
    print("Grad-CAM 테스트 완료!")
