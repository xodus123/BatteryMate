"""개별 모델 Predictor - CT CNN, RGB AutoEncoder"""

import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
_project_root = Path(__file__).parent.parent.parent.absolute()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
import numpy as np
import json
from typing import Dict, Tuple, Optional

from models.ct_cnn.model import create_model as create_cnn_model
from models.rgb_ae.model import create_model as create_ae_model
from training.configs.config_loader import ConfigLoader
from training.data.transforms import get_transforms, get_albumentations_transforms


class CTCNNPredictor:
    """CT CNN Predictor (5클래스 다중분류)"""

    # 클래스 이름
    CLASS_NAMES = ['cell_normal', 'cell_porosity', 'module_normal',
                   'module_porosity', 'module_resin_overflow']

    # 불량 클래스 인덱스
    DEFECT_CLASSES = [1, 3, 4]  # cell_porosity, module_porosity, module_resin_overflow
    NORMAL_CLASSES = [0, 2]     # cell_normal, module_normal

    def __init__(self, checkpoint_path: str, config_name: str = 'cnn_ct_unified'):
        """
        Args:
            checkpoint_path: 체크포인트 파일 경로
            config_name: config 파일 이름
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Config 로드
        config_loader = ConfigLoader()
        self.config = config_loader.load(config_name)

        # 모델 로드
        self.model = create_cnn_model(self.config).to(self.device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # Transform (config에서 설정 읽기)
        image_size = self.config['data']['image_size']
        preprocessed = self.config['data'].get('preprocessed', False)
        use_albumentations = self.config['data'].get('use_albumentations', False)

        if use_albumentations:
            self.transform = get_albumentations_transforms('ct', 'test', image_size, preprocessed)
        else:
            self.transform = get_transforms('ct', 'test', image_size, preprocessed)

        # Grad-CAM
        self.gradcam = None
        self.image_size = image_size
        self.preprocessed = preprocessed

        print(f"CTCNNPredictor 로드 완료")
        print(f"   - Checkpoint: {checkpoint_path}")
        print(f"   - Image Size: {image_size}")
        print(f"   - Preprocessed: {preprocessed}")
        print(f"   - Device: {self.device}")

    def _init_gradcam(self, target_layer: str = 'layer4'):
        """Grad-CAM 초기화 (lazy loading)"""
        if self.gradcam is None:
            from models.inspector.gradcam import GradCAM
            self.gradcam = GradCAM(self.model, target_layer=target_layer)
        return self.gradcam

    @torch.no_grad()
    def predict(self, image_path: str) -> Dict:
        """
        단일 이미지 예측

        Args:
            image_path: 이미지 파일 경로

        Returns:
            {
                "class_idx": 1,
                "class_name": "cell_porosity",
                "probabilities": [0.1, 0.7, 0.05, 0.1, 0.05],
                "is_defect": True,
                "defect_probability": 0.85,
                "confidence": 0.7
            }
        """
        # 이미지 로드
        image = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # 추론
        logits = self.model(input_tensor)
        probabilities = F.softmax(logits, dim=1).cpu().numpy()[0]

        # 예측 클래스
        class_idx = int(np.argmax(probabilities))
        class_name = self.CLASS_NAMES[class_idx]
        confidence = float(probabilities[class_idx])

        # 불량 확률 계산 (불량 클래스들의 확률 합)
        defect_probability = float(sum(probabilities[i] for i in self.DEFECT_CLASSES))
        is_defect = class_idx in self.DEFECT_CLASSES

        return {
            "class_idx": class_idx,
            "class_name": class_name,
            "probabilities": probabilities.tolist(),
            "is_defect": is_defect,
            "defect_probability": defect_probability,
            "confidence": confidence
        }

    @torch.no_grad()
    def predict_batch(self, image_paths: list) -> list:
        """배치 예측"""
        results = []
        for path in image_paths:
            results.append(self.predict(path))
        return results

    def predict_with_gradcam(
        self,
        image_path: str,
        target_class: Optional[int] = None,
        target_layer: str = 'layer4',
        alpha: float = 0.4
    ) -> Dict:
        """
        Grad-CAM과 함께 예측

        Args:
            image_path: 이미지 파일 경로
            target_class: 타겟 클래스 (None이면 예측 클래스 사용)
            target_layer: Grad-CAM 타겟 레이어
            alpha: 오버레이 투명도

        Returns:
            {
                "class_idx": 1,
                "class_name": "cell_porosity",
                "probabilities": [...],
                "is_defect": True,
                "confidence": 0.7,
                "gradcam": {
                    "heatmap": np.ndarray (H, W),
                    "heatmap_colored": np.ndarray (H, W, 3),
                    "overlay": np.ndarray (H, W, 3),
                    "original": np.ndarray (H, W, 3)
                }
            }
        """
        import cv2

        # Grad-CAM 초기화
        gradcam = self._init_gradcam(target_layer)

        # 이미지 로드
        image = Image.open(image_path).convert('RGB')
        original_np = np.array(image.resize((self.image_size, self.image_size)))
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Grad-CAM 생성
        heatmap, pred_class, confidence = gradcam.generate(input_tensor, target_class)

        # 히트맵 리사이즈 및 컬러맵 적용
        h, w = original_np.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h))
        heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        # 오버레이
        overlay = cv2.addWeighted(original_np, 1 - alpha, heatmap_colored, alpha, 0)

        # 기본 예측 결과
        probabilities = F.softmax(self.model(input_tensor), dim=1).detach().cpu().numpy()[0]
        class_idx = int(pred_class)
        class_name = self.CLASS_NAMES[class_idx]
        defect_probability = float(sum(probabilities[i] for i in self.DEFECT_CLASSES))
        is_defect = class_idx in self.DEFECT_CLASSES

        return {
            "class_idx": class_idx,
            "class_name": class_name,
            "probabilities": probabilities.tolist(),
            "is_defect": is_defect,
            "defect_probability": defect_probability,
            "confidence": float(confidence),
            "gradcam": {
                "heatmap": heatmap_resized,
                "heatmap_colored": heatmap_colored,
                "overlay": overlay,
                "original": original_np
            }
        }


class RGBAEPredictor:
    """RGB AutoEncoder Predictor (이상 탐지)"""

    def __init__(self, checkpoint_path: str, config_name: str = 'autoencoder_rgb',
                 threshold_path: Optional[str] = None):
        """
        Args:
            checkpoint_path: 체크포인트 파일 경로
            config_name: config 파일 이름
            threshold_path: threshold.json 파일 경로 (None이면 checkpoint에서 로드)
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Config 로드
        config_loader = ConfigLoader()
        self.config = config_loader.load(config_name)

        # 모델 로드
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # Config가 checkpoint에 있으면 사용
        if 'config' in checkpoint:
            self.config = checkpoint['config']

        self.model = create_ae_model(self.config).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # Threshold 로드
        self.threshold = checkpoint.get('threshold', None)
        if self.threshold is None and threshold_path:
            threshold_file = Path(threshold_path)
            if threshold_file.exists():
                with open(threshold_file, 'r') as f:
                    threshold_data = json.load(f)
                    self.threshold = threshold_data.get('threshold', 0.1)

        if self.threshold is None:
            # 기본값
            self.threshold = 0.1
            print(f"   - 기본 threshold 사용: {self.threshold}")

        # 정규화를 위한 통계값 (threshold 파일에서 로드)
        self.score_mean = None
        self.score_std = None
        if threshold_path:
            threshold_file = Path(threshold_path)
            if threshold_file.exists():
                with open(threshold_file, 'r') as f:
                    threshold_data = json.load(f)
                    self.score_mean = threshold_data.get('mean', None)
                    self.score_std = threshold_data.get('std', None)

        # Transform (config에서 설정 읽기)
        image_size = self.config['data']['image_size']
        preprocessed = self.config['data'].get('preprocessed', False)
        use_albumentations = self.config['data'].get('use_albumentations', False)

        if use_albumentations:
            self.transform = get_albumentations_transforms('rgb', 'test', image_size, preprocessed)
        else:
            self.transform = get_transforms('rgb', 'test', image_size, preprocessed)

        self.image_size = image_size
        self.preprocessed = preprocessed

        print(f"RGBAEPredictor 로드 완료")
        print(f"   - Checkpoint: {checkpoint_path}")
        print(f"   - Image Size: {image_size}")
        print(f"   - Preprocessed: {preprocessed}")
        print(f"   - Threshold: {self.threshold:.4f}")
        print(f"   - Device: {self.device}")

    @torch.no_grad()
    def predict(self, image_path: str) -> Dict:
        """
        단일 이미지 예측

        Args:
            image_path: 이미지 파일 경로

        Returns:
            {
                "anomaly_score": 0.72,
                "normalized_score": 0.85,
                "threshold": 0.65,
                "is_defect": True,
                "confidence": 0.85
            }
        """
        # 이미지 로드
        image = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # 이상 점수 계산
        anomaly_score = float(self.model.get_anomaly_score(input_tensor).cpu().numpy()[0])

        # 정규화된 점수 계산 (0~1 범위)
        if self.score_mean is not None and self.score_std is not None:
            # Z-score 기반 정규화 후 sigmoid로 0~1 변환
            z_score = (anomaly_score - self.score_mean) / (self.score_std + 1e-8)
            normalized_score = float(1 / (1 + np.exp(-z_score)))
        else:
            # threshold 기준 정규화
            normalized_score = float(min(anomaly_score / (self.threshold * 2), 1.0))

        # 불량 여부 판정
        # AE는 정상 데이터로 학습 → 정상 패턴을 학습
        # 정상 입력 시 재구성 오류가 낮음 (학습 패턴과 유사)
        # 불량 입력 시 재구성 오류가 높음 (학습하지 않은 패턴)
        # 따라서 score > threshold → defect (정상 패턴과 다름)
        is_defect = anomaly_score > self.threshold

        # 신뢰도 계산 (개선된 방식)
        if is_defect:
            # 불량: threshold 대비 얼마나 높은지 (높을수록 정상 패턴과 다름)
            # threshold를 조금만 넘으면 60%, 2배가 되면 100%
            excess_ratio = min((anomaly_score - self.threshold) / self.threshold, 1.0)
            confidence = float(0.6 + 0.4 * excess_ratio)
        else:
            # 정상: threshold 대비 얼마나 낮은지 (낮을수록 정상 패턴과 유사)
            # score가 0이면 100% 확신, threshold에 가까우면 60% 확신
            ratio = anomaly_score / self.threshold
            confidence = float(0.6 + 0.4 * (1 - ratio))

        confidence = max(0.0, min(1.0, confidence))  # 0~1 클램핑

        return {
            "anomaly_score": anomaly_score,
            "normalized_score": normalized_score,
            "threshold": self.threshold,
            "is_defect": is_defect,
            "confidence": confidence
        }

    @torch.no_grad()
    def predict_batch(self, image_paths: list) -> list:
        """배치 예측"""
        results = []
        for path in image_paths:
            results.append(self.predict(path))
        return results

    @torch.no_grad()
    def get_reconstruction(self, image_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        재구성 이미지 및 에러맵 반환

        Returns:
            (original, reconstructed, error_map) - 모두 numpy array (H, W, 3)
        """
        # 이미지 로드
        image = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # 재구성
        reconstructed, _ = self.model(input_tensor)

        # Denormalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)

        original_denorm = input_tensor * std + mean
        reconstructed_denorm = reconstructed * std + mean

        # 클램핑
        original_denorm = torch.clamp(original_denorm, 0, 1)
        reconstructed_denorm = torch.clamp(reconstructed_denorm, 0, 1)

        # 에러맵
        error_map = torch.abs(original_denorm - reconstructed_denorm).mean(dim=1, keepdim=True)
        error_map = error_map / error_map.max()  # 정규화

        # numpy 변환
        original_np = original_denorm[0].cpu().permute(1, 2, 0).numpy()
        reconstructed_np = reconstructed_denorm[0].cpu().permute(1, 2, 0).numpy()
        error_np = error_map[0, 0].cpu().numpy()

        return original_np, reconstructed_np, error_np


# 테스트
if __name__ == "__main__":
    print("Predictor 테스트")

    # CT CNN 테스트 (체크포인트가 있을 경우)
    cnn_checkpoint = "models/ct_cnn/checkpoints/ct_unified_best_20260105_140553.pt"
    if Path(cnn_checkpoint).exists():
        cnn_predictor = CTCNNPredictor(cnn_checkpoint)
        print("CNN Predictor 테스트 완료")

    # RGB AE 테스트 (최신 체크포인트 자동 탐지)
    ae_dir = Path("models/rgb_ae/checkpoints")
    ae_files = sorted(ae_dir.glob("rgb_ae_best_*.pt"), reverse=True)
    if ae_files:
        ae_checkpoint = str(ae_files[0])
        ae_predictor = RGBAEPredictor(ae_checkpoint)
        print(f"AE Predictor 테스트 완료: {ae_checkpoint}")
