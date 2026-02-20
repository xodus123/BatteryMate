"""
CT 앙상블 검사기 - CNN+Metadata + Autoencoder 결합
- CNN+Metadata: Defect crop → 분류
- Autoencoder: Battery outline crop → 이상 탐지
"""

import sys
from pathlib import Path

_project_root = Path(__file__).parent.parent.parent.absolute()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import Dict, Tuple, Optional, List
import json

from training.configs.config_loader import ConfigLoader


class CTEnsembleInspector:
    """
    CT 앙상블 검사기

    - CNN+Metadata: defect crop 이미지 → 5클래스 분류
    - Autoencoder: battery outline crop 이미지 → 복원 에러 기반 이상 탐지
    - 앙상블: 두 모델 결과 결합
    """

    CLASS_NAMES = [
        'cell_normal', 'cell_porosity',
        'module_normal', 'module_porosity', 'module_resin_overflow'
    ]

    DEFECT_CLASSES = [1, 3, 4]  # porosity, module_porosity, resin_overflow
    NORMAL_CLASSES = [0, 2]     # cell_normal, module_normal

    def __init__(
        self,
        cnn_checkpoint: str,
        ae_checkpoint: str,
        ae_threshold_path: Optional[str] = None,
        cnn_config: str = 'cnn_ct_defect_crop',
        ae_config: str = 'autoencoder_ct',
        device: str = 'cuda',
        ensemble_weights: Tuple[float, float] = (0.7, 0.3)  # CNN, AE
    ):
        """
        Args:
            cnn_checkpoint: CNN+Metadata 체크포인트 경로
            ae_checkpoint: Autoencoder 체크포인트 경로
            ae_threshold_path: AE threshold.json 경로
            cnn_config: CNN config 이름
            ae_config: AE config 이름
            device: 디바이스
            ensemble_weights: (CNN 가중치, AE 가중치)
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.ensemble_weights = ensemble_weights

        print("=" * 60)
        print("CT 앙상블 검사기 초기화")
        print("=" * 60)

        # CNN+Metadata 모델 로드
        self.cnn_model, self.cnn_config = self._load_cnn_model(cnn_checkpoint, cnn_config)
        self.cnn_transform = self._get_cnn_transform()

        # Autoencoder 모델 로드
        self.ae_model, self.ae_config = self._load_ae_model(ae_checkpoint, ae_config)
        self.ae_transform = self._get_ae_transform()

        # AE threshold 로드
        self.ae_threshold = self._load_ae_threshold(ae_threshold_path)

        print(f"앙상블 가중치: CNN={ensemble_weights[0]}, AE={ensemble_weights[1]}")
        print("=" * 60)

    def _load_cnn_model(self, checkpoint_path: str, config_name: str):
        """CNN+Metadata 모델 로드"""
        print(f"CNN+Metadata 모델 로드: {checkpoint_path}")

        config = ConfigLoader.load(config_name)

        # 모델 생성
        from models.ct_cnn.model_metadata import ResNetMetadataFusion
        model = ResNetMetadataFusion(
            num_classes=config['model']['num_classes'],
            pretrained=False,
            dropout=config['model'].get('dropout', 0.3),
            metadata_hidden_dim=config['model'].get('metadata_hidden_dim', 32),
            metadata_output_dim=config['model'].get('metadata_output_dim', 32),
            metadata_dropout=config['model'].get('metadata_dropout', 0.5),
            fusion_hidden_dim=config['model'].get('fusion_hidden_dim', 256)
        )

        # 체크포인트 로드
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model = model.to(self.device)
        model.eval()

        print(f"  Classes: {config['model']['num_classes']}")
        return model, config

    def _load_ae_model(self, checkpoint_path: str, config_name: str):
        """Autoencoder 모델 로드"""
        print(f"Autoencoder 모델 로드: {checkpoint_path}")

        config = ConfigLoader.load(config_name)

        # 모델 생성
        from models.rgb_ae.model import ConvAutoEncoder
        model_config = config['model']
        encoder_config = model_config.get('encoder', {})
        decoder_config = model_config.get('decoder', {})

        model = ConvAutoEncoder(
            image_size=config['data']['image_size'],
            latent_dim=model_config.get('latent_dim', 512),
            encoder_channels=encoder_config.get('channels', [3, 64, 128, 256, 512]),
            decoder_channels=decoder_config.get('channels', [512, 256, 128, 64, 3]),
            dropout=model_config.get('dropout', 0.2)
        )

        # 체크포인트 로드
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model = model.to(self.device)
        model.eval()

        print(f"  Latent dim: {config['model'].get('latent_dim', 256)}")
        return model, config

    def _load_ae_threshold(self, threshold_path: Optional[str]) -> Dict:
        """AE threshold 로드 (Cell/Module 별도)"""
        default_thresholds = {
            'threshold': 0.186,
            'cell_threshold': 0.151,
            'module_threshold': 0.340
        }

        if threshold_path and Path(threshold_path).exists():
            with open(threshold_path, 'r') as f:
                data = json.load(f)
                thresholds = {
                    'threshold': data.get('threshold', default_thresholds['threshold']),
                    'cell_threshold': data.get('cell_threshold', default_thresholds['cell_threshold']),
                    'module_threshold': data.get('module_threshold', default_thresholds['module_threshold'])
                }
                print(f"AE Threshold 로드:")
                print(f"  Single: {thresholds['threshold']:.4f}")
                print(f"  Cell:   {thresholds['cell_threshold']:.4f}")
                print(f"  Module: {thresholds['module_threshold']:.4f}")
                return thresholds

        print(f"AE Threshold (기본값): {default_thresholds}")
        return default_thresholds

    def _get_cnn_transform(self):
        """CNN 전처리 transform"""
        from torchvision import transforms

        image_size = self.cnn_config['data'].get('image_size', 512)
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _get_ae_transform(self):
        """AE 전처리 transform"""
        from torchvision import transforms

        image_size = self.ae_config['data'].get('image_size', 1024)
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    def _extract_metadata(self, image_path: str) -> torch.Tensor:
        """이미지 경로에서 메타데이터 추출 (battery_type, axis)"""
        import re
        filename = Path(image_path).name.lower()

        # battery_type: cell=0, module=1
        battery_type = 1.0 if 'module' in filename else 0.0

        # axis: x=0, y=1, z=2
        axis_map = {'x': 0.0, 'y': 1.0, 'z': 2.0}
        axis = 0.0  # 기본값
        # 패턴: CT_cell_pouch_101_y_033 또는 CT_cell_pouch_101_y_033_p00
        match = re.search(r'_([xyz])_\d+', filename)
        if match:
            axis = axis_map.get(match.group(1), 0.0)

        return torch.tensor([[battery_type, axis]], dtype=torch.float32)

    def predict_cnn(self, defect_crop_path: str) -> Dict:
        """CNN+Metadata 예측 (defect crop 이미지)"""
        # 이미지 로드 및 전처리
        image = Image.open(defect_crop_path).convert('RGB')
        image_tensor = self.cnn_transform(image).unsqueeze(0).to(self.device)

        # 메타데이터 추출
        metadata = self._extract_metadata(defect_crop_path).to(self.device)

        # 예측
        with torch.no_grad():
            logits = self.cnn_model(image_tensor, metadata)
            probs = F.softmax(logits, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred_class].item()

        # 결함 확률 계산
        defect_prob = sum(probs[0, c].item() for c in self.DEFECT_CLASSES)

        return {
            'class_idx': pred_class,
            'class_name': self.CLASS_NAMES[pred_class],
            'confidence': confidence,
            'probabilities': probs[0].cpu().numpy(),
            'defect_probability': defect_prob,
            'is_defect': pred_class in self.DEFECT_CLASSES
        }

    def predict_ae(self, outline_crop_path: str) -> Dict:
        """Autoencoder 예측 (battery outline crop 이미지)"""
        # 이미지 로드 및 전처리
        image = Image.open(outline_crop_path).convert('RGB')
        image_tensor = self.ae_transform(image).unsqueeze(0).to(self.device)

        # Cell/Module 판별
        filename = Path(outline_crop_path).name.lower()
        is_module = 'module' in filename
        battery_type = 'module' if is_module else 'cell'

        # 해당 타입의 threshold 선택
        threshold = self.ae_threshold.get(
            f'{battery_type}_threshold',
            self.ae_threshold.get('threshold', 0.186)
        )

        # 예측
        with torch.no_grad():
            output = self.ae_model(image_tensor)
            # AE 모델이 (reconstructed, latent) 튜플을 반환
            if isinstance(output, tuple):
                reconstructed = output[0]
            else:
                reconstructed = output

            # 복원 에러 계산 (MSE)
            mse = F.mse_loss(reconstructed, image_tensor, reduction='none')
            reconstruction_error = mse.mean().item()

        # 이상 판정 (타입별 threshold 사용)
        is_anomaly = reconstruction_error > threshold

        # 이상 점수 정규화 (0~1)
        anomaly_score = min(1.0, reconstruction_error / (threshold * 2))

        return {
            'reconstruction_error': reconstruction_error,
            'anomaly_score': anomaly_score,
            'threshold': threshold,
            'battery_type': battery_type,
            'is_anomaly': is_anomaly,
            'confidence': abs(anomaly_score - 0.5) * 2  # 확신도
        }

    def predict(
        self,
        defect_crop_path: str,
        outline_crop_path: str
    ) -> Dict:
        """
        앙상블 예측

        Args:
            defect_crop_path: defect direct crop 이미지 경로 (512x512)
            outline_crop_path: battery outline crop 이미지 경로 (1024x1024)

        Returns:
            앙상블 예측 결과
        """
        # 개별 모델 예측
        cnn_result = self.predict_cnn(defect_crop_path)
        ae_result = self.predict_ae(outline_crop_path)

        # 앙상블 결합
        verdict, confidence, details = self._ensemble_combine(cnn_result, ae_result)

        return {
            'verdict': verdict,
            'verdict_class': self.CLASS_NAMES.index(verdict) if verdict in self.CLASS_NAMES else -1,
            'confidence': confidence,
            'is_defect': verdict not in ['cell_normal', 'module_normal'],
            'cnn_result': cnn_result,
            'ae_result': ae_result,
            'details': details
        }

    def _ensemble_combine(
        self,
        cnn_result: Dict,
        ae_result: Dict
    ) -> Tuple[str, float, Dict]:
        """
        앙상블 결합 로직

        전략:
        1. CNN이 결함으로 판정 + AE도 이상 → 결함 (높은 확신)
        2. CNN이 결함으로 판정 + AE는 정상 → CNN 결과 따름 (중간 확신)
        3. CNN이 정상으로 판정 + AE가 이상 → 재검토 필요
        4. 둘 다 정상 → 정상 (높은 확신)
        """
        cnn_weight, ae_weight = self.ensemble_weights

        cnn_is_defect = cnn_result['is_defect']
        ae_is_anomaly = ae_result['is_anomaly']

        details = {
            'cnn_class': cnn_result['class_name'],
            'cnn_defect_prob': cnn_result['defect_probability'],
            'cnn_confidence': cnn_result['confidence'],
            'ae_anomaly_score': ae_result['anomaly_score'],
            'ae_reconstruction_error': ae_result['reconstruction_error'],
            'ae_threshold': ae_result['threshold'],
            'ae_battery_type': ae_result.get('battery_type', 'unknown'),
            'ensemble_weights': self.ensemble_weights
        }

        if cnn_is_defect and ae_is_anomaly:
            # 둘 다 결함 → 높은 확신으로 결함
            verdict = cnn_result['class_name']
            confidence = cnn_weight * cnn_result['confidence'] + ae_weight * ae_result['confidence']
            details['agreement'] = 'both_defect'

        elif cnn_is_defect and not ae_is_anomaly:
            # CNN만 결함 → CNN 결과 따름 (확신도 낮춤)
            verdict = cnn_result['class_name']
            confidence = cnn_result['confidence'] * 0.8  # 20% 감소
            details['agreement'] = 'cnn_only_defect'

        elif not cnn_is_defect and ae_is_anomaly:
            # AE만 이상 → 정상으로 판정하되 주의 표시
            # CNN의 판정을 우선시 (CNN이 더 정확)
            verdict = cnn_result['class_name']
            confidence = cnn_result['confidence'] * 0.7  # 30% 감소
            details['agreement'] = 'ae_only_anomaly'
            details['warning'] = 'AE detected anomaly but CNN says normal'

        else:
            # 둘 다 정상 → 높은 확신으로 정상
            verdict = cnn_result['class_name']
            confidence = cnn_weight * cnn_result['confidence'] + ae_weight * (1 - ae_result['anomaly_score'])
            details['agreement'] = 'both_normal'

        return verdict, float(confidence), details

    def predict_batch(
        self,
        defect_crop_paths: List[str],
        outline_crop_paths: List[str]
    ) -> List[Dict]:
        """배치 예측"""
        results = []
        for defect_path, outline_path in zip(defect_crop_paths, outline_crop_paths):
            result = self.predict(defect_path, outline_path)
            results.append(result)
        return results

    def evaluate(
        self,
        defect_crop_paths: List[str],
        outline_crop_paths: List[str],
        labels: List[int],
        save_csv: Optional[str] = None
    ) -> Dict:
        """
        테스트셋 평가

        Args:
            defect_crop_paths: defect crop 이미지 경로 리스트
            outline_crop_paths: outline crop 이미지 경로 리스트
            labels: 정답 라벨 리스트
            save_csv: 결과 저장할 CSV 경로

        Returns:
            평가 메트릭 딕셔너리
        """
        from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
        from tqdm import tqdm
        import csv
        from datetime import datetime

        predictions = []
        all_results = []

        print(f"평가 시작: {len(labels)}개 샘플")

        for defect_path, outline_path, label in tqdm(
            zip(defect_crop_paths, outline_crop_paths, labels),
            total=len(labels),
            desc="Evaluating"
        ):
            result = self.predict(defect_path, outline_path)
            pred_class = result['verdict_class']
            predictions.append(pred_class)

            all_results.append({
                'defect_crop_path': defect_path,
                'outline_crop_path': outline_path,
                'true_label': label,
                'pred_label': pred_class,
                'pred_class_name': result['verdict'],
                'confidence': result['confidence'],
                'is_defect': result['is_defect'],
                'cnn_class': result['cnn_result']['class_name'],
                'cnn_confidence': result['cnn_result']['confidence'],
                'cnn_defect_prob': result['cnn_result']['defect_probability'],
                'ae_anomaly_score': result['ae_result']['anomaly_score'],
                'ae_is_anomaly': result['ae_result']['is_anomaly'],
                'agreement': result['details'].get('agreement', '')
            })

        # 메트릭 계산
        accuracy = accuracy_score(labels, predictions)
        f1_macro = f1_score(labels, predictions, average='macro')
        f1_per_class = f1_score(labels, predictions, average=None)
        cm = confusion_matrix(labels, predictions)

        metrics = {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_per_class': {self.CLASS_NAMES[i]: f1_per_class[i] for i in range(len(f1_per_class))},
            'confusion_matrix': cm.tolist(),
            'classification_report': classification_report(labels, predictions, target_names=self.CLASS_NAMES)
        }

        print(f"\n=== 평가 결과 ===")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Macro: {f1_macro:.4f}")
        print(f"\nF1 per class:")
        for name, f1 in metrics['f1_per_class'].items():
            print(f"  {name}: {f1:.4f}")
        print(f"\nConfusion Matrix:\n{cm}")

        # CSV 저장
        if save_csv:
            csv_path = Path(save_csv)
            csv_path.parent.mkdir(parents=True, exist_ok=True)

            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
                writer.writeheader()
                writer.writerows(all_results)

            print(f"\n결과 저장: {save_csv}")

            # 메트릭 요약 JSON도 저장
            metrics_path = csv_path.with_suffix('.json')
            with open(metrics_path, 'w') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'num_samples': len(labels),
                    'accuracy': accuracy,
                    'f1_macro': f1_macro,
                    'f1_per_class': metrics['f1_per_class'],
                    'confusion_matrix': metrics['confusion_matrix']
                }, f, indent=2)
            print(f"메트릭 저장: {metrics_path}")

        return metrics


def create_ct_ensemble_inspector(
    cnn_checkpoint: Optional[str] = None,
    ae_checkpoint: Optional[str] = None,
    ensemble_weights: Tuple[float, float] = (0.7, 0.3)
) -> CTEnsembleInspector:
    """
    CT 앙상블 검사기 생성 헬퍼

    체크포인트가 None이면 자동으로 최신 파일 탐색
    """
    # CNN 체크포인트 탐색
    if cnn_checkpoint is None:
        cnn_dir = Path("models/ct_cnn/checkpoints")
        cnn_files = sorted(cnn_dir.glob("ct_metadata_best_*.pt"), reverse=True)
        if not cnn_files:
            cnn_files = sorted(cnn_dir.glob("*.pt"), reverse=True)
        if cnn_files:
            cnn_checkpoint = str(cnn_files[0])
        else:
            raise FileNotFoundError("CNN 체크포인트를 찾을 수 없습니다.")

    # AE 체크포인트 탐색
    if ae_checkpoint is None:
        ae_dir = Path("models/ct_ae/checkpoints")
        if not ae_dir.exists():
            ae_dir = Path("models/rgb_ae/checkpoints")
        ae_files = sorted(ae_dir.glob("*_best_*.pt"), reverse=True)
        if ae_files:
            ae_checkpoint = str(ae_files[0])
        else:
            raise FileNotFoundError("AE 체크포인트를 찾을 수 없습니다.")

    # Threshold 파일
    ae_threshold_path = Path(ae_checkpoint).parent / "threshold.json"
    ae_threshold_path = str(ae_threshold_path) if ae_threshold_path.exists() else None

    return CTEnsembleInspector(
        cnn_checkpoint=cnn_checkpoint,
        ae_checkpoint=ae_checkpoint,
        ae_threshold_path=ae_threshold_path,
        ensemble_weights=ensemble_weights
    )


# 테스트
if __name__ == "__main__":
    print("CT 앙상블 검사기 테스트")
    print("=" * 60)

    # 테스트 이미지 경로 (예시)
    defect_crop = "/mnt/d/battery-defect-direct/Training/CT_cell_pouch_101_y_033.jpg"
    outline_crop = "/mnt/d/battery-cropped-v2/Training/CT_cell_pouch_101_y_033.jpg"

    if Path(defect_crop).exists() and Path(outline_crop).exists():
        try:
            inspector = create_ct_ensemble_inspector()
            result = inspector.predict(defect_crop, outline_crop)

            print(f"\n예측 결과:")
            print(f"  Verdict: {result['verdict']}")
            print(f"  Confidence: {result['confidence']:.4f}")
            print(f"  Is Defect: {result['is_defect']}")
            print(f"\nCNN 결과:")
            print(f"  Class: {result['cnn_result']['class_name']}")
            print(f"  Defect Prob: {result['cnn_result']['defect_probability']:.4f}")
            print(f"\nAE 결과:")
            print(f"  Anomaly Score: {result['ae_result']['anomaly_score']:.4f}")
            print(f"  Is Anomaly: {result['ae_result']['is_anomaly']}")

        except Exception as e:
            print(f"테스트 실패: {e}")
    else:
        print("테스트 이미지가 없습니다. 전처리를 먼저 실행하세요.")
