"""앙상블 Predictor - CT CNN + RGB AutoEncoder 결합"""

import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
_project_root = Path(__file__).parent.parent.parent.absolute()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import yaml
from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np

from models.ensemble.predictor import CTCNNPredictor, RGBAEPredictor


class EnsemblePredictor:
    """
    CT CNN + RGB AutoEncoder 앙상블

    최종 판정:
    - "정상": CT 정상 AND RGB 정상
    - "내부불량": CT 불량 (porosity/resin_overflow)
    - "외부불량": RGB 불량 (오염/손상)
    - "복합불량": CT 불량 AND RGB 불량
    """

    # 최종 판정 클래스
    VERDICT_NORMAL = "정상"
    VERDICT_INTERNAL_DEFECT = "내부불량"
    VERDICT_EXTERNAL_DEFECT = "외부불량"
    VERDICT_COMPLEX_DEFECT = "복합불량"

    def __init__(
        self,
        ct_checkpoint: str,
        ae_checkpoint: str,
        ct_config: str = 'cnn_ct_unified',
        ae_config: str = 'autoencoder_rgb',
        ae_threshold_path: Optional[str] = None,
        ensemble_config_path: Optional[str] = None
    ):
        """
        Args:
            ct_checkpoint: CT CNN 체크포인트 경로
            ae_checkpoint: RGB AE 체크포인트 경로
            ct_config: CT CNN config 이름
            ae_config: RGB AE config 이름
            ae_threshold_path: AE threshold.json 경로
            ensemble_config_path: 앙상블 config 경로 (optional)
        """
        # 개별 모델 로드
        print("=" * 60)
        print("앙상블 모델 초기화")
        print("=" * 60)

        self.ct_predictor = CTCNNPredictor(ct_checkpoint, ct_config)
        self.ae_predictor = RGBAEPredictor(ae_checkpoint, ae_config, ae_threshold_path)

        # 앙상블 설정 로드
        self.ensemble_config = self._load_ensemble_config(ensemble_config_path)

        print("=" * 60)
        print("앙상블 모델 초기화 완료")
        print("=" * 60)

    def _load_ensemble_config(self, config_path: Optional[str]) -> dict:
        """앙상블 설정 로드"""
        default_config = {
            'ct_defect_classes': [1, 3, 4],  # cell_porosity, module_porosity, module_resin_overflow
            'ct_normal_classes': [0, 2],     # cell_normal, module_normal
            'rgb_threshold_multiplier': 1.0,
            'weighted_average': {
                'enabled': False,
                'w_ct': 0.6,
                'w_rgb': 0.4,
                'threshold': 0.5
            }
        }

        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                if 'ensemble' in config:
                    default_config.update(config['ensemble'])

        return default_config

    def predict(self, ct_image_path: str, rgb_image_path: str) -> Dict:
        """
        앙상블 예측

        Args:
            ct_image_path: CT 이미지 경로
            rgb_image_path: RGB 이미지 경로

        Returns:
            {
                "verdict": "내부불량",
                "verdict_en": "internal_defect",
                "confidence": 0.85,
                "ct_result": {...},
                "rgb_result": {...},
                "details": {...}
            }
        """
        # 개별 모델 예측
        ct_result = self.ct_predictor.predict(ct_image_path)
        rgb_result = self.ae_predictor.predict(rgb_image_path)

        # 최종 판정
        verdict, verdict_en, confidence, details = self._combine_results(ct_result, rgb_result)

        return {
            "verdict": verdict,
            "verdict_en": verdict_en,
            "confidence": confidence,
            "ct_result": ct_result,
            "rgb_result": rgb_result,
            "details": details
        }

    def _combine_results(self, ct_result: Dict, rgb_result: Dict) -> Tuple[str, str, float, Dict]:
        """
        CT와 RGB 결과 결합

        Returns:
            (verdict_kr, verdict_en, confidence, details)
        """
        ct_is_defect = ct_result['is_defect']
        rgb_is_defect = rgb_result['is_defect']

        # 상세 정보
        details = {
            "ct_class": ct_result['class_name'],
            "ct_defect_prob": ct_result['defect_probability'],
            "rgb_anomaly_score": rgb_result['anomaly_score'],
            "rgb_threshold": rgb_result['threshold'],
            "ct_is_defect": ct_is_defect,
            "rgb_is_defect": rgb_is_defect
        }

        # 판정 로직
        if not ct_is_defect and not rgb_is_defect:
            # 둘 다 정상
            verdict = self.VERDICT_NORMAL
            verdict_en = "normal"
            # 신뢰도: 두 모델의 정상 확신도 평균
            confidence = (1 - ct_result['defect_probability'] + rgb_result['confidence']) / 2

        elif ct_is_defect and not rgb_is_defect:
            # CT만 불량 → 내부 불량
            verdict = self.VERDICT_INTERNAL_DEFECT
            verdict_en = "internal_defect"
            confidence = ct_result['confidence']
            details["defect_type"] = ct_result['class_name']

        elif not ct_is_defect and rgb_is_defect:
            # RGB만 불량 → 외부 불량
            verdict = self.VERDICT_EXTERNAL_DEFECT
            verdict_en = "external_defect"
            confidence = rgb_result['confidence']
            details["defect_type"] = "외관 이상 (오염/손상)"

        else:
            # 둘 다 불량 → 복합 불량
            verdict = self.VERDICT_COMPLEX_DEFECT
            verdict_en = "complex_defect"
            # 신뢰도: 두 모델의 불량 확신도 평균
            confidence = (ct_result['confidence'] + rgb_result['confidence']) / 2
            details["defect_type"] = f"{ct_result['class_name']} + 외관 이상"

        return verdict, verdict_en, float(confidence), details

    def predict_ct_only(self, ct_image_path: str) -> Dict:
        """CT 이미지만 예측 (RGB 없을 때)"""
        ct_result = self.ct_predictor.predict(ct_image_path)

        if ct_result['is_defect']:
            verdict = self.VERDICT_INTERNAL_DEFECT
            verdict_en = "internal_defect"
        else:
            verdict = self.VERDICT_NORMAL
            verdict_en = "normal"

        return {
            "verdict": verdict,
            "verdict_en": verdict_en,
            "confidence": ct_result['confidence'],
            "ct_result": ct_result,
            "rgb_result": None,
            "details": {
                "ct_class": ct_result['class_name'],
                "ct_defect_prob": ct_result['defect_probability'],
                "mode": "ct_only"
            }
        }

    def predict_rgb_only(self, rgb_image_path: str) -> Dict:
        """RGB 이미지만 예측 (CT 없을 때)"""
        rgb_result = self.ae_predictor.predict(rgb_image_path)

        if rgb_result['is_defect']:
            verdict = self.VERDICT_EXTERNAL_DEFECT
            verdict_en = "external_defect"
        else:
            verdict = self.VERDICT_NORMAL
            verdict_en = "normal"

        return {
            "verdict": verdict,
            "verdict_en": verdict_en,
            "confidence": rgb_result['confidence'],
            "ct_result": None,
            "rgb_result": rgb_result,
            "details": {
                "rgb_anomaly_score": rgb_result['anomaly_score'],
                "rgb_threshold": rgb_result['threshold'],
                "mode": "rgb_only"
            }
        }

    def get_rgb_reconstruction(self, rgb_image_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """RGB AE 재구성 결과 반환"""
        return self.ae_predictor.get_reconstruction(rgb_image_path)

    def predict_with_visualization(
        self,
        ct_image_path: str,
        rgb_image_path: str,
        gradcam_alpha: float = 0.4
    ) -> Dict:
        """
        시각화 포함 앙상블 예측

        Args:
            ct_image_path: CT 이미지 경로
            rgb_image_path: RGB 이미지 경로
            gradcam_alpha: Grad-CAM 오버레이 투명도

        Returns:
            {
                "verdict": "내부불량",
                "verdict_en": "internal_defect",
                "confidence": 0.85,
                "ct_result": {..., "gradcam": {...}},
                "rgb_result": {...},
                "visualizations": {
                    "ct_gradcam_overlay": np.ndarray,
                    "ct_gradcam_heatmap": np.ndarray,
                    "ct_original": np.ndarray,
                    "rgb_original": np.ndarray,
                    "rgb_reconstructed": np.ndarray,
                    "rgb_error_map": np.ndarray
                }
            }
        """
        # CT: Grad-CAM 포함 예측
        ct_result = self.ct_predictor.predict_with_gradcam(
            ct_image_path,
            target_layer='layer4',
            alpha=gradcam_alpha
        )

        # RGB: 일반 예측 + 재구성
        rgb_result = self.ae_predictor.predict(rgb_image_path)
        rgb_original, rgb_reconstructed, rgb_error_map = self.ae_predictor.get_reconstruction(rgb_image_path)

        # 최종 판정
        verdict, verdict_en, confidence, details = self._combine_results(
            {**ct_result, "gradcam": None},  # gradcam 제외하고 판정
            rgb_result
        )

        # 시각화 데이터 구성
        visualizations = {
            "ct_gradcam_overlay": ct_result["gradcam"]["overlay"],
            "ct_gradcam_heatmap": ct_result["gradcam"]["heatmap_colored"],
            "ct_original": ct_result["gradcam"]["original"],
            "rgb_original": rgb_original,
            "rgb_reconstructed": rgb_reconstructed,
            "rgb_error_map": rgb_error_map
        }

        return {
            "verdict": verdict,
            "verdict_en": verdict_en,
            "confidence": confidence,
            "ct_result": ct_result,
            "rgb_result": rgb_result,
            "details": details,
            "visualizations": visualizations
        }


def create_ensemble(
    ct_checkpoint: Optional[str] = None,
    ae_checkpoint: Optional[str] = None,
    ensemble_config: str = "training/configs/ensemble.yaml"
) -> EnsemblePredictor:
    """
    앙상블 모델 생성 헬퍼 함수

    체크포인트 경로가 None이면 자동으로 최신 체크포인트 탐색
    """
    # CT 체크포인트 - 기본 ResNet18 사용 (CBAM은 성능 하락으로 제외)
    if ct_checkpoint is None:
        # 기본 ResNet18 모델 (CBAM 아닌 것)
        ct_checkpoint = "models/ct_cnn/checkpoints/ct_unified_best_20260105_140553.pt"
        if not Path(ct_checkpoint).exists():
            ct_dir = Path("models/ct_cnn/checkpoints")
            ct_files = sorted(ct_dir.glob("ct_unified_best_*.pt"), reverse=True)
            if ct_files:
                ct_checkpoint = str(ct_files[0])
            else:
                raise FileNotFoundError("CT CNN 체크포인트를 찾을 수 없습니다.")

    # AE 체크포인트 자동 탐색
    if ae_checkpoint is None:
        ae_dir = Path("models/rgb_ae/checkpoints")
        ae_files = sorted(ae_dir.glob("rgb_ae_best_*.pt"), reverse=True)
        if ae_files:
            ae_checkpoint = str(ae_files[0])
        else:
            raise FileNotFoundError("RGB AE 체크포인트를 찾을 수 없습니다.")

    # Threshold 파일 경로
    ae_threshold = Path("models/rgb_ae/checkpoints/threshold.json")
    ae_threshold_path = str(ae_threshold) if ae_threshold.exists() else None

    return EnsemblePredictor(
        ct_checkpoint=ct_checkpoint,
        ae_checkpoint=ae_checkpoint,
        ae_threshold_path=ae_threshold_path,
        ensemble_config_path=ensemble_config
    )


# 테스트
if __name__ == "__main__":
    from pathlib import Path

    print("앙상블 테스트")
    print("=" * 60)

    # 체크포인트 확인
    ct_checkpoint = "models/ct_cnn/checkpoints/ct_unified_best_20260105_140553.pt"
    ae_dir = Path("models/rgb_ae/checkpoints")
    ae_files = sorted(ae_dir.glob("rgb_ae_best_*.pt"), reverse=True)

    if Path(ct_checkpoint).exists() and ae_files:
        ae_checkpoint = str(ae_files[0])
        print(f"CT Checkpoint: {ct_checkpoint}")
        print(f"AE Checkpoint: {ae_checkpoint}")

        # 앙상블 생성
        ensemble = EnsemblePredictor(
            ct_checkpoint=ct_checkpoint,
            ae_checkpoint=ae_checkpoint
        )

        print("\n앙상블 모델 로드 성공!")

        # 샘플 이미지로 테스트 (이미지가 있을 경우)
        # result = ensemble.predict("test_ct.png", "test_rgb.png")
        # print(result)
    else:
        print("체크포인트 파일이 없어 테스트 스킵")
        print(f"CT exists: {Path(ct_checkpoint).exists()}")
        print(f"AE files: {list(ae_files)}")
