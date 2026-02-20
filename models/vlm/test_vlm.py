"""VLM 테스트 스크립트"""
import sys
from pathlib import Path

# 프로젝트 루트 추가
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def test_prompts():
    """프롬프트 테스트"""
    # prompts 모듈만 직접 import (transformers 의존성 없음)
    sys.path.insert(0, str(Path(__file__).parent))
    from prompts import BatteryDefectPrompts

    print("=" * 60)
    print("VLM 프롬프트 테스트")
    print("=" * 60)

    # CT 프롬프트
    print("\n[CT 상세 프롬프트]")
    print(BatteryDefectPrompts.get_ct_prompt(detailed=True)[:200] + "...")

    print("\n[CT 간단 프롬프트]")
    print(BatteryDefectPrompts.get_ct_prompt(detailed=False))

    # RGB 프롬프트
    print("\n[RGB 상세 프롬프트]")
    print(BatteryDefectPrompts.get_rgb_prompt(detailed=True)[:200] + "...")

    # Zero-shot 프롬프트
    print("\n[Zero-shot 프롬프트]")
    print(BatteryDefectPrompts.ZERO_SHOT_CLASSIFICATION[:200] + "...")

    print("\n✅ 프롬프트 테스트 완료!")


def test_inference_mock():
    """추론 모듈 테스트 (모델 로드 없이)"""
    print("\n" + "=" * 60)
    print("VLM 추론 모듈 테스트 (Mock)")
    print("=" * 60)

    try:
        from models.vlm.inference import VLMInference

        # 모델 크기 확인
        print("\n[지원 모델 크기]")
        for size, name in VLMInference.MODEL_SIZES.items():
            print(f"  - {size}: {name}")

        # 결함 클래스 확인
        print("\n[결함 클래스]")
        for modality, classes in VLMInference.DEFECT_CLASSES.items():
            print(f"  - {modality}: {classes}")

        print("\n✅ 추론 모듈 테스트 완료!")

    except ImportError as e:
        print(f"\n⚠️ 의존성 미설치: {e}")
        print("   설치: pip install transformers qwen-vl-utils")

        # 모델 정보만 출력
        print("\n[지원 모델 크기]")
        model_sizes = {
            '2b': 'Qwen/Qwen3-VL-2B-Instruct',
            '4b': 'Qwen/Qwen3-VL-4B-Instruct',
            '8b': 'Qwen/Qwen3-VL-8B-Instruct',
            '32b': 'Qwen/Qwen3-VL-32B-Instruct',
        }
        for size, name in model_sizes.items():
            print(f"  - {size}: {name}")


def test_inference_with_model():
    """실제 모델로 추론 테스트"""
    try:
        from models.vlm.inference import create_vlm_inference
        from PIL import Image
        import torch

        print("\n" + "=" * 60)
        print("VLM 실제 추론 테스트")
        print("=" * 60)

        # GPU 확인
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"\n사용 디바이스: {device}")

        # 모델 로드 (2B 모델로 테스트)
        print("\n모델 로드 중... (시간이 걸릴 수 있습니다)")
        vlm = create_vlm_inference(model_size='2b', device=device)

        # 모델 정보
        info = vlm.get_model_info()
        print(f"\n[모델 정보]")
        for key, value in info.items():
            print(f"  - {key}: {value}")

        # 테스트 이미지 찾기
        test_images = list(PROJECT_ROOT.glob("data/**/images/*.jpg"))[:3]

        if test_images:
            print(f"\n테스트 이미지 {len(test_images)}개 발견")

            for img_path in test_images:
                print(f"\n분석 중: {img_path.name}")
                result = vlm.analyze_image(str(img_path), modality='ct')

                print(f"  예측: {result.get('prediction')}")
                print(f"  정상 여부: {result.get('is_normal')}")
                print(f"  신뢰도: {result.get('confidence')}")
                if result.get('defect_type'):
                    print(f"  결함 유형: {result.get('defect_type')}")
        else:
            print("\n테스트할 이미지가 없습니다.")

            # 더미 이미지로 테스트
            print("더미 이미지로 테스트...")
            dummy_image = Image.new('RGB', (512, 512), color='gray')
            result = vlm.analyze_image(dummy_image, modality='ct')
            print(f"  예측: {result.get('prediction')}")

        print("\n✅ 실제 추론 테스트 완료!")

    except Exception as e:
        print(f"\n⚠️ 실제 추론 테스트 실패: {e}")
        print("(모델이 설치되지 않았을 수 있습니다)")


def test_detection_with_model():
    """Qwen3-VL BBox 탐지 테스트"""
    try:
        from models.vlm.inference import create_vlm_inference
        from PIL import Image
        import torch

        print("\n" + "=" * 60)
        print("Qwen3-VL BBox 탐지 테스트")
        print("=" * 60)

        # GPU 확인
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"\n사용 디바이스: {device}")

        # 모델 로드 (2B 모델로 테스트)
        print("\n모델 로드 중... (시간이 걸릴 수 있습니다)")
        vlm = create_vlm_inference(model_size='2b', device=device)

        # 테스트 이미지 찾기
        test_images = list(PROJECT_ROOT.glob("data/**/images/*.jpg"))[:2]

        if test_images:
            print(f"\n테스트 이미지 {len(test_images)}개 발견")

            for img_path in test_images:
                print(f"\n{'=' * 40}")
                print(f"이미지: {img_path.name}")
                print("=" * 40)

                # 1. 결함 탐지 (BBox)
                print("\n[1] 결함 탐지 (detect_defects)")
                result = vlm.detect_defects(str(img_path), modality='ct')
                detections = result.get('detections', [])
                print(f"  정상 여부: {result.get('is_normal')}")
                print(f"  탐지 수: {len(detections)}")
                for i, det in enumerate(detections):
                    print(f"  [{i+1}] {det.get('label', 'unknown')}")
                    if 'bbox_pixel' in det:
                        bbox = det['bbox_pixel']
                        print(f"      BBox (pixel): [{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}]")
                    if 'confidence' in det:
                        print(f"      신뢰도: {det['confidence']:.2f}")

                # 2. 그라운딩 분석 (텍스트 + BBox)
                print("\n[2] 그라운딩 분석 (analyze_with_grounding)")
                result = vlm.analyze_with_grounding(str(img_path), modality='ct')
                print(f"  예측: {result.get('prediction')}")
                print(f"  정상 여부: {result.get('is_normal')}")
                if result.get('detections'):
                    print(f"  탐지된 결함: {len(result['detections'])}개")
                    for det in result['detections']:
                        print(f"    - {det.get('label')}: {det.get('bbox_pixel')}")
        else:
            print("\n테스트할 이미지가 없습니다.")

            # 더미 이미지로 테스트
            print("더미 이미지로 테스트...")
            dummy_image = Image.new('RGB', (512, 512), color='gray')

            result = vlm.detect_defects(dummy_image, modality='ct')
            print(f"  정상 여부: {result.get('is_normal')}")
            print(f"  탐지 수: {len(result.get('detections', []))}")

        print("\n✅ BBox 탐지 테스트 완료!")

    except Exception as e:
        import traceback
        print(f"\n⚠️ BBox 탐지 테스트 실패: {e}")
        traceback.print_exc()


def main():
    """메인 테스트 함수"""
    import argparse

    parser = argparse.ArgumentParser(description='VLM 테스트')
    parser.add_argument('--full', action='store_true', help='모델 로드 포함 전체 테스트')
    parser.add_argument('--detection', action='store_true', help='BBox 탐지 테스트만')
    args = parser.parse_args()

    # 기본 테스트
    test_prompts()
    test_inference_mock()

    # BBox 탐지 테스트만
    if args.detection:
        test_detection_with_model()
    # 전체 테스트 (모델 로드 포함)
    elif args.full:
        test_inference_with_model()
        test_detection_with_model()
    else:
        print("\n💡 테스트 옵션:")
        print("   --full      : 모델 로드 포함 전체 테스트")
        print("   --detection : Qwen3-VL BBox 탐지 테스트만")


if __name__ == "__main__":
    main()
