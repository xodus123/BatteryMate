"""VLG 테스트 스크립트"""
import sys
from pathlib import Path

# 프로젝트 루트 추가
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def test_prompts():
    """프롬프트 테스트"""
    from models.vlg.prompts import GroundingPrompts

    print("=" * 60)
    print("VLG 프롬프트 테스트")
    print("=" * 60)

    # CT 프롬프트
    print("\n[CT Porosity 프롬프트]")
    prompts = GroundingPrompts.get_ct_prompts('porosity')
    print(f"  키워드: {prompts}")
    print(f"  Grounding 텍스트: {GroundingPrompts.to_grounding_text(prompts)}")

    print("\n[CT Resin Overflow 프롬프트]")
    prompts = GroundingPrompts.get_ct_prompts('resin_overflow')
    print(f"  키워드: {prompts}")

    print("\n[CT 전체 결함 프롬프트]")
    prompts = GroundingPrompts.get_ct_prompts('all')
    print(f"  키워드: {prompts}")

    # RGB 프롬프트
    print("\n[RGB 전체 결함 프롬프트]")
    prompts = GroundingPrompts.get_rgb_prompts('all')
    print(f"  키워드: {prompts}")

    print("\n✅ 프롬프트 테스트 완료!")


def test_inference_mock():
    """추론 모듈 테스트 (모델 로드 없이)"""
    from models.vlg.inference import VLGInference, DetectionResult

    print("\n" + "=" * 60)
    print("VLG 추론 모듈 테스트 (Mock)")
    print("=" * 60)

    # 모델 설정 확인
    print("\n[지원 모델 설정]")
    for model_type, config in VLGInference.MODEL_CONFIGS.items():
        print(f"  - {model_type}:")
        print(f"      Config: {config['config']}")
        print(f"      Weights: {config['weights']}")

    # DetectionResult 테스트
    print("\n[DetectionResult 테스트]")
    result = DetectionResult(
        boxes=[[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]],
        labels=['porosity', 'bubble'],
        scores=[0.85, 0.72],
        phrases=['porosity defect', 'gas bubble'],
    )
    print(f"  박스 수: {len(result.boxes)}")
    print(f"  라벨: {result.labels}")
    print(f"  점수: {result.scores}")

    print("\n✅ 추론 모듈 테스트 완료!")


def test_inference_with_model():
    """실제 모델로 추론 테스트"""
    try:
        from models.vlg.inference import create_vlg_inference
        from PIL import Image
        import torch

        print("\n" + "=" * 60)
        print("VLG 실제 추론 테스트")
        print("=" * 60)

        # GPU 확인
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"\n사용 디바이스: {device}")

        # 모델 로드
        print("\n모델 로드 중...")
        vlg = create_vlg_inference(model_type='swinT', device=device)

        # 모델 정보
        info = vlg.get_model_info()
        print(f"\n[모델 정보]")
        for key, value in info.items():
            print(f"  - {key}: {value}")

        if not info['model_loaded']:
            print("\n⚠️ GroundingDINO 모델이 로드되지 않았습니다.")
            print("   설치: pip install groundingdino")
            return

        # 테스트 이미지 찾기
        test_images = list(PROJECT_ROOT.glob("data/**/images/*.jpg"))[:3]

        if test_images:
            print(f"\n테스트 이미지 {len(test_images)}개 발견")

            for img_path in test_images:
                print(f"\n탐지 중: {img_path.name}")
                result = vlg.analyze_image(str(img_path), modality='ct')

                print(f"  예측: {result.get('prediction')}")
                print(f"  결함 수: {result.get('num_defects')}")
                if result.get('defect_types'):
                    print(f"  결함 유형: {result.get('defect_types')}")
                print(f"  신뢰도: {result.get('confidence'):.2f}")

                # 시각화 (선택)
                # visualized = vlg.visualize(str(img_path), detection)
                # visualized.save(f"/tmp/vlg_result_{img_path.stem}.jpg")
        else:
            print("\n테스트할 이미지가 없습니다.")

        print("\n✅ 실제 추론 테스트 완료!")

    except Exception as e:
        print(f"\n⚠️ 실제 추론 테스트 실패: {e}")
        print("(GroundingDINO가 설치되지 않았을 수 있습니다)")


def test_visualization():
    """시각화 테스트"""
    from models.vlg.inference import VLGInference, DetectionResult
    from PIL import Image

    print("\n" + "=" * 60)
    print("VLG 시각화 테스트")
    print("=" * 60)

    # 더미 이미지 생성
    dummy_image = Image.new('RGB', (512, 512), color='gray')

    # 더미 탐지 결과
    detection = DetectionResult(
        boxes=[[0.1, 0.1, 0.3, 0.3], [0.6, 0.6, 0.9, 0.9]],
        labels=['porosity', 'resin_overflow'],
        scores=[0.92, 0.78],
        phrases=['porosity defect', 'resin overflow'],
    )

    # VLG 인스턴스 (모델 없이)
    vlg = VLGInference.__new__(VLGInference)
    vlg.model = None
    vlg.prompts = None

    # 시각화
    output_path = "/tmp/vlg_visualization_test.jpg"
    result_image = vlg.visualize(dummy_image, detection, output_path=output_path)

    print(f"  시각화 이미지 크기: {result_image.size}")
    print(f"  저장 경로: {output_path}")

    print("\n✅ 시각화 테스트 완료!")


def main():
    """메인 테스트 함수"""
    import argparse

    parser = argparse.ArgumentParser(description='VLG 테스트')
    parser.add_argument('--full', action='store_true', help='모델 로드 포함 전체 테스트')
    parser.add_argument('--viz', action='store_true', help='시각화 테스트')
    args = parser.parse_args()

    # 기본 테스트
    test_prompts()
    test_inference_mock()

    # 시각화 테스트
    if args.viz:
        test_visualization()

    # 전체 테스트 (모델 로드 포함)
    if args.full:
        test_inference_with_model()
    else:
        print("\n💡 전체 테스트를 실행하려면 --full 옵션을 사용하세요")
        print("   python models/vlg/test_vlg.py --full")
        print("   python models/vlg/test_vlg.py --viz  # 시각화 테스트")


if __name__ == "__main__":
    main()
