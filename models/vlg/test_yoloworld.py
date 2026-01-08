"""YOLO-World VLG 테스트 스크립트"""
import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
_project_root = Path(__file__).parent.parent.parent.absolute()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import argparse
from pathlib import Path


def test_model_loading():
    """모델 로딩 테스트"""
    print("\n[1] 모델 로딩 테스트")
    print("-" * 40)

    try:
        from models.vlg.inference_yoloworld import YOLOWorldInference

        # 각 모델 크기 테스트
        for model_type in ['yolov8s-world']:  # 기본은 small만 테스트
            print(f"  Loading {model_type}...", end=" ")
            vlg = YOLOWorldInference(model_type=model_type, device='cuda')
            if vlg.model is not None:
                print("✅ 성공")
            else:
                print("❌ 실패")
            del vlg

        return True
    except Exception as e:
        print(f"❌ 에러: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_class_setting():
    """클래스 설정 테스트"""
    print("\n[2] 클래스 설정 테스트")
    print("-" * 40)

    try:
        from models.vlg.inference_yoloworld import YOLOWorldInference

        vlg = YOLOWorldInference(model_type='yolov8s-world')

        # CT 모달리티 클래스
        vlg._set_classes('ct')
        print(f"  CT 클래스: {len(vlg.current_classes)}개")
        print(f"    {vlg.current_classes[:5]}...")

        # RGB 모달리티 클래스
        vlg._set_classes('rgb')
        print(f"  RGB 클래스: {len(vlg.current_classes)}개")
        print(f"    {vlg.current_classes[:5]}...")

        print("✅ 클래스 설정 성공")
        return True
    except Exception as e:
        print(f"❌ 에러: {e}")
        return False


def test_detection(image_path: str = None, modality: str = 'ct'):
    """탐지 테스트"""
    print("\n[3] 탐지 테스트")
    print("-" * 40)

    try:
        from models.vlg.inference_yoloworld import YOLOWorldInference
        from PIL import Image

        vlg = YOLOWorldInference(model_type='yolov8s-world')

        if image_path and Path(image_path).exists():
            print(f"  이미지: {image_path}")
            result = vlg.detect(image_path, modality=modality)
        else:
            # 더미 이미지 생성
            print("  더미 이미지 사용 (256x256)")
            import numpy as np
            dummy_image = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
            result = vlg.detect(dummy_image, modality=modality)

        print(f"  탐지 결과:")
        print(f"    - 박스 수: {len(result.boxes)}")
        print(f"    - 라벨: {result.labels[:5] if result.labels else 'None'}")
        print(f"    - 점수: {[f'{s:.3f}' for s in result.scores[:5]] if result.scores else 'None'}")

        print("✅ 탐지 테스트 성공")
        return True
    except Exception as e:
        print(f"❌ 에러: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_analyze(image_path: str = None, modality: str = 'ct'):
    """분석 테스트"""
    print("\n[4] 분석 테스트")
    print("-" * 40)

    try:
        from models.vlg.inference_yoloworld import YOLOWorldInference
        from PIL import Image

        vlg = YOLOWorldInference(model_type='yolov8s-world')

        if image_path and Path(image_path).exists():
            print(f"  이미지: {image_path}")
            result = vlg.analyze_image(image_path, modality=modality)
        else:
            print("  더미 이미지 사용")
            import numpy as np
            dummy_image = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
            result = vlg.analyze_image(dummy_image, modality=modality)

        print(f"  분석 결과:")
        print(f"    - 예측: {result['prediction']}")
        print(f"    - 정상 여부: {result['is_normal']}")
        print(f"    - 결함 수: {result['num_defects']}")
        print(f"    - 결함 유형: {result['defect_types']}")
        print(f"    - 신뢰도: {result['confidence']:.4f}")
        print(f"    - 모델: {result['model']}")

        print("✅ 분석 테스트 성공")
        return True
    except Exception as e:
        print(f"❌ 에러: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_visualization(image_path: str = None, modality: str = 'ct', save_path: str = None):
    """시각화 테스트"""
    print("\n[5] 시각화 테스트")
    print("-" * 40)

    try:
        from models.vlg.inference_yoloworld import YOLOWorldInference
        from PIL import Image

        vlg = YOLOWorldInference(model_type='yolov8s-world')

        if image_path and Path(image_path).exists():
            print(f"  이미지: {image_path}")
            vis_image = vlg.visualize(image_path, modality=modality)
        else:
            print("  더미 이미지 사용")
            import numpy as np
            dummy_image = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
            vis_image = vlg.visualize(dummy_image, modality=modality)

        if save_path:
            vis_image.save(save_path)
            print(f"  저장: {save_path}")

        print(f"  결과 이미지 크기: {vis_image.size}")
        print("✅ 시각화 테스트 성공")
        return True
    except Exception as e:
        print(f"❌ 에러: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_comparison():
    """GroundingDINO와 비교 테스트"""
    print("\n[6] GroundingDINO 비교 테스트")
    print("-" * 40)

    try:
        from models.vlg.inference_yoloworld import YOLOWorldInference
        from PIL import Image
        import numpy as np

        vlg = YOLOWorldInference(model_type='yolov8s-world')

        # 더미 이미지
        dummy_image = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))

        result = vlg.compare_with_groundingdino(dummy_image, modality='ct')

        print(f"  YOLO-World:")
        print(f"    - 탐지 수: {result['yolo_world']['num_detections']}")

        if result['groundingdino']:
            print(f"  GroundingDINO:")
            print(f"    - 탐지 수: {result['groundingdino']['num_detections']}")
        else:
            print(f"  GroundingDINO: 비교 불가")

        print("✅ 비교 테스트 완료")
        return True
    except Exception as e:
        print(f"❌ 에러: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='YOLO-World VLG 테스트')
    parser.add_argument('--image', type=str, help='테스트 이미지 경로')
    parser.add_argument('--modality', type=str, default='ct', choices=['ct', 'rgb', 'all'])
    parser.add_argument('--save', type=str, help='시각화 결과 저장 경로')
    parser.add_argument('--full', action='store_true', help='전체 테스트 실행')
    parser.add_argument('--compare', action='store_true', help='GroundingDINO 비교')
    args = parser.parse_args()

    print("=" * 60)
    print("YOLO-World VLG 테스트")
    print("=" * 60)

    results = {}

    # 기본 테스트
    results['loading'] = test_model_loading()

    if args.full or args.image:
        results['class_setting'] = test_class_setting()
        results['detection'] = test_detection(args.image, args.modality)
        results['analyze'] = test_analyze(args.image, args.modality)
        results['visualization'] = test_visualization(args.image, args.modality, args.save)

        if args.compare:
            results['comparison'] = test_comparison()

    # 결과 요약
    print("\n" + "=" * 60)
    print("테스트 결과 요약")
    print("=" * 60)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {name}: {status}")

    print(f"\n총 {passed}/{total} 테스트 통과")

    if passed == total:
        print("\n✅ 모든 테스트 통과!")
    else:
        print("\n⚠️ 일부 테스트 실패")


if __name__ == "__main__":
    main()
