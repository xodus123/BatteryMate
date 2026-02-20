"""
YOLOv12 Segmentation 테스트 스크립트 - CT 배터리 결함 탐지

사용법:
    # 기본 (최신 best 모델)
    python models/ct_yolo/test.py

    # 특정 모델 지정
    python models/ct_yolo/test.py --weights /mnt/d/yolo-ct-seg/runs/ct_yolo12_seg_v2/weights/best.pt

    # Val 데이터로 평가
    python models/ct_yolo/test.py --split val
"""

import os
import argparse
from pathlib import Path

os.environ["YOLO_CONFIG_DIR"] = "/mnt/d/yolo_config"

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="YOLOv12-seg CT 테스트")
    parser.add_argument("--weights", type=str, default=None,
                        help="모델 가중치 경로 (.pt)")
    parser.add_argument("--data", type=str, default="/mnt/d/yolo-ct-seg/dataset.yaml",
                        help="데이터셋 YAML 경로")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"],
                        help="평가 데이터 split")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="이미지 크기")
    parser.add_argument("--batch", type=int, default=16,
                        help="배치 크기")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7,
                        help="IoU threshold for NMS")
    parser.add_argument("--device", type=int, default=0,
                        help="GPU 디바이스 번호")
    args = parser.parse_args()

    # 모델 가중치 자동 탐색
    if args.weights is None:
        runs_dir = Path("/mnt/d/yolo-ct-seg/runs")
        candidates = sorted(runs_dir.glob("ct_yolo12_seg*/weights/best.pt"),
                            key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            print("❌ best.pt를 찾을 수 없습니다. --weights로 직접 지정하세요.")
            return
        args.weights = str(candidates[0])
        print(f"자동 탐색된 모델: {args.weights}")

    # 모델 로드
    print(f"\n모델 로드: {args.weights}")
    model = YOLO(args.weights)

    # 평가 실행
    print(f"\n=== {args.split.upper()} 데이터 평가 ===")
    print(f"데이터셋: {args.data}")
    print(f"이미지 크기: {args.imgsz}")
    print(f"Confidence: {args.conf}")
    print(f"IoU: {args.iou}\n")

    metrics = model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        project="/mnt/d/yolo-ct-seg/runs",
        name=f"test_{args.split}",
        plots=True,
        save_json=True,
    )

    # 결과 출력
    print(f"\n{'='*60}")
    print(f"  YOLOv12-seg CT 테스트 결과 ({args.split})")
    print(f"{'='*60}")

    print(f"\n📦 Box Detection:")
    print(f"  Precision:  {metrics.box.mp:.4f}")
    print(f"  Recall:     {metrics.box.mr:.4f}")
    print(f"  mAP50:      {metrics.box.map50:.4f}")
    print(f"  mAP50-95:   {metrics.box.map:.4f}")

    print(f"\n🎭 Segmentation:")
    print(f"  Precision:  {metrics.seg.mp:.4f}")
    print(f"  Recall:     {metrics.seg.mr:.4f}")
    print(f"  mAP50:      {metrics.seg.map50:.4f}")
    print(f"  mAP50-95:   {metrics.seg.map:.4f}")

    # 클래스별 결과
    class_names = model.names
    print(f"\n📊 클래스별 mAP50:")
    for i, name in class_names.items():
        box_ap = metrics.box.ap50[i] if i < len(metrics.box.ap50) else 0
        seg_ap = metrics.seg.ap50[i] if i < len(metrics.seg.ap50) else 0
        print(f"  {name:20s}  Box: {box_ap:.4f}  Seg: {seg_ap:.4f}")

    print(f"\n✅ 결과 저장: /mnt/d/yolo-ct-seg/runs/test_{args.split}/")


if __name__ == "__main__":
    main()
