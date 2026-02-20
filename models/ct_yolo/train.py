"""
YOLOv12 Segmentation 학습 스크립트 - CT 배터리 결함 탐지

사용법:
    python models/ct_yolo/train.py

모델 저장: /mnt/d/yolo-ct-seg/runs/
"""

import os

# YOLO 모델/캐시를 D드라이브에 저장
os.environ["YOLO_CONFIG_DIR"] = "/mnt/d/yolo_config"

from ultralytics import YOLO


def main():
    # === 설정 ===
    DATASET_YAML = "/mnt/d/yolo-ct-seg/dataset.yaml"
    PROJECT_DIR = "/mnt/d/yolo-ct-seg/runs"
    MODEL_NAME = "yolo12s-seg"
    EXPERIMENT_NAME = "ct_yolo12_seg_v2"

    # === 모델 로드 (seg yaml + detection pretrained backbone) ===
    print(f"모델 로드: {MODEL_NAME} (detection backbone transfer)")
    model = YOLO(f"{MODEL_NAME}.yaml").load("yolo12s.pt")

    # === 학습 (YOLO 기본값 기반) ===
    print(f"\n학습 시작: {EXPERIMENT_NAME}")
    print(f"데이터셋: {DATASET_YAML}")
    print(f"저장 경로: {PROJECT_DIR}/{EXPERIMENT_NAME}\n")

    results = model.train(
        data=DATASET_YAML,
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,

        # 저장
        project=PROJECT_DIR,
        name=EXPERIMENT_NAME,
        save=True,
        save_period=10,
        plots=True,

        # 기타
        workers=4,
        patience=100,
        amp=True,
        verbose=True,
    )

    print(f"\n✅ 학습 완료!")
    print(f"결과: {PROJECT_DIR}/{EXPERIMENT_NAME}")

    # === Test 평가 ===
    print("\n=== Test 데이터 평가 ===")
    metrics = model.val(
        data=DATASET_YAML,
        split="test",
        imgsz=640,
        batch=16,
        device=0,
        project=PROJECT_DIR,
        name=f"{EXPERIMENT_NAME}_test",
        plots=True,
    )

    print(f"\n=== 결과 ===")
    print(f"mAP50 (Box): {metrics.box.map50:.4f}")
    print(f"mAP50-95 (Box): {metrics.box.map:.4f}")
    print(f"mAP50 (Seg): {metrics.seg.map50:.4f}")
    print(f"mAP50-95 (Seg): {metrics.seg.map:.4f}")


if __name__ == "__main__":
    main()
