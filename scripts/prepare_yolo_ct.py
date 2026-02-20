"""
CT 데이터를 YOLO Segmentation 포맷으로 변환하는 스크립트

YOLO seg 라벨 포맷: class_id x1 y1 x2 y2 ... (정규화 좌표, 0~1)
기존 battery_id 기반 split 파일을 재활용하여 데이터 누수 방지

사용법:
    python scripts/prepare_yolo_ct.py

출력:
    /mnt/d/yolo-ct-seg/
    ├── images/
    │   ├── train/  (심볼릭 링크)
    │   ├── val/
    │   └── test/
    ├── labels/
    │   ├── train/  (YOLO seg 포맷 txt)
    │   ├── val/
    │   └── test/
    └── dataset.yaml
"""

import json
import os
import re
import yaml
from pathlib import Path
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

# === 설정 ===
LABEL_DIR = "/mnt/d/103.배터리 불량 이미지 데이터/3.개방데이터/1.데이터/Training/02.라벨링데이터/TL_CT_Datasets_label"
VAL_LABEL_DIR = "/mnt/d/103.배터리 불량 이미지 데이터/3.개방데이터/1.데이터/Validation/02.라벨링데이터/VL_CT_Datasets_label"
SPLIT_DIR = "/home/ubuntu/projects/battery-inspection/training/data/splits/ct/resize512"
OUTPUT_DIR = "/mnt/d/yolo-ct-seg"
IMAGE_SIZE = 4000  # 원본 이미지 크기 (정규화 기준)

# YOLO 클래스 매핑 (불량 유형만 - 정상은 빈 라벨)
# 기존 5클래스에서 불량 클래스만 추출
DEFECT_CLASS_MAP = {
    "porosity": 0,      # 기공
    "resin overflow": 1, # 레진 오버플로우
    "resin_overflow": 1, # 언더스코어 변형 대응
}

# 기존 분류 라벨 → 정상/불량 구분
NORMAL_LABELS = {0, 2}  # cell_normal, module_normal
DEFECT_LABELS = {1, 3, 4}  # cell_porosity, module_porosity, module_resin_overflow


def find_json_label(image_filename: str) -> str | None:
    """이미지 파일명으로 JSON 라벨 파일 경로 찾기"""
    stem = Path(image_filename).stem  # CT_cell_pouch_101_y_033
    json_name = f"{stem}.json"

    # Training 라벨 디렉토리에서 찾기
    path = os.path.join(LABEL_DIR, json_name)
    if os.path.exists(path):
        return path

    # Validation 라벨 디렉토리에서 찾기
    if VAL_LABEL_DIR and os.path.exists(VAL_LABEL_DIR):
        path = os.path.join(VAL_LABEL_DIR, json_name)
        if os.path.exists(path):
            return path

    return None


def polygon_points_to_yolo_seg(points: list, img_w: int, img_h: int) -> list[float]:
    """
    폴리곤 points [x1, y1, x2, y2, ...] → YOLO seg 정규화 좌표
    YOLO seg 포맷: x1/w y1/h x2/w y2/h ...
    """
    coords = []
    for i in range(0, len(points), 2):
        x = points[i] / img_w
        y = points[i + 1] / img_h
        # 0~1 범위 클리핑
        x = max(0.0, min(1.0, x))
        y = max(0.0, min(1.0, y))
        coords.append(x)
        coords.append(y)
    return coords


def convert_single_file(args):
    """단일 파일 변환 (멀티프로세싱용)"""
    image_path, cls_label, split_name, output_dir = args
    image_filename = os.path.basename(image_path)
    stem = Path(image_filename).stem

    # 라벨 출력 경로
    label_out = os.path.join(output_dir, "labels", split_name, f"{stem}.txt")

    # 이미지 심볼릭 링크 경로
    img_out = os.path.join(output_dir, "images", split_name, image_filename)

    # 이미지 심볼릭 링크 생성
    if not os.path.exists(img_out):
        try:
            os.symlink(image_path, img_out)
        except FileExistsError:
            pass

    # 정상 이미지: 빈 라벨 파일 (배경)
    if cls_label in NORMAL_LABELS:
        with open(label_out, 'w') as f:
            pass  # 빈 파일
        return "normal", 0

    # 불량 이미지: JSON에서 폴리곤 추출
    json_path = find_json_label(image_filename)
    if not json_path:
        # JSON 없으면 빈 라벨 (경고)
        with open(label_out, 'w') as f:
            pass
        return "no_json", 0

    with open(json_path) as f:
        data = json.load(f)

    img_info = data.get("image_info", {})
    img_w = img_info.get("width", IMAGE_SIZE)
    img_h = img_info.get("height", IMAGE_SIZE)

    defects = data.get("defects") or []
    valid_defects = [d for d in defects if d and d.get("points")]

    if not valid_defects:
        # 불량인데 폴리곤이 없으면 빈 라벨
        with open(label_out, 'w') as f:
            pass
        return "no_polygon", 0

    lines = []
    for defect in valid_defects:
        name = defect.get("name", "").lower().strip()
        if name not in DEFECT_CLASS_MAP:
            continue

        yolo_class = DEFECT_CLASS_MAP[name]
        points = defect["points"]

        # 최소 3개 좌표쌍(6개 값) 필요
        if len(points) < 6:
            continue

        coords = polygon_points_to_yolo_seg(points, img_w, img_h)
        coord_str = " ".join(f"{c:.6f}" for c in coords)
        lines.append(f"{yolo_class} {coord_str}")

    with open(label_out, 'w') as f:
        f.write("\n".join(lines))

    return "converted", len(lines)


def main():
    print("=== CT → YOLO Segmentation 데이터 변환 ===\n")

    # 출력 디렉토리 생성
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(OUTPUT_DIR, "images", split), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, "labels", split), exist_ok=True)

    # Split 파일 로드
    split_files = {
        "train": os.path.join(SPLIT_DIR, "battery_train.txt"),
        "val": os.path.join(SPLIT_DIR, "battery_val.txt"),
        "test": os.path.join(SPLIT_DIR, "battery_test.txt"),
    }

    total_stats = defaultdict(Counter)
    tasks = []

    for split_name, split_path in split_files.items():
        with open(split_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                image_path = parts[0]
                cls_label = int(parts[1])
                tasks.append((image_path, cls_label, split_name, OUTPUT_DIR))

    print(f"총 {len(tasks)}개 파일 변환 시작...\n")

    # 멀티프로세싱으로 변환
    converted_count = 0
    defect_count = 0

    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(convert_single_file, t): t for t in tasks}
        for i, future in enumerate(as_completed(futures)):
            task_args = futures[future]
            split_name = task_args[2]
            status, n_defects = future.result()
            total_stats[split_name][status] += 1
            if status == "converted":
                defect_count += n_defects
            converted_count += 1

            if converted_count % 20000 == 0:
                print(f"  진행: {converted_count}/{len(tasks)} ({converted_count/len(tasks)*100:.1f}%)")

    # 통계 출력
    print(f"\n=== 변환 완료 ===")
    for split_name in ["train", "val", "test"]:
        stats = total_stats[split_name]
        total = sum(stats.values())
        print(f"\n{split_name} ({total}개):")
        print(f"  정상 (빈 라벨): {stats['normal']}")
        print(f"  불량 (폴리곤 변환): {stats['converted']}")
        if stats['no_json']:
            print(f"  ⚠️ JSON 없음: {stats['no_json']}")
        if stats['no_polygon']:
            print(f"  ⚠️ 폴리곤 없음: {stats['no_polygon']}")

    # dataset.yaml 생성
    dataset_config = {
        "path": OUTPUT_DIR,
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {
            0: "porosity",
            1: "resin_overflow",
        },
    }

    yaml_path = os.path.join(OUTPUT_DIR, "dataset.yaml")
    with open(yaml_path, 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False, allow_unicode=True)

    print(f"\n✅ dataset.yaml 생성: {yaml_path}")
    print(f"✅ 총 결함 어노테이션: {defect_count}개")


if __name__ == "__main__":
    main()
