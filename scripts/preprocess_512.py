"""
CT 이미지 전처리 스크립트 (512 Resize)
- 원본 4000x4000 이미지를 512x512로 resize
- 크롭 없이 원본 비율 유지 (정사각형 → 정사각형)
- 배터리 ID 기준 Train/Val/Test 분할
"""

import os
import json
import argparse
import random
from pathlib import Path
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from collections import defaultdict, Counter
import re


# 설정
DEFAULT_IMAGE_SIZE = 512
DEFAULT_OUTPUT_DIR = "/mnt/d/battery-512"

# 데이터 경로
LABEL_BASE = "/mnt/d/103.배터리 불량 이미지 데이터/3.개방데이터/1.데이터"
IMAGE_BASE = "/home/ubuntu/battery-data/103.배터리 불량 이미지 데이터/3.개방데이터/1.데이터"

SPLITS_PATH = "/home/ubuntu/projects/battery-inspection/training/data/splits/ct/resize512"


def get_label_path(image_path: str) -> str:
    """이미지 경로에서 라벨 JSON 경로 생성"""
    filename = os.path.basename(image_path)
    base_name = os.path.splitext(filename)[0]
    json_name = base_name + '.json'

    if 'Training' in image_path:
        label_dir = os.path.join(LABEL_BASE, 'Training/02.라벨링데이터/TL_CT_Datasets_label')
    elif 'Validation' in image_path:
        label_dir = os.path.join(LABEL_BASE, 'Validation/02.라벨링데이터/VL_CT_Datasets_label')
    else:
        return None

    label_path = os.path.join(label_dir, json_name)
    return label_path if os.path.exists(label_path) else None


def load_label_info(label_path: str) -> dict:
    """라벨 JSON에서 클래스 정보 로드"""
    try:
        with open(label_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        data_info = data.get('data_info', {})
        image_info = data.get('image_info', {})
        defects = data.get('defects')

        battery_type = data_info.get('type', '')  # 'cell' or 'module'
        is_normal = image_info.get('is_normal', True)

        defect_type = None
        if defects and len(defects) > 0:
            defect_type = defects[0].get('name', '').lower()

        # 클래스 라벨 결정
        # 0: cell_normal, 1: cell_porosity, 2: module_normal,
        # 3: module_porosity, 4: module_resin_overflow
        if battery_type == 'cell':
            if is_normal or defect_type is None:
                label = 0
            elif 'porosity' in defect_type:
                label = 1
            else:
                label = 0
        else:  # module
            if is_normal or defect_type is None:
                label = 2
            elif 'porosity' in defect_type:
                label = 3
            elif 'resin' in defect_type:
                label = 4
            else:
                label = 2

        return {
            'label': label,
            'battery_type': battery_type,
            'is_normal': is_normal,
            'defect_type': defect_type
        }
    except Exception as e:
        return None


def extract_battery_id(filename: str) -> str:
    """파일명에서 배터리 ID 추출"""
    # CT_module_pouch_015_x_128 → module_pouch_015
    name = filename.replace('CT_', '')
    parts = name.split('_')
    if len(parts) >= 3:
        return f"{parts[0]}_{parts[1]}_{parts[2]}"
    return None


def preprocess_image(src_path: str, dst_path: str, image_size: int = 512):
    """단일 이미지 전처리 (512 resize)"""
    try:
        # 라벨 정보 로드
        label_path = get_label_path(src_path)
        label_info = None
        label = None
        if label_path:
            label_info = load_label_info(label_path)
            if label_info:
                label = label_info['label']

        # 이미지 로드 및 resize
        img = Image.open(src_path)

        # RGB 변환
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # 512x512로 resize (원본이 정사각형이므로 비율 유지)
        img = img.resize((image_size, image_size), Image.LANCZOS)

        # 저장
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        img.save(dst_path, 'JPEG', quality=95)

        return True, "OK", label
    except Exception as e:
        return False, str(e), None


def collect_all_images():
    """원본 이미지 경로 수집"""
    images = []

    # Training
    train_dirs = [
        os.path.join(IMAGE_BASE, 'Training/01.원천데이터/TS_CT_Datasets_images_1'),
        os.path.join(IMAGE_BASE, 'Training/01.원천데이터/TS_CT_Datasets_images_2'),
    ]

    for dir_path in train_dirs:
        if os.path.exists(dir_path):
            for f in os.listdir(dir_path):
                if f.endswith(('.jpg', '.png', '.jpeg')):
                    images.append((os.path.join(dir_path, f), 'Training'))

    # Validation
    val_dirs = [
        os.path.join(IMAGE_BASE, 'Validation/01.원천데이터/VS_CT_Datasets_images'),
    ]

    for dir_path in val_dirs:
        if os.path.exists(dir_path):
            for f in os.listdir(dir_path):
                if f.endswith(('.jpg', '.png', '.jpeg')):
                    images.append((os.path.join(dir_path, f), 'Validation'))

    return images


def main():
    parser = argparse.ArgumentParser(description='CT 이미지 전처리 (512 resize)')
    parser.add_argument('--size', type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--skip-existing', action='store_true')

    args = parser.parse_args()

    print("=" * 60)
    print("CT 이미지 전처리 (원본 → 512 resize)")
    print("=" * 60)
    print(f"출력 크기: {args.size}x{args.size}")
    print(f"출력 경로: {args.output}")
    print("=" * 60)

    # 이미지 수집
    print("\n이미지 수집 중...")
    all_images = collect_all_images()
    print(f"총 {len(all_images):,}개 이미지 발견")

    # 작업 목록 생성
    tasks = []
    for img_path, split_type in all_images:
        filename = os.path.basename(img_path)
        base_name = os.path.splitext(filename)[0]
        dst_path = os.path.join(args.output, split_type, base_name + '.jpg')

        if args.skip_existing and os.path.exists(dst_path):
            continue

        tasks.append((img_path, dst_path, split_type))

    print(f"처리할 이미지: {len(tasks):,}개")

    if not tasks:
        print("처리할 이미지가 없습니다.")
        return

    # 병렬 처리
    print(f"\n전처리 시작 (workers: {args.workers})...")
    success_count = 0
    fail_count = 0
    results = []  # (dst_path, label, battery_id)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(preprocess_image, src, dst, args.size): (src, dst)
            for src, dst, _ in tasks
        }

        with tqdm(total=len(tasks), desc="Processing") as pbar:
            for future in as_completed(futures):
                src, dst = futures[future]
                success, msg, label = future.result()
                if success:
                    success_count += 1
                    if label is not None:
                        battery_id = extract_battery_id(os.path.basename(src))
                        results.append((dst, label, battery_id))
                else:
                    fail_count += 1
                pbar.update(1)

    print(f"\n완료: {success_count:,}개 성공, {fail_count}개 실패")

    # 배터리 단위 Split 생성
    print("\n배터리 단위 Split 생성 중...")

    battery_groups = defaultdict(list)
    battery_main_class = {}

    for dst, label, battery_id in results:
        if battery_id:
            battery_groups[battery_id].append((dst, label))

    # 각 배터리의 주요 클래스
    for battery_id, items in battery_groups.items():
        labels = [label for _, label in items]
        main_class = Counter(labels).most_common(1)[0][0]
        battery_main_class[battery_id] = main_class

    # 클래스별 배터리 분류
    class_battery_ids = defaultdict(list)
    for battery_id, main_class in battery_main_class.items():
        class_battery_ids[main_class].append(battery_id)

    print(f"고유 배터리 수: {len(battery_groups)}개")
    print("클래스별 배터리:")
    for cls, ids in sorted(class_battery_ids.items()):
        print(f"  Class {cls}: {len(ids)}개")

    # Stratified Split (70/15/15)
    random.seed(42)
    train_ids = set()
    val_ids = set()
    test_ids = set()

    for class_id, ids in class_battery_ids.items():
        random.shuffle(ids)
        n = len(ids)
        n_train = int(n * 0.7)
        n_val = int(n * 0.15)

        train_ids.update(ids[:n_train])
        val_ids.update(ids[n_train:n_train + n_val])
        test_ids.update(ids[n_train + n_val:])

    print(f"\n배터리 분할: Train={len(train_ids)}, Val={len(val_ids)}, Test={len(test_ids)}")

    # 파일 분류
    train_items = []
    val_items = []
    test_items = []

    for battery_id, items in battery_groups.items():
        if battery_id in train_ids:
            train_items.extend(items)
        elif battery_id in val_ids:
            val_items.extend(items)
        else:
            test_items.extend(items)

    print(f"파일 분할: Train={len(train_items):,}, Val={len(val_items):,}, Test={len(test_items):,}")

    # Split 파일 저장
    os.makedirs(SPLITS_PATH, exist_ok=True)

    for name, items in [('battery_train.txt', train_items),
                        ('battery_val.txt', val_items),
                        ('battery_test.txt', test_items)]:
        filepath = os.path.join(SPLITS_PATH, name)
        with open(filepath, 'w') as f:
            for dst, label in items:
                f.write(f"{dst}\t{label}\n")
        print(f"저장: {filepath}")

    # 검증
    print("\n데이터 누수 검증...")
    train_bat = set(extract_battery_id(os.path.basename(p)) for p, _ in train_items)
    val_bat = set(extract_battery_id(os.path.basename(p)) for p, _ in val_items)
    test_bat = set(extract_battery_id(os.path.basename(p)) for p, _ in test_items)

    if not (train_bat & val_bat) and not (train_bat & test_bat) and not (val_bat & test_bat):
        print("✅ 검증 통과: 데이터 누수 없음")
    else:
        print("❌ 데이터 누수 발견!")

    print("\n" + "=" * 60)
    print("전처리 완료!")
    print(f"출력 경로: {args.output}")
    print(f"Split 경로: {SPLITS_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
