"""
CT 이미지 전처리 스크립트
- JSON 라벨에서 battery_outline 읽어서 배터리 영역만 crop
- 워터마크/프레임 제거
- 1024x1024로 리사이즈 후 저장
"""

import os
import json
import argparse
from pathlib import Path
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from typing import List, Tuple, Optional


# 설정
DEFAULT_IMAGE_SIZE = 1024
DEFAULT_OUTPUT_DIR = "/mnt/d/battery-cropped"

# 데이터 경로
# 원본 4000x4000 이미지 사용 (좌표 스케일링 불필요)
LABEL_BASE = "/mnt/d/103.배터리 불량 이미지 데이터/3.개방데이터/1.데이터"
IMAGE_BASE = "/home/ubuntu/battery-data/103.배터리 불량 이미지 데이터/3.개방데이터/1.데이터"

# 라벨 폴더 매핑
LABEL_DIRS = {
    'Training': 'Training/02.라벨링데이터/TL_CT_Datasets_label',
    'Validation': 'Validation/02.라벨링데이터/VL_CT_Datasets_label'
}

SPLITS_PATH = "/home/ubuntu/projects/battery-inspection/training/data/splits/ct"


def polygon_to_bbox(points: List[float], padding: int = 10) -> Tuple[int, int, int, int]:
    """
    폴리곤 좌표를 bounding box로 변환

    Args:
        points: [x1, y1, x2, y2, ...] 형태의 좌표 리스트
        padding: bbox 주변 여백 (픽셀)

    Returns:
        (x_min, y_min, x_max, y_max)
    """
    x_coords = points[0::2]  # 짝수 인덱스 (x 좌표)
    y_coords = points[1::2]  # 홀수 인덱스 (y 좌표)

    x_min = max(0, int(min(x_coords)) - padding)
    y_min = max(0, int(min(y_coords)) - padding)
    x_max = int(max(x_coords)) + padding
    y_max = int(max(y_coords)) + padding

    return x_min, y_min, x_max, y_max


def get_label_path(image_path: str) -> Optional[str]:
    """이미지 경로에서 라벨 JSON 경로 생성"""
    # 이미지 파일명 추출
    filename = os.path.basename(image_path)
    base_name = os.path.splitext(filename)[0]
    json_name = base_name + '.json'

    # Training/Validation 구분
    if 'Training' in image_path:
        label_dir = os.path.join(LABEL_BASE, LABEL_DIRS['Training'])
    elif 'Validation' in image_path:
        label_dir = os.path.join(LABEL_BASE, LABEL_DIRS['Validation'])
    else:
        return None

    label_path = os.path.join(label_dir, json_name)
    return label_path if os.path.exists(label_path) else None


def load_label_info(label_path: str) -> Optional[dict]:
    """
    라벨 JSON에서 battery_outline과 클래스 정보 로드

    Returns:
        {
            'outline': [...],  # battery_outline 좌표
            'label': int,      # 클래스 라벨 (0-4)
            'is_normal': bool,
            'defect_type': str or None
        }
    """
    try:
        with open(label_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # battery_outline 추출
        outline = data.get('swelling', {}).get('battery_outline')
        if not outline or len(outline) < 6:
            outline = None

        # 클래스 정보 추출
        data_info = data.get('data_info', {})
        image_info = data.get('image_info', {})
        defects = data.get('defects')

        battery_type = data_info.get('type', '')  # 'cell' or 'module'
        is_normal = image_info.get('is_normal', True)

        # 결함 타입 확인
        defect_type = None
        if defects and len(defects) > 0:
            defect_type = defects[0].get('name', '').lower()

        # 클래스 라벨 결정
        # 0: cell_normal, 1: cell_porosity, 2: module_normal,
        # 3: module_porosity, 4: module_resin_overflow
        if battery_type == 'cell':
            if is_normal or defect_type is None:
                label = 0  # cell_normal
            elif 'porosity' in defect_type:
                label = 1  # cell_porosity
            else:
                label = 0  # unknown -> normal
        else:  # module
            if is_normal or defect_type is None:
                label = 2  # module_normal
            elif 'porosity' in defect_type:
                label = 3  # module_porosity
            elif 'resin' in defect_type:
                label = 4  # module_resin_overflow
            else:
                label = 2  # unknown -> normal

        # 원본 이미지 크기 (좌표 스케일링용)
        orig_width = image_info.get('width', 4000)
        orig_height = image_info.get('height', 4000)

        return {
            'outline': outline,
            'label': label,
            'is_normal': is_normal,
            'defect_type': defect_type,
            'battery_type': battery_type,
            'orig_width': orig_width,
            'orig_height': orig_height
        }
    except Exception as e:
        return None


def preprocess_image(
    src_path: str,
    dst_path: str,
    image_size: int = 1024
) -> Tuple[bool, str, Optional[int]]:
    """
    단일 이미지 전처리 (battery_outline 기반 crop)

    Returns:
        (성공여부, 메시지, 라벨)
    """
    try:
        # 라벨 경로 찾기
        label_path = get_label_path(src_path)

        # 라벨 정보 로드
        label_info = None
        label = None
        if label_path:
            label_info = load_label_info(label_path)
            if label_info:
                label = label_info['label']

        # 이미지 로드
        img = Image.open(src_path)
        original_size = img.size

        # battery_outline으로 crop
        if label_info and label_info['outline']:
            # 원본 좌표 기준 bbox
            bbox = polygon_to_bbox(label_info['outline'], padding=20)
            x_min, y_min, x_max, y_max = bbox

            # 좌표 스케일링 (라벨은 원본 크기 기준, 이미지는 리사이즈된 상태일 수 있음)
            orig_w = label_info.get('orig_width', 4000)
            orig_h = label_info.get('orig_height', 4000)

            if original_size[0] != orig_w or original_size[1] != orig_h:
                # 스케일 팩터 계산
                scale_x = original_size[0] / orig_w
                scale_y = original_size[1] / orig_h

                # 좌표 스케일링
                x_min = int(x_min * scale_x)
                y_min = int(y_min * scale_y)
                x_max = int(x_max * scale_x)
                y_max = int(y_max * scale_y)

            # bbox가 이미지 범위 내인지 확인
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(x_max, original_size[0])
            y_max = min(y_max, original_size[1])

            if x_max > x_min and y_max > y_min:
                img = img.crop((x_min, y_min, x_max, y_max))

        # RGB 변환
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # 리사이즈 (비율 유지하며 패딩)
        img.thumbnail((image_size, image_size), Image.LANCZOS)

        # 정사각형 캔버스에 중앙 배치
        canvas = Image.new('RGB', (image_size, image_size), (0, 0, 0))
        x_offset = (image_size - img.size[0]) // 2
        y_offset = (image_size - img.size[1]) // 2
        canvas.paste(img, (x_offset, y_offset))

        # 디렉토리 생성 및 저장
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        canvas.save(dst_path, 'JPEG', quality=95)

        return True, "OK", label
    except Exception as e:
        return False, str(e), None


def collect_all_images() -> List[Tuple[str, str]]:
    """
    원본 이미지 폴더에서 모든 CT 이미지 경로 수집

    Returns:
        [(이미지경로, 'Training'/'Validation'), ...]
    """
    images = []

    # Training 이미지
    train_dirs = [
        os.path.join(IMAGE_BASE, 'Training/01.원천데이터/TS_CT_Datasets_images_1'),
        os.path.join(IMAGE_BASE, 'Training/01.원천데이터/TS_CT_Datasets_images_2'),
    ]

    for dir_path in train_dirs:
        if os.path.exists(dir_path):
            for f in os.listdir(dir_path):
                if f.endswith(('.jpg', '.png', '.jpeg')):
                    images.append((os.path.join(dir_path, f), 'Training'))

    # Validation 이미지
    val_dirs = [
        os.path.join(IMAGE_BASE, 'Validation/01.원천데이터/VS_CT_Datasets_images_1'),
        os.path.join(IMAGE_BASE, 'Validation/01.원천데이터/VS_CT_Datasets_images_2'),
    ]

    for dir_path in val_dirs:
        if os.path.exists(dir_path):
            for f in os.listdir(dir_path):
                if f.endswith(('.jpg', '.png', '.jpeg')):
                    images.append((os.path.join(dir_path, f), 'Validation'))

    return images


def extract_battery_id(filename: str) -> Optional[int]:
    """파일명에서 배터리 ID 추출"""
    import re
    match = re.search(r'_(\d{3})_[xyz]_', filename)
    if match:
        return int(match.group(1))
    return None


def main():
    parser = argparse.ArgumentParser(description='CT 이미지 전처리 (battery_outline crop)')
    parser.add_argument('--size', type=int, default=DEFAULT_IMAGE_SIZE,
                        help=f'리사이즈 크기 (기본: {DEFAULT_IMAGE_SIZE})')
    parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f'출력 디렉토리 (기본: {DEFAULT_OUTPUT_DIR})')
    parser.add_argument('--workers', type=int, default=8,
                        help='병렬 처리 워커 수 (기본: 8)')
    parser.add_argument('--skip-existing', action='store_true',
                        help='이미 존재하는 파일 건너뛰기')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                        help='Train 비율 (기본: 0.7)')
    parser.add_argument('--val-ratio', type=float, default=0.15,
                        help='Validation 비율 (기본: 0.15)')

    args = parser.parse_args()

    print("=" * 60)
    print("CT 이미지 전처리 (battery_outline 기반 crop)")
    print("=" * 60)
    print(f"이미지 크기: {args.size}x{args.size}")
    print(f"출력 디렉토리: {args.output}")
    print(f"라벨 경로: {LABEL_BASE}")
    print("=" * 60)

    # 원본 이미지 폴더에서 모든 이미지 수집
    print("\n이미지 경로 수집 중...")
    all_images = collect_all_images()
    print(f"총 {len(all_images)}개 이미지 발견")

    # 처리할 작업 목록 생성
    tasks = []
    for img_path, split_type in all_images:
        filename = os.path.basename(img_path)
        base_name = os.path.splitext(filename)[0]
        dst_path = os.path.join(args.output, split_type, base_name + '.jpg')

        # 이미 존재하면 건너뛰기
        if args.skip_existing and os.path.exists(dst_path):
            continue

        tasks.append((img_path, dst_path, split_type))

    print(f"처리할 이미지: {len(tasks)}개")

    if not tasks:
        print("처리할 이미지가 없습니다.")
        return

    # 병렬 처리 및 라벨 수집
    print(f"\n전처리 시작 (workers: {args.workers})...")
    success_count = 0
    fail_count = 0
    no_label_count = 0
    errors = []
    results = []  # (dst_path, label, split_type, battery_id)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(preprocess_image, src, dst, args.size): (src, dst, split_type)
            for src, dst, split_type in tasks
        }

        with tqdm(total=len(tasks), desc="Processing") as pbar:
            for future in as_completed(futures):
                src, dst, split_type = futures[future]
                success, msg, label = future.result()
                if success:
                    success_count += 1
                    if label is not None:
                        battery_id = extract_battery_id(os.path.basename(src))
                        results.append((dst, label, split_type, battery_id))
                    else:
                        no_label_count += 1
                else:
                    fail_count += 1
                    errors.append((src, msg))
                pbar.update(1)

    print(f"\n완료: {success_count}개 성공, {fail_count}개 실패, {no_label_count}개 라벨 없음")

    if errors and len(errors) <= 10:
        print("\n실패 목록:")
        for src, msg in errors:
            print(f"  {os.path.basename(src)}: {msg}")

    # 배터리 단위로 Train/Val/Test 분할 (Stratified)
    print("\n배터리 단위 Stratified Split 생성 중...")

    # 배터리 ID별로 그룹화 및 주요 클래스 결정
    from collections import defaultdict, Counter
    import random

    battery_groups = defaultdict(list)
    battery_main_class = {}  # 배터리별 주요 클래스

    for dst, label, split_type, battery_id in results:
        if battery_id is not None:
            battery_groups[battery_id].append((dst, label))

    # 각 배터리의 주요 클래스 결정 (가장 많은 클래스)
    for battery_id, items in battery_groups.items():
        labels = [label for _, label in items]
        main_class = Counter(labels).most_common(1)[0][0]
        battery_main_class[battery_id] = main_class

    # 클래스별로 배터리 ID 그룹화
    class_battery_ids = defaultdict(list)
    for battery_id, main_class in battery_main_class.items():
        class_battery_ids[main_class].append(battery_id)

    # 클래스별 Stratified 분할
    random.seed(42)
    train_ids = set()
    val_ids = set()
    test_ids = set()

    for class_id, ids in class_battery_ids.items():
        random.shuffle(ids)
        n = len(ids)
        n_train = max(1, int(n * args.train_ratio))
        n_val = max(1, int(n * args.val_ratio)) if n > 2 else 0
        # 최소 1개는 test에 배정 (가능한 경우)
        if n > 2:
            n_test = n - n_train - n_val
            if n_test < 1:
                n_train -= 1
                n_test = 1
        else:
            n_test = 0

        train_ids.update(ids[:n_train])
        val_ids.update(ids[n_train:n_train + n_val])
        test_ids.update(ids[n_train + n_val:])

        print(f"  Class {class_id}: {n}개 배터리 → Train {n_train}, Val {n_val}, Test {n - n_train - n_val}")

    print(f"  배터리 수: Train {len(train_ids)}, Val {len(val_ids)}, Test {len(test_ids)}")

    # Split 파일 생성
    new_splits_dir = os.path.join(SPLITS_PATH, 'cropped')
    os.makedirs(new_splits_dir, exist_ok=True)

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

    # 클래스별 통계
    from collections import Counter
    class_names = ['cell_normal', 'cell_porosity', 'module_normal', 'module_porosity', 'module_resin']

    print("\n클래스별 분포:")
    for name, items in [('Train', train_items), ('Val', val_items), ('Test', test_items)]:
        counts = Counter([label for _, label in items])
        print(f"  {name}: {dict(counts)}")

    # 파일 저장
    for filename, items in [
        ('battery_train.txt', train_items),
        ('battery_val.txt', val_items),
        ('battery_test.txt', test_items)
    ]:
        filepath = os.path.join(new_splits_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            for dst, label in items:
                f.write(f"{dst}\t{label}\n")
        print(f"Created: {filepath} ({len(items)}개)")

    # 용량 확인
    print("\n저장된 데이터 용량 확인...")
    os.system(f"du -sh {args.output}")

    print("\n전처리 완료!")
    print(f"새 Split 파일 위치: {new_splits_dir}")


if __name__ == "__main__":
    main()
