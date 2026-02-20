"""
원본 이미지에서 직접 결함 영역 Crop
- 원본 4000x4000 이미지에서 defect bbox로 직접 crop
- 좌표 스케일링 문제 없음
- Normal 이미지는 배터리 내부에서 랜덤 crop (결함 이미지와 동일한 스타일)
"""

import os
import json
import argparse
from pathlib import Path
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from typing import List, Tuple, Optional
from collections import defaultdict, Counter
import random
import re


# 설정
DEFAULT_IMAGE_SIZE = 512
DEFAULT_OUTPUT_DIR = "/mnt/d/battery-defect-direct"

# 데이터 경로 (원본 4000x4000)
IMAGE_BASE = "/home/ubuntu/battery-data/103.배터리 불량 이미지 데이터/3.개방데이터/1.데이터"
LABEL_BASE = "/mnt/d/103.배터리 불량 이미지 데이터/3.개방데이터/1.데이터"

# 이미지/라벨 폴더 매핑
IMAGE_DIRS = {
    'Training': [
        'Training/01.원천데이터/TS_CT_Datasets_images_1',
        'Training/01.원천데이터/TS_CT_Datasets_images_2',
    ],
    'Validation': [
        'Validation/01.원천데이터/VS_CT_Datasets_images_1',
        'Validation/01.원천데이터/VS_CT_Datasets_images_2',
    ]
}

LABEL_DIRS = {
    'Training': 'Training/02.라벨링데이터/TL_CT_Datasets_label',
    'Validation': 'Validation/02.라벨링데이터/VL_CT_Datasets_label'
}

SPLITS_PATH = "/home/ubuntu/projects/battery-inspection/training/data/splits/ct"


def polygon_to_bbox(points: List[float], padding: int = 50) -> Tuple[int, int, int, int]:
    """폴리곤 좌표를 bounding box로 변환"""
    x_coords = points[0::2]
    y_coords = points[1::2]

    x_min = int(min(x_coords)) - padding
    y_min = int(min(y_coords)) - padding
    x_max = int(max(x_coords)) + padding
    y_max = int(max(y_coords)) + padding

    return x_min, y_min, x_max, y_max


def make_square_bbox(bbox: Tuple[int, int, int, int], img_size: Tuple[int, int]) -> Tuple[int, int, int, int]:
    """bbox를 정사각형으로 확장 (이미지 범위 내)"""
    x_min, y_min, x_max, y_max = bbox
    img_w, img_h = img_size

    width = x_max - x_min
    height = y_max - y_min

    # 더 큰 쪽에 맞춤
    size = max(width, height)

    # 중심 기준 확장
    center_x = (x_min + x_max) // 2
    center_y = (y_min + y_max) // 2

    x_min = center_x - size // 2
    x_max = center_x + size // 2
    y_min = center_y - size // 2
    y_max = center_y + size // 2

    # 이미지 범위 내로 조정
    if x_min < 0:
        x_max -= x_min
        x_min = 0
    if y_min < 0:
        y_max -= y_min
        y_min = 0
    if x_max > img_w:
        x_min -= (x_max - img_w)
        x_max = img_w
    if y_max > img_h:
        y_min -= (y_max - img_h)
        y_max = img_h

    # 최종 범위 확인
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(img_w, x_max)
    y_max = min(img_h, y_max)

    return x_min, y_min, x_max, y_max


def random_crop_in_outline(
    outline_points: List[float],
    img_size: Tuple[int, int],
    crop_size: int,
    defect_padding: int
) -> Tuple[int, int, int, int]:
    """
    배터리 outline 내부에서 결함과 비슷한 형태의 랜덤 영역 선택

    결함 bbox 특성:
    - 매우 가늘고 긴 형태 (width ~50px, height ~600-1000px)
    - 정사각형화하면 배터리 바깥 검은 영역 포함

    정상 이미지도 동일한 스타일로 크롭:
    - 가늘고 긴 영역 선택
    - 정사각형화하여 검은 패딩 포함

    Args:
        outline_points: battery_outline 폴리곤 좌표
        img_size: 원본 이미지 크기
        crop_size: 목표 crop 크기
        defect_padding: padding 크기

    Returns:
        (x_min, y_min, x_max, y_max) crop 영역
    """
    img_w, img_h = img_size

    # outline bbox 계산
    outline_bbox = polygon_to_bbox(outline_points, padding=0)
    ox_min, oy_min, ox_max, oy_max = outline_bbox

    # 범위 보정
    ox_min = max(0, ox_min)
    oy_min = max(0, oy_min)
    ox_max = min(img_w, ox_max)
    oy_max = min(img_h, oy_max)

    outline_w = ox_max - ox_min
    outline_h = oy_max - oy_min

    # 결함 bbox와 동일한 분포로 영역 생성
    # 실제 결함 통계: width 5~33px, height 43~3256px
    thin_width = random.randint(5, 33)  # 결함 실제 분포 (10%ile=5, 90%ile=33)
    thin_height = random.randint(43, 3256)  # 결함 실제 분포 (10%ile=43, 90%ile=3256)

    # outline 크기에 맞게 조정
    thin_height = min(thin_height, outline_h - 100)
    thin_height = max(thin_height, 43)  # 최소 높이 (결함 분포와 동일)

    # 랜덤 위치 선택 (outline 내부, 중앙 근처)
    # x: outline 중앙 근처에서 약간 랜덤
    center_x = (ox_min + ox_max) // 2
    x_jitter = random.randint(-outline_w // 4, outline_w // 4)
    x = center_x + x_jitter - thin_width // 2

    # y: outline 내부에서 랜덤
    y_margin = 50
    y_min_pos = oy_min + y_margin
    y_max_pos = oy_max - thin_height - y_margin

    if y_max_pos <= y_min_pos:
        y = oy_min
    else:
        y = random.randint(y_min_pos, y_max_pos)

    # 가늘고 긴 bbox 생성
    thin_bbox = (x, y, x + thin_width, y + thin_height)

    # padding 추가 후 정사각형화 (결함 이미지와 동일한 처리)
    padded_bbox = (
        thin_bbox[0] - defect_padding,
        thin_bbox[1] - defect_padding,
        thin_bbox[2] + defect_padding,
        thin_bbox[3] + defect_padding
    )

    # 정사각형화 - 이때 배터리 바깥 검은 영역 포함됨
    return make_square_bbox(padded_bbox, img_size)


def get_label_path(image_name: str, split_type: str) -> Optional[str]:
    """이미지 이름에서 라벨 JSON 경로 생성"""
    base_name = os.path.splitext(image_name)[0]
    json_name = base_name + '.json'

    label_dir = os.path.join(LABEL_BASE, LABEL_DIRS[split_type])
    label_path = os.path.join(label_dir, json_name)

    return label_path if os.path.exists(label_path) else None


def load_label_info(label_path: str) -> Optional[dict]:
    """라벨 JSON에서 모든 정보 로드"""
    try:
        with open(label_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # battery_outline
        outline = data.get('swelling', {}).get('battery_outline')
        if outline and len(outline) < 6:
            outline = None

        # defects
        defects = data.get('defects')
        defect_points = None
        defect_name = None

        if defects and len(defects) > 0:
            # 모든 결함 영역 합치기
            all_points = []
            for defect in defects:
                points = defect.get('points', [])
                if points and len(points) >= 4:
                    all_points.extend(points)

            if all_points:
                defect_points = all_points
                defect_name = defects[0].get('name', '').lower()

        # 클래스 정보
        data_info = data.get('data_info', {})
        image_info = data.get('image_info', {})

        battery_type = data_info.get('type', '')
        is_normal = image_info.get('is_normal', True)

        # 클래스 라벨 결정
        if battery_type == 'cell':
            if is_normal or defect_name is None:
                label = 0  # cell_normal
            elif 'porosity' in defect_name:
                label = 1  # cell_porosity
            else:
                label = 0
        else:  # module
            if is_normal or defect_name is None:
                label = 2  # module_normal
            elif 'porosity' in defect_name:
                label = 3  # module_porosity
            elif 'resin' in defect_name:
                label = 4  # module_resin
            else:
                label = 2

        return {
            'outline': outline,
            'defect_points': defect_points,
            'defect_name': defect_name,
            'label': label,
            'is_normal': is_normal,
            'battery_type': battery_type
        }
    except Exception as e:
        return None


def process_image(
    src_path: str,
    dst_path: str,
    label_info: Optional[dict],
    image_size: int,
    defect_padding: int,
    normal_mode: str
) -> Tuple[bool, str, Optional[int]]:
    """
    원본 이미지에서 직접 crop

    Args:
        src_path: 원본 4000x4000 이미지 경로
        dst_path: 저장 경로
        label_info: 라벨 정보
        image_size: 출력 이미지 크기
        defect_padding: 결함 영역 주변 패딩
        normal_mode: normal 이미지 처리 방식 ('outline', 'center', 'skip')
    """
    try:
        label = label_info['label'] if label_info else None

        # 이미지 로드
        img = Image.open(src_path)
        orig_w, orig_h = img.size

        if label_info and label_info['defect_points']:
            # 결함 이미지: defect bbox로 crop
            bbox = polygon_to_bbox(label_info['defect_points'], padding=defect_padding)
            bbox = make_square_bbox(bbox, (orig_w, orig_h))

            x_min, y_min, x_max, y_max = bbox
            if x_max > x_min and y_max > y_min:
                img = img.crop((x_min, y_min, x_max, y_max))

        elif label_info and label_info['outline']:
            # Normal 이미지: battery_outline 처리
            if normal_mode == 'skip':
                return False, "Normal skipped", label

            if normal_mode == 'random':
                # 배터리 내부에서 랜덤 영역 crop (결함 이미지와 비슷한 스타일)
                bbox = random_crop_in_outline(
                    label_info['outline'],
                    (orig_w, orig_h),
                    image_size,
                    defect_padding
                )
            else:
                # outline 전체 사용
                bbox = polygon_to_bbox(label_info['outline'], padding=20)

            x_min, y_min, x_max, y_max = bbox

            # 이미지 범위 내로 조정
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(orig_w, x_max)
            y_max = min(orig_h, y_max)

            if x_max > x_min and y_max > y_min:
                img = img.crop((x_min, y_min, x_max, y_max))
        else:
            # 라벨 없음
            if normal_mode == 'skip':
                return False, "No label, skipped", label

        # RGB 변환
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # 리사이즈 (비율 유지)
        img.thumbnail((image_size, image_size), Image.LANCZOS)

        # 정사각형 캔버스에 중앙 배치
        canvas = Image.new('RGB', (image_size, image_size), (0, 0, 0))
        x_offset = (image_size - img.size[0]) // 2
        y_offset = (image_size - img.size[1]) // 2
        canvas.paste(img, (x_offset, y_offset))

        # 저장
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        canvas.save(dst_path, 'JPEG', quality=95)

        return True, "OK", label
    except Exception as e:
        return False, str(e), None


def extract_battery_id(filename: str) -> Optional[int]:
    """파일명에서 배터리 ID 추출"""
    match = re.search(r'_(\d{3})_[xyz]_', filename)
    if match:
        return int(match.group(1))
    return None


def collect_all_images() -> List[Tuple[str, str]]:
    """원본 이미지 폴더에서 모든 CT 이미지 경로 수집"""
    images = []

    for split_type, dirs in IMAGE_DIRS.items():
        for dir_rel in dirs:
            dir_path = os.path.join(IMAGE_BASE, dir_rel)
            if os.path.exists(dir_path):
                for f in os.listdir(dir_path):
                    if f.endswith(('.jpg', '.png', '.jpeg')):
                        images.append((os.path.join(dir_path, f), split_type))

    return images


def main():
    parser = argparse.ArgumentParser(description='원본에서 직접 결함 영역 Crop')
    parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f'출력 디렉토리 (기본: {DEFAULT_OUTPUT_DIR})')
    parser.add_argument('--size', type=int, default=DEFAULT_IMAGE_SIZE,
                        help=f'출력 이미지 크기 (기본: {DEFAULT_IMAGE_SIZE})')
    parser.add_argument('--defect-padding', type=int, default=200,
                        help='결함 영역 주변 패딩 픽셀 (기본: 200)')
    parser.add_argument('--normal-mode', type=str, default='random',
                        choices=['random', 'outline', 'center', 'skip'],
                        help='Normal 이미지 처리: random(배터리 내부 랜덤crop), outline(전체), center(중앙), skip(제외)')
    parser.add_argument('--workers', type=int, default=8,
                        help='병렬 처리 워커 수 (기본: 8)')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                        help='Train 비율 (기본: 0.7)')
    parser.add_argument('--val-ratio', type=float, default=0.15,
                        help='Validation 비율 (기본: 0.15)')

    args = parser.parse_args()

    print("=" * 60)
    print("원본에서 직접 결함 영역 Crop")
    print("=" * 60)
    print(f"원본 이미지: {IMAGE_BASE}")
    print(f"출력 디렉토리: {args.output}")
    print(f"이미지 크기: {args.size}x{args.size}")
    print(f"결함 패딩: {args.defect_padding}px")
    print(f"Normal 모드: {args.normal_mode}")
    print("=" * 60)

    # 원본 이미지 수집
    print("\n원본 이미지 수집 중...")
    all_images = collect_all_images()
    print(f"총 {len(all_images)}개 이미지 발견")

    if not all_images:
        print("처리할 이미지가 없습니다.")
        return

    # 이미지 처리
    print(f"\n처리 시작 (workers: {args.workers})...")
    results = []
    success_count = 0
    fail_count = 0
    skip_count = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {}

        print("작업 제출 중...")
        for i, (src_path, split_type) in enumerate(all_images):
            filename = os.path.basename(src_path)
            base_name = os.path.splitext(filename)[0]
            dst_path = os.path.join(args.output, split_type, base_name + '.jpg')

            label_path = get_label_path(filename, split_type)
            label_info = load_label_info(label_path) if label_path else None

            future = executor.submit(
                process_image,
                src_path, dst_path, label_info,
                args.size, args.defect_padding, args.normal_mode
            )
            futures[future] = (src_path, dst_path, split_type, label_info)

            # 진행 상황 표시 (10000개마다)
            if (i + 1) % 10000 == 0:
                print(f"  {i + 1}/{len(all_images)} 제출됨...")

        print(f"전체 {len(futures)}개 작업 제출 완료, 처리 중...\n")

        with tqdm(total=len(futures), desc="Processing", mininterval=0.5) as pbar:
            for future in as_completed(futures):
                src_path, dst_path, split_type, label_info = futures[future]
                success, msg, label = future.result()

                if success:
                    success_count += 1
                    battery_id = extract_battery_id(os.path.basename(src_path))
                    if label is not None:
                        results.append((dst_path, label, split_type, battery_id))
                elif "skipped" in msg.lower():
                    skip_count += 1
                else:
                    fail_count += 1

                pbar.update(1)

    print(f"\n완료: {success_count}개 성공, {skip_count}개 스킵, {fail_count}개 실패")

    # 클래스별 통계
    label_counts = Counter([label for _, label, _, _ in results])
    class_names = ['cell_normal', 'cell_porosity', 'module_normal', 'module_porosity', 'module_resin']
    print("\n클래스별 이미지 수:")
    for i, name in enumerate(class_names):
        print(f"  {i}: {name} = {label_counts.get(i, 0)}개")

    # Stratified Split 생성
    print("\n배터리 단위 Stratified Split 생성 중...")

    battery_groups = defaultdict(list)
    battery_main_class = {}

    for dst, label, split_type, battery_id in results:
        if battery_id is not None and label is not None:
            battery_groups[battery_id].append((dst, label))

    # 각 배터리의 주요 클래스 결정
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

    for class_id, ids in sorted(class_battery_ids.items()):
        random.shuffle(ids)
        n = len(ids)
        n_train = max(1, int(n * args.train_ratio))
        n_val = max(1, int(n * args.val_ratio)) if n > 2 else 0

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

        print(f"  Class {class_id} ({class_names[class_id]}): {n}개 배터리 → Train {n_train}, Val {n_val}, Test {n - n_train - n_val}")

    # Split 파일 생성
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

    # 클래스별 분포 출력
    print("\n최종 Split 클래스별 분포:")
    for name, items in [('Train', train_items), ('Val', val_items), ('Test', test_items)]:
        counts = Counter([label for _, label in items])
        print(f"  {name}: {dict(counts)}")

    # Split 파일 저장
    new_splits_dir = os.path.join(SPLITS_PATH, 'defect_random')
    os.makedirs(new_splits_dir, exist_ok=True)

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
    print("\n저장된 데이터 용량:")
    os.system(f"du -sh {args.output}")

    print("\n전처리 완료!")
    print(f"Split 파일 위치: {new_splits_dir}")


if __name__ == "__main__":
    main()
