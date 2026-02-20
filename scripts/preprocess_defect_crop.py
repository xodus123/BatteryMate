"""
2차 전처리: 결함 영역 Crop
- 1차 전처리(battery_outline crop) 후 실행
- JSON 라벨의 defects[].points로 결함 영역만 crop
- Normal 이미지는 배터리 내부에서 랜덤 crop (스타일 통일)
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


# 설정
DEFAULT_IMAGE_SIZE = 512  # 결함 영역은 더 작게
DEFAULT_INPUT_DIR = "/mnt/d/battery-cropped"  # 1차 crop 결과
DEFAULT_OUTPUT_DIR = "/mnt/d/battery-defect-crop"

# 라벨 경로
LABEL_BASE = "/mnt/d/103.배터리 불량 이미지 데이터/3.개방데이터/1.데이터"
LABEL_DIRS = {
    'Training': 'Training/02.라벨링데이터/TL_CT_Datasets_label',
    'Validation': 'Validation/02.라벨링데이터/VL_CT_Datasets_label'
}

SPLITS_PATH = "/home/ubuntu/projects/battery-inspection/training/data/splits/ct"


def polygon_to_bbox(points: List[float], padding: int = 50) -> Tuple[int, int, int, int]:
    """폴리곤 좌표를 bounding box로 변환"""
    x_coords = points[0::2]
    y_coords = points[1::2]

    x_min = max(0, int(min(x_coords)) - padding)
    y_min = max(0, int(min(y_coords)) - padding)
    x_max = int(max(x_coords)) + padding
    y_max = int(max(y_coords)) + padding

    return x_min, y_min, x_max, y_max


def load_battery_outline(label_path: str) -> Optional[Tuple[int, int, int, int]]:
    """라벨 JSON에서 battery_outline bbox 추출"""
    try:
        with open(label_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        battery_outline = data.get('battery_outline', {})
        points = battery_outline.get('points', [])

        if points and len(points) >= 6:
            return polygon_to_bbox(points, padding=0)
        return None
    except:
        return None


def random_crop_in_bbox(
    img: Image.Image,
    bbox: Tuple[int, int, int, int],
    crop_size: int,
    orig_size: Tuple[int, int],
    cropped_size: Tuple[int, int]
) -> Image.Image:
    """
    배터리 내부 bbox 영역에서 랜덤 crop

    Args:
        img: 1차 전처리된 이미지 (battery_outline crop된 상태)
        bbox: 원본 이미지에서의 battery_outline bbox (원본 좌표계)
        crop_size: 최종 crop 크기 (정사각형)
        orig_size: 원본 이미지 크기 (4000x4000)
        cropped_size: 1차 전처리 후 이미지 크기 (1024x1024)

    Returns:
        랜덤 crop된 이미지
    """
    img_w, img_h = img.size

    # bbox를 1차 전처리된 이미지 좌표계로 변환
    # 원본에서 bbox 영역만 crop 후 cropped_size로 resize되었음
    orig_x_min, orig_y_min, orig_x_max, orig_y_max = bbox
    orig_bbox_w = orig_x_max - orig_x_min
    orig_bbox_h = orig_y_max - orig_y_min

    # 1차 전처리에서는 bbox 영역을 crop 후 resize했으므로
    # 현재 이미지 전체가 배터리 영역임
    # 따라서 이미지 내부에서 랜덤 crop하면 됨

    # crop 가능한 영역 계산 (이미지 내부에서)
    # 최소 crop_size 만큼의 영역이 필요
    margin = 50  # 가장자리 제외

    max_x = max(0, img_w - crop_size - margin)
    max_y = max(0, img_h - crop_size - margin)

    if max_x <= margin or max_y <= margin:
        # 이미지가 너무 작으면 중앙 crop
        left = (img_w - crop_size) // 2
        top = (img_h - crop_size) // 2
    else:
        left = random.randint(margin, max_x)
        top = random.randint(margin, max_y)

    # 범위 보정
    left = max(0, min(left, img_w - crop_size))
    top = max(0, min(top, img_h - crop_size))
    right = min(left + crop_size, img_w)
    bottom = min(top + crop_size, img_h)

    return img.crop((left, top, right, bottom))


def get_label_path(image_name: str, split_type: str) -> Optional[str]:
    """이미지 이름에서 라벨 JSON 경로 생성"""
    base_name = os.path.splitext(image_name)[0]
    json_name = base_name + '.json'

    if split_type == 'Training':
        label_dir = os.path.join(LABEL_BASE, LABEL_DIRS['Training'])
    else:
        label_dir = os.path.join(LABEL_BASE, LABEL_DIRS['Validation'])

    label_path = os.path.join(label_dir, json_name)
    return label_path if os.path.exists(label_path) else None


def load_defect_info(label_path: str) -> Optional[dict]:
    """라벨 JSON에서 결함 정보 로드"""
    try:
        with open(label_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        defects = data.get('defects')
        if not defects or len(defects) == 0:
            return None

        # 모든 결함 영역의 bbox 계산 (여러 결함이 있을 수 있음)
        all_points = []
        defect_names = []
        for defect in defects:
            points = defect.get('points', [])
            if points and len(points) >= 6:
                all_points.extend(points)
                defect_names.append(defect.get('name', 'unknown'))

        if not all_points:
            return None

        return {
            'points': all_points,
            'defect_names': defect_names,
            'num_defects': len(defects)
        }
    except Exception as e:
        return None


def get_class_from_label(label_path: str) -> Optional[int]:
    """라벨 JSON에서 클래스 정보 추출"""
    try:
        with open(label_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        data_info = data.get('data_info', {})
        image_info = data.get('image_info', {})
        defects = data.get('defects')

        battery_type = data_info.get('type', '')
        is_normal = image_info.get('is_normal', True)

        defect_type = None
        if defects and len(defects) > 0:
            defect_type = defects[0].get('name', '').lower()

        # 클래스 결정
        if battery_type == 'cell':
            if is_normal or defect_type is None:
                return 0  # cell_normal
            elif 'porosity' in defect_type:
                return 1  # cell_porosity
            else:
                return 0
        else:  # module
            if is_normal or defect_type is None:
                return 2  # module_normal
            elif 'porosity' in defect_type:
                return 3  # module_porosity
            elif 'resin' in defect_type:
                return 4  # module_resin
            else:
                return 2
    except:
        return None


def process_image(
    src_path: str,
    dst_path: str,
    label_path: Optional[str],
    image_size: int,
    defect_padding: int,
    normal_mode: str
) -> Tuple[bool, str, Optional[int]]:
    """
    이미지 처리 (결함 영역 또는 중앙 crop)

    Args:
        src_path: 1차 crop된 이미지 경로
        dst_path: 저장 경로
        label_path: 라벨 JSON 경로
        image_size: 출력 이미지 크기
        defect_padding: 결함 영역 주변 패딩
        normal_mode: normal 이미지 처리 방식 ('center', 'full', 'skip')

    Returns:
        (성공여부, 메시지, 라벨)
    """
    try:
        # 라벨 정보 로드
        label = None
        defect_info = None

        if label_path:
            label = get_class_from_label(label_path)
            defect_info = load_defect_info(label_path)

        # 이미지 로드
        img = Image.open(src_path)
        orig_w, orig_h = img.size

        if defect_info:
            # 결함 이미지: 결함 영역 crop
            bbox = polygon_to_bbox(defect_info['points'], padding=defect_padding)
            x_min, y_min, x_max, y_max = bbox

            # 이미지 범위 내로 조정
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(orig_w, x_max)
            y_max = min(orig_h, y_max)

            if x_max > x_min and y_max > y_min:
                img = img.crop((x_min, y_min, x_max, y_max))
        else:
            # Normal 이미지
            if normal_mode == 'skip':
                return False, "Normal skipped", label
            elif normal_mode == 'random':
                # 배터리 내부에서 랜덤 crop (결함 이미지와 동일한 스타일)
                # 1차 전처리된 이미지 전체가 배터리 영역이므로 내부에서 랜덤 crop
                battery_bbox = None
                if label_path:
                    battery_bbox = load_battery_outline(label_path)

                # 결함 이미지와 비슷한 크기로 crop (image_size보다 약간 큰 영역)
                crop_target = int(image_size * 1.2)  # 약간 여유 있게

                if orig_w >= crop_target and orig_h >= crop_target:
                    img = random_crop_in_bbox(
                        img,
                        battery_bbox or (0, 0, orig_w, orig_h),
                        crop_target,
                        (4000, 4000),
                        (orig_w, orig_h)
                    )
                else:
                    # 이미지가 작으면 그대로 사용
                    pass
            elif normal_mode == 'center':
                # 중앙 정사각형 crop
                min_dim = min(orig_w, orig_h)
                left = (orig_w - min_dim) // 2
                top = (orig_h - min_dim) // 2
                img = img.crop((left, top, left + min_dim, top + min_dim))
            # 'full'이면 그대로 유지

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
    import re
    match = re.search(r'_(\d{3})_[xyz]_', filename)
    if match:
        return int(match.group(1))
    return None


def main():
    parser = argparse.ArgumentParser(description='2차 전처리: 결함 영역 Crop')
    parser.add_argument('--input', type=str, default=DEFAULT_INPUT_DIR,
                        help=f'1차 crop 이미지 디렉토리 (기본: {DEFAULT_INPUT_DIR})')
    parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f'출력 디렉토리 (기본: {DEFAULT_OUTPUT_DIR})')
    parser.add_argument('--size', type=int, default=DEFAULT_IMAGE_SIZE,
                        help=f'출력 이미지 크기 (기본: {DEFAULT_IMAGE_SIZE})')
    parser.add_argument('--defect-padding', type=int, default=100,
                        help='결함 영역 주변 패딩 픽셀 (기본: 100)')
    parser.add_argument('--normal-mode', type=str, default='random',
                        choices=['random', 'center', 'full', 'skip'],
                        help='Normal 이미지 처리: random(배터리 내부 랜덤crop), center(중앙crop), full(전체), skip(제외)')
    parser.add_argument('--workers', type=int, default=8,
                        help='병렬 처리 워커 수 (기본: 8)')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                        help='Train 비율 (기본: 0.7)')
    parser.add_argument('--val-ratio', type=float, default=0.15,
                        help='Validation 비율 (기본: 0.15)')

    args = parser.parse_args()

    print("=" * 60)
    print("2차 전처리: 결함 영역 Crop")
    print("=" * 60)
    print(f"입력 디렉토리: {args.input}")
    print(f"출력 디렉토리: {args.output}")
    print(f"이미지 크기: {args.size}x{args.size}")
    print(f"결함 패딩: {args.defect_padding}px")
    print(f"Normal 모드: {args.normal_mode}")
    print("=" * 60)

    # 1차 crop 이미지 수집
    print("\n1차 crop 이미지 수집 중...")
    images = []

    for split_type in ['Training', 'Validation']:
        input_dir = os.path.join(args.input, split_type)
        if not os.path.exists(input_dir):
            print(f"  경고: {input_dir} 없음")
            continue

        for f in os.listdir(input_dir):
            if f.endswith(('.jpg', '.png', '.jpeg')):
                images.append((os.path.join(input_dir, f), split_type))

    print(f"총 {len(images)}개 이미지 발견")

    if not images:
        print("처리할 이미지가 없습니다. 1차 전처리를 먼저 실행하세요.")
        return

    # 이미지 처리
    print(f"\n처리 시작 (workers: {args.workers})...")
    results = []
    success_count = 0
    fail_count = 0
    skip_count = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {}

        for src_path, split_type in images:
            filename = os.path.basename(src_path)
            base_name = os.path.splitext(filename)[0]
            dst_path = os.path.join(args.output, split_type, base_name + '.jpg')

            label_path = get_label_path(filename, split_type)

            future = executor.submit(
                process_image,
                src_path, dst_path, label_path,
                args.size, args.defect_padding, args.normal_mode
            )
            futures[future] = (src_path, dst_path, split_type)

        with tqdm(total=len(futures), desc="Processing") as pbar:
            for future in as_completed(futures):
                src_path, dst_path, split_type = futures[future]
                success, msg, label = future.result()

                if success:
                    success_count += 1
                    battery_id = extract_battery_id(os.path.basename(src_path))
                    results.append((dst_path, label, split_type, battery_id))
                elif msg == "Normal skipped":
                    skip_count += 1
                else:
                    fail_count += 1

                pbar.update(1)

    print(f"\n완료: {success_count}개 성공, {skip_count}개 스킵, {fail_count}개 실패")

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

        print(f"  Class {class_id}: {n}개 배터리 → Train {n_train}, Val {n_val}, Test {n - n_train - n_val}")

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

    # 클래스별 통계
    class_names = ['cell_normal', 'cell_porosity', 'module_normal', 'module_porosity', 'module_resin']

    print("\n클래스별 분포:")
    for name, items in [('Train', train_items), ('Val', val_items), ('Test', test_items)]:
        counts = Counter([label for _, label in items])
        print(f"  {name}: {dict(counts)}")

    # Split 파일 저장
    new_splits_dir = os.path.join(SPLITS_PATH, 'defect_crop')
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

    print("\n2차 전처리 완료!")
    print(f"Split 파일 위치: {new_splits_dir}")


if __name__ == "__main__":
    main()
