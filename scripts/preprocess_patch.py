"""
패치 기반 전처리 (Fixed-size Patching)

전략:
- 가로: 결함 중심 512px (좌우 배터리 영역 포함)
- 세로: 512px 단위로 분할 (긴 결함은 여러 패치)
- 정상: 배터리 영역에서 512x512 랜덤 패치 추출
- split 파일에 메타데이터 포함
"""

import os
import json
import argparse
from pathlib import Path
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from typing import List, Tuple, Optional, Dict
from collections import defaultdict, Counter
import random
import re


# 설정
PATCH_SIZE = 512
DEFAULT_OUTPUT_DIR = "/mnt/d/battery-patch"

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

# 클래스 이름
CLASS_NAMES = ['cell_normal', 'cell_porosity', 'module_normal', 'module_porosity', 'module_resin']


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
        defect_bboxes = []

        if defects and len(defects) > 0:
            for defect in defects:
                points = defect.get('points', [])
                if points and len(points) >= 4:
                    x_coords = points[0::2]
                    y_coords = points[1::2]
                    x_min, x_max = int(min(x_coords)), int(max(x_coords))
                    y_min, y_max = int(min(y_coords)), int(max(y_coords))
                    defect_bboxes.append((x_min, y_min, x_max, y_max))

        # 클래스 정보
        data_info = data.get('data_info', {})
        image_info = data.get('image_info', {})

        battery_type = data_info.get('type', '')  # 'cell' or 'module'
        is_normal = image_info.get('is_normal', True)

        # 결함 이름
        defect_name = None
        if defects and len(defects) > 0:
            defect_name = defects[0].get('name', '').lower()

        # 클래스 라벨 결정
        if battery_type == 'cell':
            if is_normal or not defect_bboxes:
                label = 0  # cell_normal
            elif defect_name and 'porosity' in defect_name:
                label = 1  # cell_porosity
            else:
                label = 0
        else:  # module
            if is_normal or not defect_bboxes:
                label = 2  # module_normal
            elif defect_name and 'porosity' in defect_name:
                label = 3  # module_porosity
            elif defect_name and 'resin' in defect_name:
                label = 4  # module_resin
            else:
                label = 2

        return {
            'outline': outline,
            'defect_bboxes': defect_bboxes,
            'label': label,
            'is_normal': is_normal,
            'battery_type': 1 if battery_type == 'module' else 0  # 0=cell, 1=module
        }
    except Exception as e:
        return None


def extract_axis_from_filename(filename: str) -> int:
    """파일명에서 axis 추출 (x=0, y=1, z=2)"""
    base_name = os.path.splitext(os.path.basename(filename))[0]
    parts = base_name.split('_')
    axis_map = {'x': 0, 'y': 1, 'z': 2}

    if len(parts) >= 2:
        axis_char = parts[-2].lower()
        if axis_char in axis_map:
            return axis_map[axis_char]
    return 0


def extract_battery_id(filename: str) -> Optional[int]:
    """파일명에서 배터리 ID 추출"""
    match = re.search(r'_(\d{3})_[xyz]_', filename)
    if match:
        return int(match.group(1))
    return None


def get_outline_bbox(outline_points: List[float]) -> Tuple[int, int, int, int]:
    """outline 폴리곤에서 bbox 추출"""
    x_coords = outline_points[0::2]
    y_coords = outline_points[1::2]
    return (
        int(min(x_coords)),
        int(min(y_coords)),
        int(max(x_coords)),
        int(max(y_coords))
    )


def create_defect_patches(
    img: Image.Image,
    defect_bboxes: List[Tuple[int, int, int, int]],
    outline_points: List[float],
    patch_size: int = 512,
    overlap: float = 0.25
) -> List[Image.Image]:
    """결함 영역에서 패치 생성

    가로: 결함 중심 기준 patch_size 폭
    세로: patch_size 단위로 분할 (overlap 적용)
    """
    patches = []
    img_w, img_h = img.size

    # outline bbox
    if outline_points:
        ox_min, oy_min, ox_max, oy_max = get_outline_bbox(outline_points)
    else:
        ox_min, oy_min, ox_max, oy_max = 0, 0, img_w, img_h

    for bbox in defect_bboxes:
        x_min, y_min, x_max, y_max = bbox
        defect_w = x_max - x_min
        defect_h = y_max - y_min

        # 가로: 결함 중심 기준 patch_size 폭
        center_x = (x_min + x_max) // 2
        patch_x_min = center_x - patch_size // 2
        patch_x_max = patch_x_min + patch_size

        # 경계 조정
        if patch_x_min < 0:
            patch_x_min = 0
            patch_x_max = patch_size
        if patch_x_max > img_w:
            patch_x_max = img_w
            patch_x_min = max(0, img_w - patch_size)

        # 세로: patch_size 단위로 분할
        step = int(patch_size * (1 - overlap))

        if defect_h <= patch_size:
            # 결함이 patch_size 이하면 중앙 정렬
            center_y = (y_min + y_max) // 2
            patch_y_min = center_y - patch_size // 2
            patch_y_max = patch_y_min + patch_size

            # 경계 조정
            if patch_y_min < 0:
                patch_y_min = 0
                patch_y_max = patch_size
            if patch_y_max > img_h:
                patch_y_max = img_h
                patch_y_min = max(0, img_h - patch_size)

            # 패치 추출
            patch = extract_patch(img, patch_x_min, patch_y_min, patch_size)
            if patch:
                patches.append(patch)
        else:
            # 결함이 patch_size 초과면 분할
            y_start = max(0, y_min - patch_size // 4)  # 결함 시작 약간 전부터
            y_end = min(img_h, y_max + patch_size // 4)  # 결함 끝 약간 후까지

            y = y_start
            while y + patch_size <= img_h and y < y_end:
                patch = extract_patch(img, patch_x_min, y, patch_size)
                if patch:
                    patches.append(patch)
                y += step

            # 마지막 패치 (끝부분)
            if y < y_end:
                final_y = min(y_end, img_h) - patch_size
                if final_y >= 0 and final_y != y - step:
                    patch = extract_patch(img, patch_x_min, final_y, patch_size)
                    if patch:
                        patches.append(patch)

    return patches


def create_normal_patches(
    img: Image.Image,
    outline_points: List[float],
    patch_size: int = 512,
    num_patches: int = 3
) -> List[Image.Image]:
    """정상 이미지에서 랜덤 패치 생성"""
    patches = []
    img_w, img_h = img.size

    if not outline_points:
        return patches

    # outline bbox
    ox_min, oy_min, ox_max, oy_max = get_outline_bbox(outline_points)

    # 유효 영역 계산
    valid_w = ox_max - ox_min - patch_size
    valid_h = oy_max - oy_min - patch_size

    if valid_w <= 0 or valid_h <= 0:
        # outline이 patch_size보다 작으면 중앙 패치 하나만
        center_x = (ox_min + ox_max) // 2 - patch_size // 2
        center_y = (oy_min + oy_max) // 2 - patch_size // 2
        center_x = max(0, min(center_x, img_w - patch_size))
        center_y = max(0, min(center_y, img_h - patch_size))
        patch = extract_patch(img, center_x, center_y, patch_size)
        if patch:
            patches.append(patch)
        return patches

    # 랜덤 패치 생성
    for _ in range(num_patches):
        x = random.randint(ox_min, ox_min + valid_w)
        y = random.randint(oy_min, oy_min + valid_h)

        # 경계 조정
        x = max(0, min(x, img_w - patch_size))
        y = max(0, min(y, img_h - patch_size))

        patch = extract_patch(img, x, y, patch_size)
        if patch:
            patches.append(patch)

    return patches


def extract_patch(img: Image.Image, x: int, y: int, size: int) -> Optional[Image.Image]:
    """이미지에서 패치 추출"""
    try:
        img_w, img_h = img.size

        # 경계 조정
        x = max(0, min(x, img_w - size))
        y = max(0, min(y, img_h - size))

        patch = img.crop((x, y, x + size, y + size))

        # RGB 변환
        if patch.mode != 'RGB':
            patch = patch.convert('RGB')

        return patch
    except Exception:
        return None


def process_image(
    src_path: str,
    dst_dir: str,
    label_info: Optional[dict],
    patch_size: int,
    num_normal_patches: int
) -> List[Tuple[str, int, int, int]]:
    """이미지 처리 및 패치 생성

    Returns:
        List of (dst_path, label, battery_type, axis)
    """
    results = []

    try:
        filename = os.path.basename(src_path)
        base_name = os.path.splitext(filename)[0]

        if not label_info:
            return results

        # 메타데이터
        label = label_info['label']
        battery_type = label_info['battery_type']
        axis = extract_axis_from_filename(filename)

        # 이미지 로드
        img = Image.open(src_path)

        # 패치 생성
        if label_info['defect_bboxes']:
            # 결함 이미지
            patches = create_defect_patches(
                img,
                label_info['defect_bboxes'],
                label_info['outline'],
                patch_size
            )
        elif label_info['outline']:
            # 정상 이미지
            patches = create_normal_patches(
                img,
                label_info['outline'],
                patch_size,
                num_normal_patches
            )
        else:
            return results

        # 패치 저장
        for i, patch in enumerate(patches):
            patch_name = f"{base_name}_p{i:02d}.jpg"
            dst_path = os.path.join(dst_dir, patch_name)

            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            patch.save(dst_path, 'JPEG', quality=95)

            results.append((dst_path, label, battery_type, axis))

    except Exception as e:
        pass

    return results


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
    parser = argparse.ArgumentParser(description='패치 기반 전처리')
    parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f'출력 디렉토리 (기본: {DEFAULT_OUTPUT_DIR})')
    parser.add_argument('--patch-size', type=int, default=PATCH_SIZE,
                        help=f'패치 크기 (기본: {PATCH_SIZE})')
    parser.add_argument('--num-normal-patches', type=int, default=3,
                        help='정상 이미지당 패치 수 (기본: 3)')
    parser.add_argument('--workers', type=int, default=8,
                        help='병렬 처리 워커 수 (기본: 8)')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                        help='Train 비율 (기본: 0.7)')
    parser.add_argument('--val-ratio', type=float, default=0.15,
                        help='Validation 비율 (기본: 0.15)')

    args = parser.parse_args()

    print("=" * 60)
    print("패치 기반 전처리 (Fixed-size Patching)")
    print("=" * 60)
    print(f"원본 이미지: {IMAGE_BASE}")
    print(f"출력 디렉토리: {args.output}")
    print(f"패치 크기: {args.patch_size}x{args.patch_size}")
    print(f"정상 패치 수: {args.num_normal_patches}")
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
    all_results = []  # (dst_path, label, battery_type, axis, battery_id)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {}

        print("작업 제출 중...")
        for i, (src_path, split_type) in enumerate(all_images):
            filename = os.path.basename(src_path)
            dst_dir = os.path.join(args.output, split_type)

            label_path = get_label_path(filename, split_type)
            label_info = load_label_info(label_path) if label_path else None

            future = executor.submit(
                process_image,
                src_path, dst_dir, label_info,
                args.patch_size, args.num_normal_patches
            )
            futures[future] = (src_path, split_type)

            if (i + 1) % 10000 == 0:
                print(f"  {i + 1}/{len(all_images)} 제출됨...")

        print(f"전체 {len(futures)}개 작업 제출 완료, 처리 중...\n")

        with tqdm(total=len(futures), desc="Processing", mininterval=0.5) as pbar:
            for future in as_completed(futures):
                src_path, split_type = futures[future]
                results = future.result()

                battery_id = extract_battery_id(os.path.basename(src_path))

                for dst_path, label, battery_type, axis in results:
                    all_results.append((dst_path, label, battery_type, axis, battery_id))

                pbar.update(1)

    print(f"\n총 {len(all_results)}개 패치 생성 완료")

    # 클래스별 통계
    label_counts = Counter([r[1] for r in all_results])
    print("\n클래스별 패치 수:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {i}: {name} = {label_counts.get(i, 0)}개")

    # 배터리 단위 Stratified Split
    print("\n배터리 단위 Stratified Split 생성 중...")

    battery_groups = defaultdict(list)
    battery_main_class = {}

    for dst, label, bt, axis, bid in all_results:
        if bid is not None:
            battery_groups[bid].append((dst, label, bt, axis))

    # 각 배터리의 주요 클래스 결정
    for battery_id, items in battery_groups.items():
        labels = [item[1] for item in items]
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

        print(f"  Class {class_id} ({CLASS_NAMES[class_id]}): {n}개 배터리 → Train {n_train}, Val {n_val}, Test {n - n_train - n_val}")

    # Split 데이터 분류
    train_items = []
    val_items = []
    test_items = []

    for dst, label, bt, axis, bid in all_results:
        if bid in train_ids:
            train_items.append((dst, label, bt, axis))
        elif bid in val_ids:
            val_items.append((dst, label, bt, axis))
        else:
            test_items.append((dst, label, bt, axis))

    # 클래스별 분포 출력
    print("\n최종 Split 클래스별 분포:")
    for name, items in [('Train', train_items), ('Val', val_items), ('Test', test_items)]:
        counts = Counter([item[1] for item in items])
        print(f"  {name}: {len(items)}개 - {dict(counts)}")

    # Split 파일 저장 (메타데이터 포함)
    new_splits_dir = os.path.join(SPLITS_PATH, 'patch')
    os.makedirs(new_splits_dir, exist_ok=True)

    for filename, items in [
        ('battery_train.txt', train_items),
        ('battery_val.txt', val_items),
        ('battery_test.txt', test_items)
    ]:
        filepath = os.path.join(new_splits_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            # 헤더 (주석)
            f.write("# image_path\tlabel\tbattery_type\taxis\n")
            for dst, label, bt, axis in items:
                f.write(f"{dst}\t{label}\t{bt}\t{axis}\n")
        print(f"Created: {filepath} ({len(items)}개)")

    # 용량 확인
    print("\n저장된 데이터 용량:")
    os.system(f"du -sh {args.output}")

    print("\n전처리 완료!")
    print(f"Split 파일 위치: {new_splits_dir}")
    print(f"Split 파일 형식: image_path\\tlabel\\tbattery_type\\taxis")


if __name__ == "__main__":
    main()
