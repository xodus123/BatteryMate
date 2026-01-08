"""
최종 데이터 분할 스크립트

구조:
1. CT 통합 (Cell + Module): 5클래스 CNN
   - cell_normal, cell_porosity
   - module_normal, module_porosity, module_resin_overflow

2. RGB: 3클래스 AE용
   - normal, pollution, mixed

3. 앙상블용: CT-RGB 겹치는 배터리만 추출

사용법:
    python scripts/create_splits_final.py
"""

import json
import os
import random
import re
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional, Set
from multiprocessing import Pool


DATA_BASE = Path("data/103.배터리 불량 이미지 데이터/3.개방데이터/1.데이터")

DEFECT_MAPPING = {
    'ct': {'porosity': 'porosity', 'resin overflow': 'resin_overflow', 'resin_overflow': 'resin_overflow'},
    'rgb': {'pollution': 'pollution', 'Pollution': 'pollution', 'damaged': 'damaged', 'Damaged': 'damaged'}
}


def fast_listdir(directory: Path, prefix: str = "", suffix: str = ".json") -> List[Path]:
    files = []
    try:
        with os.scandir(directory) as entries:
            for entry in entries:
                if entry.is_file() and entry.name.endswith(suffix):
                    if not prefix or entry.name.startswith(prefix):
                        files.append(Path(entry.path))
    except:
        pass
    return files


def process_label(args: Tuple[Path, str]) -> Optional[dict]:
    label_path, modality = args
    try:
        with open(label_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        is_normal = data.get('image_info', {}).get('is_normal', True)
        defect_types = []

        if not is_normal:
            defects = data.get('defects', [])
            if defects:
                mapping = DEFECT_MAPPING[modality]
                for d in defects:
                    name = d.get('name', '')
                    mapped = mapping.get(name, name.lower())
                    if mapped not in defect_types:
                        defect_types.append(mapped)
            else:
                is_normal = True

        return {
            'label_path': str(label_path),
            'filename': label_path.name,
            'is_normal': is_normal,
            'defect_types': defect_types
        }
    except:
        return None


def extract_battery_id(filename: str, modality: str, data_type: str) -> Optional[int]:
    if modality == 'ct':
        match = re.search(rf'CT_{data_type}_pouch_(\d+)_', filename)
    else:
        match = re.search(r'RGB_cell_cylindrical_(\d+)_', filename)
    return int(match.group(1)) if match else None


def find_image_path(label_path: Path, modality: str, data_type: str, source: str) -> Optional[Path]:
    filename = label_path.stem

    if modality == 'ct':
        ext = '.jpg'
        if source == 'train':
            for suffix in ['_1', '_2']:
                img_dir = DATA_BASE / f"Training/01.원천데이터/TS_CT_Datasets_images{suffix}"
                img_path = img_dir / f"{filename}{ext}"
                if img_path.exists():
                    return img_path
        else:
            img_dir = DATA_BASE / "Validation/01.원천데이터/VS_CT_Datasets_images"
            img_path = img_dir / f"{filename}{ext}"
            if img_path.exists():
                return img_path
    else:
        ext = '.png'
        if source == 'train':
            for suffix in ['_1', '_2', '_3', '_4']:
                img_dir = DATA_BASE / f"Training/01.원천데이터/TS_Exterior_Img_Datasets_images{suffix}"
                img_path = img_dir / f"{filename}{ext}"
                if img_path.exists():
                    return img_path
        else:
            img_dir = DATA_BASE / "Validation/01.원천데이터/VS_Exterior_Img_Datasets_images"
            img_path = img_dir / f"{filename}{ext}"
            if img_path.exists():
                return img_path
    return None


def collect_ct_data(num_workers: int = 12) -> Tuple[Dict[int, dict], Dict[int, dict]]:
    """CT Cell과 CT Module 데이터 수집"""

    train_label_dir = DATA_BASE / "Training/02.라벨링데이터/TL_CT_Datasets_label"
    val_label_dir = DATA_BASE / "Validation/02.라벨링데이터/VL_CT_Datasets_label"

    cell_data = defaultdict(lambda: {'items': [], 'defect_counts': Counter(), 'type': 'cell'})
    module_data = defaultdict(lambda: {'items': [], 'defect_counts': Counter(), 'type': 'module'})

    for data_type, battery_data in [('cell', cell_data), ('module', module_data)]:
        prefix = f"CT_{data_type}_"

        print(f"  CT {data_type} 파일 수집 중...")
        train_files = fast_listdir(train_label_dir, prefix)
        val_files = fast_listdir(val_label_dir, prefix)
        print(f"    Training: {len(train_files):,}개, Validation: {len(val_files):,}개")

        print(f"  라벨 처리 중 ({num_workers} workers)...")
        all_args = [(f, 'ct') for f in train_files + val_files]
        train_count = len(train_files)

        with Pool(num_workers) as pool:
            results = pool.map(process_label, all_args)

        for i, result in enumerate(results):
            if result is None:
                continue

            source = 'train' if i < train_count else 'val'
            bid = extract_battery_id(result['filename'], 'ct', data_type)
            if bid is None:
                continue

            battery_data[bid]['items'].append({
                'label_path': Path(result['label_path']),
                'source': source,
                'is_normal': result['is_normal'],
                'defect_types': result['defect_types']
            })

            if result['is_normal']:
                battery_data[bid]['defect_counts']['normal'] += 1
            else:
                for dt in result['defect_types']:
                    battery_data[bid]['defect_counts'][dt] += 1

        # 배터리 클래스 결정
        for bid, data in battery_data.items():
            counts = data['defect_counts']
            if counts.get('resin_overflow', 0) > 0:
                data['class'] = f'{data_type}_resin_overflow'
            elif counts.get('porosity', 0) > 0:
                data['class'] = f'{data_type}_porosity'
            else:
                data['class'] = f'{data_type}_normal'

    return dict(cell_data), dict(module_data)


def collect_rgb_data(num_workers: int = 12) -> Dict[int, dict]:
    """RGB 데이터 수집"""

    train_label_dir = DATA_BASE / "Training/02.라벨링데이터/TL_Exterior_Img_Datasets_label"
    val_label_dir = DATA_BASE / "Validation/02.라벨링데이터/VL_Exterior_Img_Datasets_label"

    battery_data = defaultdict(lambda: {'items': [], 'defect_counts': Counter(), 'type': 'rgb'})

    print(f"  RGB 파일 수집 중...")
    train_files = fast_listdir(train_label_dir)
    val_files = fast_listdir(val_label_dir)
    print(f"    Training: {len(train_files):,}개, Validation: {len(val_files):,}개")

    print(f"  라벨 처리 중 ({num_workers} workers)...")
    all_args = [(f, 'rgb') for f in train_files + val_files]
    train_count = len(train_files)

    with Pool(num_workers) as pool:
        results = pool.map(process_label, all_args)

    for i, result in enumerate(results):
        if result is None:
            continue

        source = 'train' if i < train_count else 'val'
        bid = extract_battery_id(result['filename'], 'rgb', 'cell')
        if bid is None:
            continue

        battery_data[bid]['items'].append({
            'label_path': Path(result['label_path']),
            'source': source,
            'is_normal': result['is_normal'],
            'defect_types': result['defect_types']
        })

        if result['is_normal']:
            battery_data[bid]['defect_counts']['normal'] += 1
        else:
            for dt in result['defect_types']:
                battery_data[bid]['defect_counts'][dt] += 1

    # 배터리 클래스 결정
    for bid, data in battery_data.items():
        counts = data['defect_counts']
        has_pollution = counts.get('pollution', 0) > 0
        has_damaged = counts.get('damaged', 0) > 0

        if has_pollution and has_damaged:
            data['class'] = 'mixed'
        elif has_pollution:
            data['class'] = 'pollution'
        elif has_damaged:
            data['class'] = 'damaged'
        else:
            data['class'] = 'normal'

    return dict(battery_data)


def stratified_split(
    battery_ids_by_class: Dict[str, List[int]],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[List[int], List[int], List[int]]:
    """층화 분할"""
    random.seed(seed)

    train_ids, val_ids, test_ids = [], [], []

    for cls, ids in battery_ids_by_class.items():
        ids = ids.copy()
        random.shuffle(ids)

        n = len(ids)
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio)) if n > 2 else 0

        train_ids.extend(ids[:n_train])
        val_ids.extend(ids[n_train:n_train + n_val])
        test_ids.extend(ids[n_train + n_val:])

        print(f"    {cls}: {n} → Train {n_train}, Val {n_val}, Test {n - n_train - n_val}")

    return train_ids, val_ids, test_ids


def process_ct_split(cell_data, module_data, battery_ids, ct_classes):
    """CT 통합 데이터 처리"""
    processed = []

    for bid in battery_ids:
        # Cell 데이터
        if bid in cell_data:
            data = cell_data[bid]
            for item in data['items']:
                if item['is_normal']:
                    label_name = 'cell_normal'
                elif item['defect_types']:
                    dt = item['defect_types'][0]
                    label_name = f'cell_{dt}'
                else:
                    label_name = 'cell_normal'

                if label_name in ct_classes:
                    label_idx = ct_classes.index(label_name)
                    image_path = find_image_path(item['label_path'], 'ct', 'cell', item['source'])
                    if image_path and image_path.exists():
                        processed.append({
                            'image_path': str(image_path),
                            'label': label_idx,
                            'label_name': label_name
                        })

        # Module 데이터
        if bid in module_data:
            data = module_data[bid]
            for item in data['items']:
                if item['is_normal']:
                    label_name = 'module_normal'
                elif item['defect_types']:
                    dt = item['defect_types'][0]
                    label_name = f'module_{dt}'
                else:
                    label_name = 'module_normal'

                if label_name in ct_classes:
                    label_idx = ct_classes.index(label_name)
                    image_path = find_image_path(item['label_path'], 'ct', 'module', item['source'])
                    if image_path and image_path.exists():
                        processed.append({
                            'image_path': str(image_path),
                            'label': label_idx,
                            'label_name': label_name
                        })

    return processed


def process_rgb_split(rgb_data, battery_ids, rgb_classes, defect_only=False):
    """RGB 데이터 처리"""
    processed = []

    for bid in battery_ids:
        if bid not in rgb_data:
            continue

        data = rgb_data[bid]
        for item in data['items']:
            if item['is_normal']:
                if defect_only:
                    continue  # AE 학습용: 불량만
                label_name = 'normal'
            else:
                # 결함 유형 결정
                has_pollution = 'pollution' in item['defect_types']
                has_damaged = 'damaged' in item['defect_types']

                if has_pollution and has_damaged:
                    label_name = 'mixed'
                elif has_pollution:
                    label_name = 'pollution'
                elif has_damaged:
                    label_name = 'mixed'  # damaged만 있어도 mixed로
                else:
                    label_name = 'pollution'  # 기본값

            if label_name in rgb_classes:
                label_idx = rgb_classes.index(label_name)
                image_path = find_image_path(item['label_path'], 'rgb', 'cell', item['source'])
                if image_path and image_path.exists():
                    processed.append({
                        'image_path': str(image_path),
                        'label': label_idx,
                        'label_name': label_name
                    })

    return processed


def save_split_file(data, output_path, class_names):
    """Split 파일 저장"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    class_counts = Counter()

    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(f"{item['image_path']}\t{item['label']}\n")
            class_counts[item['label_name']] += 1

    print(f"  저장: {output_path}")
    print(f"    총 {len(data):,}개")
    for name in class_names:
        count = class_counts.get(name, 0)
        pct = count / len(data) * 100 if data else 0
        print(f"    - {name}: {count:,} ({pct:.1f}%)")


def main():
    output_base = Path("training/data/splits")
    seed = 42

    print("=" * 60)
    print("최종 데이터 분할 (CT 통합 + RGB + 앙상블)")
    print("=" * 60)
    print()

    # ========== 데이터 수집 ==========
    print("📊 CT 데이터 수집")
    print("-" * 40)
    cell_data, module_data = collect_ct_data()

    cell_ids = set(cell_data.keys())
    module_ids = set(module_data.keys())
    ct_all_ids = cell_ids | module_ids

    print(f"\n  CT Cell: {len(cell_ids)}개 배터리")
    print(f"  CT Module: {len(module_ids)}개 배터리")
    print(f"  CT 통합: {len(ct_all_ids)}개 배터리")

    print()
    print("📊 RGB 데이터 수집")
    print("-" * 40)
    rgb_data = collect_rgb_data()
    rgb_ids = set(rgb_data.keys())
    print(f"\n  RGB: {len(rgb_ids)}개 배터리")

    # ========== 겹치는 배터리 확인 ==========
    ct_rgb_overlap = ct_all_ids & rgb_ids
    print(f"\n  CT ∩ RGB 겹침: {len(ct_rgb_overlap)}개 배터리 (앙상블 가능)")

    # ========== CT 통합 분할 ==========
    print()
    print("=" * 60)
    print("📊 CT 통합 분할 (5클래스)")
    print("=" * 60)

    # 클래스별 배터리 분류
    ct_by_class = defaultdict(list)
    for bid in ct_all_ids:
        if bid in cell_data:
            ct_by_class[cell_data[bid]['class']].append(bid)
        if bid in module_data:
            ct_by_class[module_data[bid]['class']].append(bid)

    print("  클래스별 배터리:")
    for cls, ids in sorted(ct_by_class.items()):
        print(f"    {cls}: {len(ids)}개")

    ct_train_ids, ct_val_ids, ct_test_ids = stratified_split(
        ct_by_class, 0.7, 0.15, seed
    )

    ct_classes = ['cell_normal', 'cell_porosity', 'module_normal', 'module_porosity', 'module_resin_overflow']

    ct_train = process_ct_split(cell_data, module_data, ct_train_ids, ct_classes)
    ct_val = process_ct_split(cell_data, module_data, ct_val_ids, ct_classes)
    ct_test = process_ct_split(cell_data, module_data, ct_test_ids, ct_classes)

    print("\n  Split 저장:")
    save_split_file(ct_train, output_base / 'ct/train.txt', ct_classes)
    save_split_file(ct_val, output_base / 'ct/val.txt', ct_classes)
    save_split_file(ct_test, output_base / 'ct/test.txt', ct_classes)

    # ========== RGB 분할 ==========
    print()
    print("=" * 60)
    print("📊 RGB 분할 (3클래스, AE용)")
    print("=" * 60)

    rgb_by_class = defaultdict(list)
    for bid, data in rgb_data.items():
        rgb_by_class[data['class']].append(bid)

    print("  클래스별 배터리:")
    for cls, ids in sorted(rgb_by_class.items()):
        print(f"    {cls}: {len(ids)}개")

    # 균형 샘플링 (각 100개)
    sampled_rgb = {}
    for cls in ['normal', 'pollution', 'mixed']:
        available = rgb_by_class.get(cls, [])
        random.seed(seed)
        if len(available) > 100:
            sampled_rgb[cls] = random.sample(available, 100)
        else:
            sampled_rgb[cls] = available
        print(f"    {cls}: {len(available)} → {len(sampled_rgb[cls])}개 샘플링")

    rgb_train_ids, rgb_val_ids, rgb_test_ids = stratified_split(
        sampled_rgb, 0.7, 0.15, seed
    )

    rgb_classes = ['normal', 'pollution', 'mixed']

    # Train: 불량만 (AE용)
    rgb_train = process_rgb_split(rgb_data, rgb_train_ids, rgb_classes, defect_only=True)
    # Val/Test: 정상+불량
    rgb_val = process_rgb_split(rgb_data, rgb_val_ids, rgb_classes, defect_only=False)
    rgb_test = process_rgb_split(rgb_data, rgb_test_ids, rgb_classes, defect_only=False)

    print("\n  Split 저장:")
    print("  [Train: 불량만 - AE 학습용]")
    save_split_file(rgb_train, output_base / 'rgb/train.txt', rgb_classes)
    print("  [Val: 정상+불량]")
    save_split_file(rgb_val, output_base / 'rgb/val.txt', rgb_classes)
    print("  [Test: 정상+불량]")
    save_split_file(rgb_test, output_base / 'rgb/test.txt', rgb_classes)

    # ========== 앙상블용 분할 ==========
    print()
    print("=" * 60)
    print("📊 앙상블용 분할 (CT ∩ RGB 겹치는 배터리)")
    print("=" * 60)

    # 겹치는 배터리만 사용
    overlap_list = list(ct_rgb_overlap)
    random.seed(seed)
    random.shuffle(overlap_list)

    n = len(overlap_list)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)

    ensemble_train_ids = overlap_list[:n_train]
    ensemble_val_ids = overlap_list[n_train:n_train + n_val]
    ensemble_test_ids = overlap_list[n_train + n_val:]

    print(f"  앙상블 배터리: {n}개 → Train {len(ensemble_train_ids)}, Val {len(ensemble_val_ids)}, Test {len(ensemble_test_ids)}")

    # CT 부분
    ensemble_ct_train = process_ct_split(cell_data, module_data, ensemble_train_ids, ct_classes)
    ensemble_ct_val = process_ct_split(cell_data, module_data, ensemble_val_ids, ct_classes)
    ensemble_ct_test = process_ct_split(cell_data, module_data, ensemble_test_ids, ct_classes)

    print("\n  앙상블 CT Split 저장:")
    save_split_file(ensemble_ct_train, output_base / 'ensemble/ct_train.txt', ct_classes)
    save_split_file(ensemble_ct_val, output_base / 'ensemble/ct_val.txt', ct_classes)
    save_split_file(ensemble_ct_test, output_base / 'ensemble/ct_test.txt', ct_classes)

    # RGB 부분
    ensemble_rgb_train = process_rgb_split(rgb_data, ensemble_train_ids, rgb_classes, defect_only=True)
    ensemble_rgb_val = process_rgb_split(rgb_data, ensemble_val_ids, rgb_classes, defect_only=False)
    ensemble_rgb_test = process_rgb_split(rgb_data, ensemble_test_ids, rgb_classes, defect_only=False)

    print("\n  앙상블 RGB Split 저장:")
    save_split_file(ensemble_rgb_train, output_base / 'ensemble/rgb_train.txt', rgb_classes)
    save_split_file(ensemble_rgb_val, output_base / 'ensemble/rgb_val.txt', rgb_classes)
    save_split_file(ensemble_rgb_test, output_base / 'ensemble/rgb_test.txt', rgb_classes)

    # ========== 요약 ==========
    print()
    print("=" * 60)
    print("✅ 완료!")
    print("=" * 60)

    print(f"\n📁 출력: {output_base}")

    print(f"\n[CT 통합] (5클래스)")
    print(f"  Train: {len(ct_train):,}개")
    print(f"  Val:   {len(ct_val):,}개")
    print(f"  Test:  {len(ct_test):,}개")

    print(f"\n[RGB] (3클래스, AE용)")
    print(f"  Train: {len(rgb_train):,}개 (불량만)")
    print(f"  Val:   {len(rgb_val):,}개")
    print(f"  Test:  {len(rgb_test):,}개")

    print(f"\n[앙상블] ({len(ct_rgb_overlap)}개 배터리)")
    print(f"  CT Train: {len(ensemble_ct_train):,}개, RGB Train: {len(ensemble_rgb_train):,}개")
    print(f"  CT Val:   {len(ensemble_ct_val):,}개, RGB Val:   {len(ensemble_rgb_val):,}개")
    print(f"  CT Test:  {len(ensemble_ct_test):,}개, RGB Test:  {len(ensemble_rgb_test):,}개")

    # 배터리 ID 저장
    with open(output_base / 'ensemble/battery_ids.txt', 'w') as f:
        f.write("# 앙상블용 배터리 ID (CT ∩ RGB)\n")
        f.write(f"# Train: {len(ensemble_train_ids)}개\n")
        for bid in sorted(ensemble_train_ids):
            f.write(f"train\t{bid}\n")
        f.write(f"# Val: {len(ensemble_val_ids)}개\n")
        for bid in sorted(ensemble_val_ids):
            f.write(f"val\t{bid}\n")
        f.write(f"# Test: {len(ensemble_test_ids)}개\n")
        for bid in sorted(ensemble_test_ids):
            f.write(f"test\t{bid}\n")

    print(f"\n  배터리 ID 저장: {output_base / 'ensemble/battery_ids.txt'}")


if __name__ == "__main__":
    main()
