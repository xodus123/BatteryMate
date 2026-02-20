"""배터리 단위 라벨링 변환 스크립트

현재: 슬라이스별 라벨 (이미지에 결함이 보이면 불량)
변경: 배터리별 라벨 (배터리에 결함이 있으면 전체 불량)

라벨 규칙:
- Cell: 하나라도 porosity(1)이 있으면 전체 1
- Module: resin(4) > porosity(3) > normal(2) 우선순위
"""

import os
import re
from pathlib import Path
from collections import defaultdict

def extract_battery_id(filepath):
    """파일 경로에서 배터리 ID 추출"""
    filename = os.path.basename(filepath)

    # CT_cell_pouch_101_x_001.jpg → cell_pouch_101
    # CT_module_pouch_015_y_002.jpg → module_pouch_015
    match = re.search(r'(cell_pouch_\d+|module_pouch_\d+)', filename)
    if match:
        return match.group(1)
    return None

def get_battery_type(battery_id):
    """배터리 타입 반환 (cell/module)"""
    if 'cell' in battery_id:
        return 'cell'
    elif 'module' in battery_id:
        return 'module'
    return None

def determine_battery_label(labels, battery_type):
    """배터리의 최종 라벨 결정

    Cell: 0(normal), 1(porosity) → 하나라도 1이면 1
    Module: 2(normal), 3(porosity), 4(resin) → 4 > 3 > 2 우선순위
    """
    labels = set(labels)

    if battery_type == 'cell':
        if 1 in labels:
            return 1  # porosity
        return 0  # normal

    elif battery_type == 'module':
        if 4 in labels:
            return 4  # resin_overflow (최우선)
        elif 3 in labels:
            return 3  # porosity
        return 2  # normal

    return None

def process_split_file(input_path, output_path):
    """Split 파일을 배터리 단위 라벨로 변환"""

    # 1단계: 모든 데이터 읽기 및 배터리별 라벨 수집
    battery_labels = defaultdict(list)
    all_data = []

    with open(input_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split('\t')
            if len(parts) != 2:
                continue

            filepath, label = parts[0], int(parts[1])
            battery_id = extract_battery_id(filepath)

            if battery_id:
                battery_labels[battery_id].append(label)
                all_data.append((filepath, label, battery_id))

    # 2단계: 배터리별 최종 라벨 결정
    battery_final_labels = {}
    for battery_id, labels in battery_labels.items():
        battery_type = get_battery_type(battery_id)
        final_label = determine_battery_label(labels, battery_type)
        battery_final_labels[battery_id] = final_label

    # 3단계: 새 라벨로 파일 작성
    changed_count = 0
    with open(output_path, 'w') as f:
        for filepath, old_label, battery_id in all_data:
            new_label = battery_final_labels[battery_id]
            if old_label != new_label:
                changed_count += 1
            f.write(f"{filepath}\t{new_label}\n")

    return len(all_data), changed_count, battery_final_labels

def print_statistics(battery_labels, split_name):
    """통계 출력"""
    cell_normal = sum(1 for bid, lbl in battery_labels.items() if 'cell' in bid and lbl == 0)
    cell_porosity = sum(1 for bid, lbl in battery_labels.items() if 'cell' in bid and lbl == 1)
    module_normal = sum(1 for bid, lbl in battery_labels.items() if 'module' in bid and lbl == 2)
    module_porosity = sum(1 for bid, lbl in battery_labels.items() if 'module' in bid and lbl == 3)
    module_resin = sum(1 for bid, lbl in battery_labels.items() if 'module' in bid and lbl == 4)

    print(f"\n  {split_name} 배터리 단위 분포:")
    print(f"    Cell   - normal: {cell_normal}, porosity: {cell_porosity}")
    print(f"    Module - normal: {module_normal}, porosity: {module_porosity}, resin: {module_resin}")

def main():
    splits_dir = Path("training/data/splits/ct")

    # 입력/출력 파일 정의
    splits = [
        ("preprocessed_train.txt", "battery_train.txt"),
        ("preprocessed_val.txt", "battery_val.txt"),
        ("preprocessed_test.txt", "battery_test.txt"),
    ]

    print("=" * 60)
    print("배터리 단위 라벨링 변환")
    print("=" * 60)
    print("\n라벨 규칙:")
    print("  Cell: 하나라도 porosity(1) → 전체 1")
    print("  Module: resin(4) > porosity(3) > normal(2)")

    total_changed = 0
    total_samples = 0

    for input_name, output_name in splits:
        input_path = splits_dir / input_name
        output_path = splits_dir / output_name

        if not input_path.exists():
            print(f"\n⚠️ 파일 없음: {input_path}")
            continue

        print(f"\n{'=' * 60}")
        print(f"처리 중: {input_name} → {output_name}")

        num_samples, num_changed, battery_labels = process_split_file(input_path, output_path)

        print(f"  총 샘플: {num_samples:,}")
        print(f"  라벨 변경: {num_changed:,} ({num_changed/num_samples*100:.1f}%)")

        print_statistics(battery_labels, input_name.replace('.txt', ''))

        total_changed += num_changed
        total_samples += num_samples

    print(f"\n{'=' * 60}")
    print(f"✅ 변환 완료!")
    print(f"   총 샘플: {total_samples:,}")
    print(f"   라벨 변경: {total_changed:,} ({total_changed/total_samples*100:.1f}%)")
    print(f"\n새 파일 위치:")
    print(f"   {splits_dir}/battery_train.txt")
    print(f"   {splits_dir}/battery_val.txt")
    print(f"   {splits_dir}/battery_test.txt")
    print("=" * 60)

if __name__ == "__main__":
    main()
