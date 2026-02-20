"""모든 CT Split 디렉토리를 배터리 ID 기준으로 수정

메인 split (training/data/splits/ct/)의 배터리 분할을 기준으로
모든 하위 디렉토리의 split을 동일하게 맞춤
"""

import re
from pathlib import Path
from collections import defaultdict
import shutil

# 메인 split에서 배터리 ID 분할 정보 로드
main_split_dir = Path('training/data/splits/ct')

def extract_battery_id(filepath: str) -> str:
    """파일 경로에서 배터리 ID 추출"""
    filename = Path(filepath).stem
    # CT_module_pouch_015_x_128 또는 CT_module_pouch_015_x_128_crop 등
    # CT_ 제거 후 처음 3개 필드 추출
    name = filename.replace('CT_', '')
    parts = name.split('_')
    if len(parts) >= 3:
        return f"{parts[0]}_{parts[1]}_{parts[2]}"
    return None

# 1. 메인 split에서 배터리-split 매핑 로드
print("=" * 60)
print("모든 CT Split 디렉토리 수정")
print("=" * 60)

battery_to_split = {}
for split_name in ['train', 'val', 'test']:
    split_file = main_split_dir / f'battery_{split_name}.txt'
    if split_file.exists():
        with open(split_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    filepath = line.split('\t')[0]
                    battery_id = extract_battery_id(filepath)
                    if battery_id:
                        battery_to_split[battery_id] = split_name

print(f"\n[1] 메인 split 배터리 매핑 로드: {len(battery_to_split)}개")
print(f"  Train: {sum(1 for v in battery_to_split.values() if v == 'train')}개")
print(f"  Val: {sum(1 for v in battery_to_split.values() if v == 'val')}개")
print(f"  Test: {sum(1 for v in battery_to_split.values() if v == 'test')}개")

# 2. 수정할 하위 디렉토리 목록
subdirs = ['cropped', 'patch', 'defect_direct', 'defect_random', 'cell', 'module']

for subdir in subdirs:
    subdir_path = main_split_dir / subdir
    if not subdir_path.exists():
        print(f"\n[SKIP] {subdir}: 디렉토리 없음")
        continue

    # 기존 파일 확인
    train_file = subdir_path / 'battery_train.txt'
    val_file = subdir_path / 'battery_val.txt'
    test_file = subdir_path / 'battery_test.txt'

    if not train_file.exists():
        print(f"\n[SKIP] {subdir}: split 파일 없음")
        continue

    print(f"\n{'='*60}")
    print(f"[처리 중] {subdir}")
    print("=" * 60)

    # 3. 모든 데이터 로드
    all_data = []
    for split_file in [train_file, val_file, test_file]:
        if split_file.exists():
            with open(split_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split('\t')
                        if len(parts) >= 2:
                            all_data.append((parts[0], parts[1]))

    print(f"  전체 데이터: {len(all_data):,}개")

    # 4. 배터리 ID 기준으로 재분할
    train_data = []
    val_data = []
    test_data = []
    unknown_batteries = set()

    for filepath, label in all_data:
        battery_id = extract_battery_id(filepath)
        if battery_id and battery_id in battery_to_split:
            split = battery_to_split[battery_id]
            if split == 'train':
                train_data.append((filepath, label))
            elif split == 'val':
                val_data.append((filepath, label))
            else:
                test_data.append((filepath, label))
        else:
            unknown_batteries.add(battery_id)

    if unknown_batteries:
        print(f"  ⚠️ 매핑 없는 배터리: {len(unknown_batteries)}개")

    print(f"  분할 결과: Train={len(train_data):,}, Val={len(val_data):,}, Test={len(test_data):,}")

    # 5. 백업
    backup_dir = subdir_path / 'backup_before_fix'
    backup_dir.mkdir(exist_ok=True)
    for f in [train_file, val_file, test_file]:
        if f.exists():
            shutil.copy(f, backup_dir / f.name)

    # 6. 저장
    for name, data in [('battery_train.txt', train_data),
                       ('battery_val.txt', val_data),
                       ('battery_test.txt', test_data)]:
        with open(subdir_path / name, 'w') as f:
            for filepath, label in data:
                f.write(f"{filepath}\t{label}\n")

    # 7. 검증
    new_train_bat = set()
    new_val_bat = set()
    new_test_bat = set()

    for filepath, _ in train_data:
        bat = extract_battery_id(filepath)
        if bat:
            new_train_bat.add(bat)
    for filepath, _ in val_data:
        bat = extract_battery_id(filepath)
        if bat:
            new_val_bat.add(bat)
    for filepath, _ in test_data:
        bat = extract_battery_id(filepath)
        if bat:
            new_test_bat.add(bat)

    overlap1 = new_train_bat & new_val_bat
    overlap2 = new_train_bat & new_test_bat
    overlap3 = new_val_bat & new_test_bat

    if not overlap1 and not overlap2 and not overlap3:
        print(f"  ✅ 검증 통과: 누수 없음")
    else:
        print(f"  ❌ Train-Val: {len(overlap1)}, Train-Test: {len(overlap2)}, Val-Test: {len(overlap3)}")

print("\n" + "=" * 60)
print("✅ 모든 CT Split 수정 완료!")
print("=" * 60)
