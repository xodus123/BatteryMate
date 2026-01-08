"""Train/Val/Test 데이터 누수 확인 스크립트"""
import re
from pathlib import Path
from collections import defaultdict

def extract_battery_id(filepath):
    """파일 경로에서 배터리 ID 추출"""
    # CT_cell_pouch_141_z_141.jpg -> cell_141
    # CT_module_pouch_036_z_216.jpg -> module_036
    match = re.search(r'CT_(cell|module)_pouch_(\d+)_[xyz]_\d+\.jpg', filepath)
    if match:
        battery_type = match.group(1)  # cell or module
        battery_id = match.group(2)
        return f"{battery_type}_{battery_id}"
    return None

def load_battery_ids(split_file):
    """Split 파일에서 배터리 ID 추출"""
    battery_ids = set()
    battery_files = defaultdict(list)

    with open(split_file, 'r') as f:
        for line in f:
            filepath = line.strip().split('\t')[0]
            battery_id = extract_battery_id(filepath)
            if battery_id:
                battery_ids.add(battery_id)
                battery_files[battery_id].append(filepath)

    return battery_ids, battery_files

# Train/Val/Test 파일 로드
print("🔍 Data Leakage 분석 중 (Train/Val/Test)...\n")

train_ids, train_files = load_battery_ids('training/data/splits/ct_cnn/train.txt')
val_ids, val_files = load_battery_ids('training/data/splits/ct_cnn/val.txt')
test_ids, test_files = load_battery_ids('training/data/splits/ct_cnn/test.txt')

print(f"📊 통계:")
print(f"  - Train 고유 배터리: {len(train_ids)}개")
print(f"  - Val   고유 배터리: {len(val_ids)}개")
print(f"  - Test  고유 배터리: {len(test_ids)}개")
print(f"  - 전체  고유 배터리: {len(train_ids | val_ids | test_ids)}개")

# 겹치는 배터리 ID 확인
train_val_overlap = train_ids & val_ids
train_test_overlap = train_ids & test_ids
val_test_overlap = val_ids & test_ids

total_overlaps = len(train_val_overlap) + len(train_test_overlap) + len(val_test_overlap)

print(f"\n{'='*60}")
if total_overlaps > 0:
    print(f"⚠️  경고: Data Leakage 발견!")
    print(f"{'='*60}")

    if len(train_val_overlap) > 0:
        print(f"\n  🔴 Train-Val 겹침: {len(train_val_overlap)}개 배터리")
        for i, battery_id in enumerate(sorted(train_val_overlap)[:5]):
            print(f"    {i+1}. {battery_id}: Train {len(train_files[battery_id])}장, Val {len(val_files[battery_id])}장")
        if len(train_val_overlap) > 5:
            print(f"    ... 외 {len(train_val_overlap) - 5}개")

    if len(train_test_overlap) > 0:
        print(f"\n  🔴 Train-Test 겹침: {len(train_test_overlap)}개 배터리")
        for i, battery_id in enumerate(sorted(train_test_overlap)[:5]):
            print(f"    {i+1}. {battery_id}: Train {len(train_files[battery_id])}장, Test {len(test_files[battery_id])}장")
        if len(train_test_overlap) > 5:
            print(f"    ... 외 {len(train_test_overlap) - 5}개")

    if len(val_test_overlap) > 0:
        print(f"\n  🔴 Val-Test 겹침: {len(val_test_overlap)}개 배터리")
        for i, battery_id in enumerate(sorted(val_test_overlap)[:5]):
            print(f"    {i+1}. {battery_id}: Val {len(val_files[battery_id])}장, Test {len(test_files[battery_id])}장")
        if len(val_test_overlap) > 5:
            print(f"    ... 외 {len(val_test_overlap) - 5}개")

    print(f"\n  💡 해결 방법:")
    print(f"     1. 배터리 ID 단위로 Train/Val/Test 분할 필요")
    print(f"     2. data_splitter.py 수정 필요")

else:
    print(f"✅ Data Leakage 없음!")
    print(f"{'='*60}")
    print(f"  Train/Val/Test가 배터리 단위로 완전히 분리되어 있습니다.")
    print(f"\n  ✓ Train-Val 겹침: 0개")
    print(f"  ✓ Train-Test 겹침: 0개")
    print(f"  ✓ Val-Test 겹침: 0개")

print()
