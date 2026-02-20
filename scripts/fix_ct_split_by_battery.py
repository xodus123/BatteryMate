"""CT 데이터 배터리 ID별 Split 재생성

문제: 같은 배터리의 다른 슬라이스/축이 train/val/test에 분산됨 → 데이터 누수
해결: 배터리 ID 단위로 split하여 같은 배터리는 같은 split에만 존재하도록 수정
"""

import re
import random
from pathlib import Path
from collections import defaultdict

random.seed(42)

# 경로 설정
splits_dir = Path('training/data/splits/ct')

def extract_battery_id(filepath: str) -> str:
    """파일 경로에서 배터리 ID 추출

    예: CT_module_pouch_015_x_128.jpg → module_pouch_015
        CT_cell_pouch_101_y_050.jpg → cell_pouch_101
    """
    filename = Path(filepath).stem  # CT_module_pouch_015_x_128
    # CT_ 제거 후 처음 3개 필드 추출
    parts = filename.replace('CT_', '').split('_')
    if len(parts) >= 3:
        return f"{parts[0]}_{parts[1]}_{parts[2]}"
    return None

# 1. 모든 데이터 로드
print("=" * 60)
print("CT 데이터 배터리 단위 Split 재생성")
print("=" * 60)

all_data = []
for split_file in ['battery_train.txt', 'battery_val.txt', 'battery_test.txt']:
    filepath = splits_dir / split_file
    if filepath.exists():
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        all_data.append((parts[0], int(parts[1])))

print(f"\n[1] 전체 데이터 로드: {len(all_data):,}개")

# 2. 배터리별로 그룹화
battery_data = defaultdict(list)
for filepath, label in all_data:
    battery_id = extract_battery_id(filepath)
    if battery_id:
        battery_data[battery_id].append((filepath, label))

print(f"[2] 고유 배터리 수: {len(battery_data)}개")

# 3. 배터리별 클래스 확인 (배터리는 단일 클래스만 가져야 함)
battery_labels = {}
for battery_id, files in battery_data.items():
    labels = set(label for _, label in files)
    if len(labels) > 1:
        print(f"  ⚠️ {battery_id}: 여러 클래스 {labels}")
    battery_labels[battery_id] = list(labels)[0]

# 4. 클래스별 배터리 분류
class_batteries = defaultdict(list)
for battery_id, label in battery_labels.items():
    class_batteries[label].append(battery_id)

print(f"\n[3] 클래스별 배터리 분포:")
for label in sorted(class_batteries.keys()):
    print(f"  Class {label}: {len(class_batteries[label])}개 배터리")

# 5. 클래스별로 배터리를 train/val/test로 분할 (70/15/15)
train_batteries = []
val_batteries = []
test_batteries = []

for label in sorted(class_batteries.keys()):
    batteries = class_batteries[label].copy()
    random.shuffle(batteries)

    n = len(batteries)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)

    train_batteries.extend(batteries[:n_train])
    val_batteries.extend(batteries[n_train:n_train + n_val])
    test_batteries.extend(batteries[n_train + n_val:])

print(f"\n[4] 배터리 분할 결과:")
print(f"  Train: {len(train_batteries)}개 배터리")
print(f"  Val: {len(val_batteries)}개 배터리")
print(f"  Test: {len(test_batteries)}개 배터리")

# 6. 파일 수집
train_files = []
val_files = []
test_files = []

for battery_id in train_batteries:
    train_files.extend(battery_data[battery_id])
for battery_id in val_batteries:
    val_files.extend(battery_data[battery_id])
for battery_id in test_batteries:
    test_files.extend(battery_data[battery_id])

# 셔플
random.shuffle(train_files)
random.shuffle(val_files)
random.shuffle(test_files)

print(f"\n[5] 파일 분할 결과:")
print(f"  Train: {len(train_files):,}개 파일")
print(f"  Val: {len(val_files):,}개 파일")
print(f"  Test: {len(test_files):,}개 파일")

# 7. 클래스 분포 확인
def count_classes(files):
    counts = defaultdict(int)
    for _, label in files:
        counts[label] += 1
    return dict(sorted(counts.items()))

print(f"\n[6] 클래스 분포:")
print(f"  Train: {count_classes(train_files)}")
print(f"  Val: {count_classes(val_files)}")
print(f"  Test: {count_classes(test_files)}")

# 8. 백업 및 저장
backup_dir = splits_dir / 'backup_before_battery_fix'
backup_dir.mkdir(exist_ok=True)

import shutil
for f in ['battery_train.txt', 'battery_val.txt', 'battery_test.txt']:
    src = splits_dir / f
    if src.exists():
        shutil.copy(src, backup_dir / f)
print(f"\n[7] 기존 파일 백업: {backup_dir}")

# 저장
for name, files in [('battery_train.txt', train_files),
                    ('battery_val.txt', val_files),
                    ('battery_test.txt', test_files)]:
    with open(splits_dir / name, 'w') as f:
        for filepath, label in files:
            f.write(f"{filepath}\t{label}\n")
    print(f"  저장: {name} ({len(files):,}개)")

# 9. 검증
print(f"\n[8] 데이터 누수 검증...")
train_bat_set = set(train_batteries)
val_bat_set = set(val_batteries)
test_bat_set = set(test_batteries)

train_val_overlap = train_bat_set & val_bat_set
train_test_overlap = train_bat_set & test_bat_set
val_test_overlap = val_bat_set & test_bat_set

if not train_val_overlap and not train_test_overlap and not val_test_overlap:
    print("  ✅ 검증 통과: 데이터 누수 없음")
else:
    print(f"  ❌ Train-Val 중복: {len(train_val_overlap)}개")
    print(f"  ❌ Train-Test 중복: {len(train_test_overlap)}개")
    print(f"  ❌ Val-Test 중복: {len(val_test_overlap)}개")

print("\n" + "=" * 60)
print("✅ CT Split 재생성 완료!")
print("=" * 60)
