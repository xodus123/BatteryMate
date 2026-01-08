"""RGB AE용 정상 데이터 학습 Split 생성"""
import random
from pathlib import Path
from collections import defaultdict

random.seed(42)

# 경로 설정
splits_dir = Path('/home/ubuntu/projects/battery-inspection/training/data/splits/rgb')
output_dir = splits_dir  # 같은 위치에 덮어쓰기

# 기존 split 파일 모두 읽기
all_data = []
for split_file in ['train.txt', 'val.txt', 'test.txt']:
    with open(splits_dir / split_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split('\t')
                if len(parts) == 2:
                    path, label = parts[0], int(parts[1])
                    all_data.append((path, label))

print(f"전체 데이터: {len(all_data)}개")

# 라벨별 분리
normal_data = [(p, l) for p, l in all_data if l == 0]
defect_data = [(p, l) for p, l in all_data if l > 0]

print(f"  Normal (label=0): {len(normal_data)}개")
print(f"  Defect (label>0): {len(defect_data)}개")

# 배터리 ID 추출 (Data Leakage 방지)
def get_battery_id(path):
    """파일명에서 배터리 ID 추출"""
    filename = Path(path).stem
    # RGB_cell_cylindrical_0110_001 -> 0110
    parts = filename.split('_')
    for i, part in enumerate(parts):
        if part.isdigit() and len(part) == 4:
            return part
    return filename

# 배터리 ID별로 그룹화
normal_by_battery = defaultdict(list)
for path, label in normal_data:
    bid = get_battery_id(path)
    normal_by_battery[bid].append((path, label))

defect_by_battery = defaultdict(list)
for path, label in defect_data:
    bid = get_battery_id(path)
    defect_by_battery[bid].append((path, label))

print(f"\n배터리 ID 수:")
print(f"  Normal: {len(normal_by_battery)}개")
print(f"  Defect: {len(defect_by_battery)}개")

# Normal 배터리 ID를 train/val/test로 분할 (70/15/15)
normal_battery_ids = list(normal_by_battery.keys())
random.shuffle(normal_battery_ids)

n_total = len(normal_battery_ids)
n_train = int(n_total * 0.7)
n_val = int(n_total * 0.15)

train_battery_ids = set(normal_battery_ids[:n_train])
val_battery_ids = set(normal_battery_ids[n_train:n_train+n_val])
test_battery_ids = set(normal_battery_ids[n_train+n_val:])

print(f"\nNormal 배터리 분할:")
print(f"  Train: {len(train_battery_ids)}개 배터리")
print(f"  Val: {len(val_battery_ids)}개 배터리")
print(f"  Test: {len(test_battery_ids)}개 배터리")

# Defect 배터리 ID를 val/test로 분할 (50/50)
defect_battery_ids = list(defect_by_battery.keys())
random.shuffle(defect_battery_ids)

n_defect = len(defect_battery_ids)
n_defect_val = n_defect // 2

defect_val_battery_ids = set(defect_battery_ids[:n_defect_val])
defect_test_battery_ids = set(defect_battery_ids[n_defect_val:])

print(f"\nDefect 배터리 분할:")
print(f"  Val: {len(defect_val_battery_ids)}개 배터리")
print(f"  Test: {len(defect_test_battery_ids)}개 배터리")

# Split 생성
train_data = []
val_data = []
test_data = []

# Normal 데이터 분배
for bid in train_battery_ids:
    train_data.extend(normal_by_battery[bid])
for bid in val_battery_ids:
    val_data.extend(normal_by_battery[bid])
for bid in test_battery_ids:
    test_data.extend(normal_by_battery[bid])

# Defect 데이터 분배 (val/test에만)
for bid in defect_val_battery_ids:
    val_data.extend(defect_by_battery[bid])
for bid in defect_test_battery_ids:
    test_data.extend(defect_by_battery[bid])

# 셔플
random.shuffle(train_data)
random.shuffle(val_data)
random.shuffle(test_data)

print(f"\n최종 Split:")
print(f"  Train: {len(train_data)}개 (Normal only)")
print(f"  Val: {len(val_data)}개")
print(f"  Test: {len(test_data)}개")

# 라벨 분포 확인
def count_labels(data):
    counts = defaultdict(int)
    for _, label in data:
        counts[label] += 1
    return dict(counts)

print(f"\n라벨 분포:")
print(f"  Train: {count_labels(train_data)}")
print(f"  Val: {count_labels(val_data)}")
print(f"  Test: {count_labels(test_data)}")

# 백업 후 저장
import shutil
backup_dir = splits_dir / 'backup_defect_training'
backup_dir.mkdir(exist_ok=True)

for f in ['train.txt', 'val.txt', 'test.txt']:
    if (splits_dir / f).exists():
        shutil.copy(splits_dir / f, backup_dir / f)
print(f"\n기존 파일 백업: {backup_dir}")

# 새 split 저장
for name, data in [('train.txt', train_data), ('val.txt', val_data), ('test.txt', test_data)]:
    with open(output_dir / name, 'w') as f:
        for path, label in data:
            f.write(f"{path}\t{label}\n")
    print(f"저장: {output_dir / name}")

print("\n✅ 정상 데이터 학습용 Split 생성 완료!")
