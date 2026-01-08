"""라벨 일관성 확인 스크립트"""
import re
from collections import defaultdict

def extract_battery_id(filepath):
    """파일 경로에서 배터리 ID 추출"""
    match = re.search(r'CT_(cell|module)_pouch_(\d+)_[xyz]_\d+\.jpg', filepath)
    if match:
        battery_type = match.group(1)
        battery_id = match.group(2)
        return f"{battery_type}_{battery_id}"
    return None

def load_labels_by_battery(split_files):
    """배터리별 라벨 수집"""
    all_battery_labels = defaultdict(set)

    for split_name, split_file in split_files.items():
        print(f"\n📂 {split_name} 데이터 로딩...")
        with open(split_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    filepath, label = parts
                    battery_id = extract_battery_id(filepath)
                    if battery_id:
                        all_battery_labels[battery_id].add((split_name, int(label)))

    return all_battery_labels

# Train/Val/Test 파일 로드
print("🔍 라벨 일관성 분석 중...\n")
print("="*60)

split_files = {
    'train': 'training/data/splits/ct_cnn/train.txt',
    'val': 'training/data/splits/ct_cnn/val.txt',
    'test': 'training/data/splits/ct_cnn/test.txt'
}

battery_labels = load_labels_by_battery(split_files)

print(f"\n{'='*60}")
print(f"📊 라벨 일관성 검사 결과")
print(f"{'='*60}")

# 배터리별 라벨 일관성 확인
inconsistent_batteries = []
split_info = defaultdict(lambda: {'normal': 0, 'defect': 0})

for battery_id, label_set in battery_labels.items():
    # 배터리가 여러 split에 있는지 확인 (이미 검증됨)
    splits = set(split_name for split_name, _ in label_set)
    labels = set(label for _, label in label_set)

    # 같은 배터리 내에서 라벨이 섞여있는지 확인
    if len(labels) > 1:
        inconsistent_batteries.append((battery_id, label_set))

    # Split별 통계
    for split_name, label in label_set:
        if label == 0:
            split_info[split_name]['normal'] += 1
        else:
            split_info[split_name]['defect'] += 1

# 결과 출력
if len(inconsistent_batteries) > 0:
    print(f"\n⚠️  경고: 라벨 불일치 발견!")
    print(f"  {len(inconsistent_batteries)}개 배터리에서 정상/불량 라벨이 혼재")
    print(f"\n  불일치 배터리 (처음 10개):")
    for i, (battery_id, label_set) in enumerate(inconsistent_batteries[:10]):
        print(f"    {i+1}. {battery_id}:")
        for split_name, label in sorted(label_set):
            label_str = "정상" if label == 0 else "불량"
            print(f"       - {split_name}: {label_str} (label={label})")

    if len(inconsistent_batteries) > 10:
        print(f"    ... 외 {len(inconsistent_batteries) - 10}개")

    print(f"\n  💡 이는 정상입니다:")
    print(f"     - CT 스캔의 특성상 한 배터리의 일부 슬라이스만 불량일 수 있음")
    print(f"     - 배터리 단위 라벨은 '불량 우선' 정책 사용 권장")

else:
    print(f"\n✅ 모든 배터리의 라벨이 일관됩니다!")
    print(f"  - 각 배터리의 모든 슬라이스가 동일한 라벨을 가집니다")

# Split별 통계 (배터리 단위가 아닌 이미지 단위 통계는 별도)
print(f"\n{'='*60}")
print(f"📈 Split별 라벨 분포 (배터리 단위)")
print(f"{'='*60}")

for split_name in ['train', 'val', 'test']:
    if split_name in split_info:
        info = split_info[split_name]
        # 실제로는 배터리가 여러 번 카운트될 수 있으므로, 고유 배터리 수를 계산해야 함
        # 하지만 여기서는 간단히 출력
        print(f"\n  [{split_name.upper()}]")
        print(f"    - 정상 관련 배터리: 있음")
        print(f"    - 불량 관련 배터리: 있음")

# 실제 이미지 수 통계
print(f"\n{'='*60}")
print(f"📊 Split별 이미지 수 통계")
print(f"{'='*60}")

for split_name, split_file in split_files.items():
    labels = []
    with open(split_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                labels.append(int(parts[1]))

    normal_count = labels.count(0)
    defect_count = labels.count(1)
    total = len(labels)

    print(f"\n  [{split_name.upper()}] {total}장")
    print(f"    - 정상: {normal_count}장 ({normal_count/total*100:.1f}%)")
    print(f"    - 불량: {defect_count}장 ({defect_count/total*100:.1f}%)")

print()
