"""
클래스 밸런싱 스크립트 - Class 3 언더샘플링
"""
import random
import argparse
from pathlib import Path
from collections import defaultdict


def balance_split(
    input_file: str,
    output_file: str,
    class_3_limit: int = 100000,
    seed: int = 42
):
    """
    Class 3 (module_porosity) 언더샘플링

    Args:
        input_file: 원본 split 파일
        output_file: 출력 split 파일
        class_3_limit: Class 3 최대 샘플 수
        seed: 랜덤 시드
    """
    random.seed(seed)

    # 클래스별로 샘플 수집
    samples_by_class = defaultdict(list)
    header = None

    with open(input_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            if i == 0 and line.startswith('#'):
                header = line
                continue

            parts = line.split('\t')
            if len(parts) >= 2:
                label = int(parts[1])
                samples_by_class[label].append(line)

    # 원본 분포 출력
    print("=== 원본 분포 ===")
    total_original = 0
    for c in range(5):
        count = len(samples_by_class[c])
        total_original += count
        print(f"  Class {c}: {count:,}")
    print(f"  Total: {total_original:,}")

    # Class 3 언더샘플링
    if len(samples_by_class[3]) > class_3_limit:
        print(f"\n=== Class 3 언더샘플링: {len(samples_by_class[3]):,} → {class_3_limit:,} ===")
        samples_by_class[3] = random.sample(samples_by_class[3], class_3_limit)

    # 밸런싱된 분포 출력
    print("\n=== 밸런싱 후 분포 ===")
    total_balanced = 0
    for c in range(5):
        count = len(samples_by_class[c])
        total_balanced += count
        pct = count / total_balanced * 100 if total_balanced > 0 else 0
        print(f"  Class {c}: {count:,}")

    # 비율 재계산
    print("\n=== 밸런싱 후 비율 ===")
    for c in range(5):
        count = len(samples_by_class[c])
        pct = count / total_balanced * 100
        print(f"  Class {c}: {pct:.1f}%")
    print(f"  Total: {total_balanced:,}")

    # 출력 파일 저장
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_samples = []
    for c in range(5):
        all_samples.extend(samples_by_class[c])

    # 셔플
    random.shuffle(all_samples)

    with open(output_file, 'w', encoding='utf-8') as f:
        if header:
            f.write(header + '\n')
        for sample in all_samples:
            f.write(sample + '\n')

    print(f"\n저장 완료: {output_file}")
    print(f"  샘플 수: {len(all_samples):,}")


def main():
    parser = argparse.ArgumentParser(description='클래스 밸런싱')
    parser.add_argument('--input', type=str,
                        default='training/data/splits/ct/patch/battery_train.txt',
                        help='입력 split 파일')
    parser.add_argument('--output', type=str,
                        default='training/data/splits/ct/patch/battery_train_balanced.txt',
                        help='출력 split 파일')
    parser.add_argument('--class3-limit', type=int, default=100000,
                        help='Class 3 최대 샘플 수')
    parser.add_argument('--seed', type=int, default=42,
                        help='랜덤 시드')

    args = parser.parse_args()

    balance_split(
        args.input,
        args.output,
        args.class3_limit,
        args.seed
    )


if __name__ == "__main__":
    main()
