"""배터리 단위로 Split 재생성 스크립트

기존 문제: 같은 배터리의 x/y/z축 이미지가 Train/Val/Test에 분산 → 데이터 누수
해결: 배터리 ID 단위로 Split하여 같은 배터리는 같은 Split에만 존재하도록 수정
"""

import re
import random
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import argparse


def extract_battery_id(filepath: str) -> str:
    """파일 경로에서 배터리 ID 추출

    예: CT_module_pouch_015_x_001.jpg → module_015
        CT_cell_pouch_123_y_050.jpg → cell_123
    """
    match = re.search(r'CT_(cell|module)_\w+_(\d+)', filepath)
    if match:
        return f"{match.group(1)}_{match.group(2)}"
    return None


def load_split_files(split_dir: Path) -> Dict[str, List[Tuple[str, str]]]:
    """기존 Split 파일들 로드

    Returns:
        {battery_id: [(filepath, label), ...]}
    """
    battery_files = defaultdict(list)

    for split_name in ['train', 'val', 'test']:
        split_file = split_dir / f'battery_{split_name}.txt'
        if not split_file.exists():
            print(f"  ⚠️ {split_file} 없음, 건너뜀")
            continue

        with open(split_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split('\t')
                filepath = parts[0]
                label = parts[1] if len(parts) > 1 else ''

                battery_id = extract_battery_id(filepath)
                if battery_id:
                    battery_files[battery_id].append((filepath, label))

    return battery_files


def split_batteries(
    battery_ids: List[str],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[List[str], List[str], List[str]]:
    """배터리 ID를 Train/Val/Test로 분할

    Args:
        battery_ids: 배터리 ID 리스트
        train_ratio: Train 비율
        val_ratio: Val 비율
        test_ratio: Test 비율
        seed: 랜덤 시드

    Returns:
        (train_ids, val_ids, test_ids)
    """
    random.seed(seed)

    # 셔플
    ids = battery_ids.copy()
    random.shuffle(ids)

    n_total = len(ids)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_ids = ids[:n_train]
    val_ids = ids[n_train:n_train + n_val]
    test_ids = ids[n_train + n_val:]

    return train_ids, val_ids, test_ids


def save_split_files(
    split_dir: Path,
    battery_files: Dict[str, List[Tuple[str, str]]],
    train_ids: List[str],
    val_ids: List[str],
    test_ids: List[str],
    backup: bool = True
):
    """새로운 Split 파일 저장

    Args:
        split_dir: Split 파일 디렉토리
        battery_files: 배터리별 파일 목록
        train_ids: Train 배터리 ID 리스트
        val_ids: Val 배터리 ID 리스트
        test_ids: Test 배터리 ID 리스트
        backup: 기존 파일 백업 여부
    """
    splits = {
        'train': train_ids,
        'val': val_ids,
        'test': test_ids
    }

    for split_name, battery_ids in splits.items():
        output_file = split_dir / f'battery_{split_name}.txt'

        # 백업
        if backup and output_file.exists():
            backup_file = split_dir / f'battery_{split_name}.txt.bak'
            output_file.rename(backup_file)
            print(f"  📦 백업: {backup_file}")

        # 파일 수집
        files = []
        for bid in battery_ids:
            if bid in battery_files:
                files.extend(battery_files[bid])

        # 저장
        with open(output_file, 'w') as f:
            for filepath, label in files:
                f.write(f"{filepath}\t{label}\n")

        print(f"  ✅ {split_name}: {len(battery_ids)} batteries, {len(files)} files")


def verify_no_leakage(split_dir: Path):
    """데이터 누수 검증"""
    battery_in_splits = defaultdict(set)

    for split_name in ['train', 'val', 'test']:
        split_file = split_dir / f'battery_{split_name}.txt'
        if not split_file.exists():
            continue

        with open(split_file, 'r') as f:
            for line in f:
                filepath = line.strip().split('\t')[0]
                battery_id = extract_battery_id(filepath)
                if battery_id:
                    battery_in_splits[battery_id].add(split_name)

    multi_split = sum(1 for b in battery_in_splits.values() if len(b) > 1)

    if multi_split == 0:
        print(f"  ✅ 검증 통과: 데이터 누수 없음")
    else:
        print(f"  ❌ 검증 실패: {multi_split}개 배터리가 여러 split에 존재")

    return multi_split == 0


def process_split_directory(
    split_dir: Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
):
    """단일 Split 디렉토리 처리"""
    print(f"\n{'='*60}")
    print(f"처리 중: {split_dir}")
    print(f"{'='*60}")

    # 1. 기존 파일 로드
    print("\n[1] 기존 Split 파일 로드...")
    battery_files = load_split_files(split_dir)
    print(f"  총 {len(battery_files)} 배터리 발견")

    if not battery_files:
        print("  ⚠️ 파일 없음, 건너뜀")
        return

    # 2. 배터리 단위 분할
    print("\n[2] 배터리 단위 분할...")
    battery_ids = list(battery_files.keys())
    train_ids, val_ids, test_ids = split_batteries(
        battery_ids, train_ratio, val_ratio, test_ratio, seed
    )
    print(f"  Train: {len(train_ids)} batteries ({train_ratio*100:.0f}%)")
    print(f"  Val: {len(val_ids)} batteries ({val_ratio*100:.0f}%)")
    print(f"  Test: {len(test_ids)} batteries ({test_ratio*100:.0f}%)")

    # 3. 새 Split 파일 저장
    print("\n[3] 새 Split 파일 저장...")
    save_split_files(split_dir, battery_files, train_ids, val_ids, test_ids)

    # 4. 검증
    print("\n[4] 데이터 누수 검증...")
    verify_no_leakage(split_dir)

    # 5. 클래스 분포 확인
    print("\n[5] 클래스 분포...")
    for split_name in ['train', 'val', 'test']:
        split_file = split_dir / f'battery_{split_name}.txt'
        if split_file.exists():
            class_counts = defaultdict(int)
            with open(split_file, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) > 1:
                        class_counts[parts[1]] += 1

            total = sum(class_counts.values())
            dist = ', '.join([f"{k}:{v}" for k, v in sorted(class_counts.items())])
            print(f"  {split_name}: {total} files - {dist}")


def main():
    parser = argparse.ArgumentParser(description='배터리 단위로 Split 재생성')
    parser.add_argument('--train-ratio', type=float, default=0.7, help='Train 비율')
    parser.add_argument('--val-ratio', type=float, default=0.15, help='Val 비율')
    parser.add_argument('--test-ratio', type=float, default=0.15, help='Test 비율')
    parser.add_argument('--seed', type=int, default=42, help='랜덤 시드')
    parser.add_argument('--only', type=str, default=None,
                        help='특정 디렉토리만 처리 (preprocessed, cropped, patch)')
    args = parser.parse_args()

    base_dir = Path('training/data/splits/ct')

    # 처리할 디렉토리 목록
    if args.only:
        if args.only == 'preprocessed':
            split_dirs = [base_dir]
        else:
            split_dirs = [base_dir / args.only]
    else:
        split_dirs = [
            base_dir,              # preprocessed (resized)
            base_dir / 'cropped',  # cropped
            base_dir / 'patch',    # patch
        ]

    print("=" * 60)
    print("배터리 단위 Split 재생성")
    print("=" * 60)
    print(f"Train:Val:Test = {args.train_ratio}:{args.val_ratio}:{args.test_ratio}")
    print(f"Seed: {args.seed}")

    for split_dir in split_dirs:
        if split_dir.exists():
            process_split_directory(
                split_dir,
                args.train_ratio,
                args.val_ratio,
                args.test_ratio,
                args.seed
            )
        else:
            print(f"\n⚠️ 디렉토리 없음: {split_dir}")

    print("\n" + "=" * 60)
    print("완료!")
    print("=" * 60)


if __name__ == '__main__':
    main()
