"""
이미지 전처리 스크립트
- 원본 이미지를 1024x1024로 리사이즈
- D 드라이브에 PNG로 저장
- 새로운 split 파일 생성
"""

import os
import argparse
from pathlib import Path
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from typing import List, Tuple, Set


# 설정
DEFAULT_IMAGE_SIZE = 1024
DEFAULT_OUTPUT_DIR = "/mnt/d/battery-preprocessed"
DEFAULT_DATA_ROOT = "/home/ubuntu/projects/battery-inspection"
SPLITS_DIR = "training/data/splits"


def load_split_file(split_path: str) -> List[Tuple[str, str]]:
    """Split 파일 로드 (이미지 경로, 라벨) 리스트 반환"""
    items = []
    with open(split_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) == 2:
                items.append((parts[0], parts[1]))
    return items


def get_all_image_paths(data_root: str, modalities: List[str] = ['ct', 'rgb']) -> Set[str]:
    """모든 split 파일에서 유니크한 이미지 경로 수집"""
    all_paths = set()
    splits_root = Path(data_root) / SPLITS_DIR

    for modality in modalities:
        modality_dir = splits_root / modality
        if not modality_dir.exists():
            continue

        # 모든 txt 파일 찾기
        for split_file in modality_dir.rglob("*.txt"):
            items = load_split_file(str(split_file))
            for img_path, _ in items:
                all_paths.add(img_path)

    return all_paths


def preprocess_image(
    src_path: str,
    dst_path: str,
    image_size: int = 1024,
    output_format: str = 'PNG'
) -> bool:
    """
    단일 이미지 전처리

    Args:
        src_path: 원본 이미지 경로
        dst_path: 저장할 경로
        image_size: 리사이즈 크기
        output_format: 저장 포맷 (PNG, JPEG)

    Returns:
        성공 여부
    """
    try:
        # 이미지 로드
        img = Image.open(src_path)

        # RGB 변환 (그레이스케일이나 RGBA 처리)
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # 리사이즈 (LANCZOS 고품질 리샘플링)
        img = img.resize((image_size, image_size), Image.LANCZOS)

        # 디렉토리 생성
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)

        # 저장
        if output_format.upper() == 'JPEG':
            img.save(dst_path, 'JPEG', quality=95)
        else:
            img.save(dst_path, 'PNG')

        return True
    except Exception as e:
        print(f"Error processing {src_path}: {e}")
        return False


def convert_path(
    original_path: str,
    output_dir: str,
    output_format: str = 'PNG'
) -> str:
    """원본 경로를 전처리된 경로로 변환"""
    # data/103.배터리... -> battery-preprocessed/103.배터리...
    if original_path.startswith('data/'):
        relative = original_path[5:]  # 'data/' 제거
    else:
        relative = original_path

    # 확장자 변경
    base, _ = os.path.splitext(relative)
    ext = '.png' if output_format.upper() == 'PNG' else '.jpg'

    return os.path.join(output_dir, base + ext)


def update_split_files(
    data_root: str,
    output_dir: str,
    output_format: str = 'PNG',
    modalities: List[str] = ['ct', 'rgb']
):
    """Split 파일들의 경로를 전처리된 경로로 업데이트"""
    splits_root = Path(data_root) / SPLITS_DIR

    for modality in modalities:
        modality_dir = splits_root / modality
        if not modality_dir.exists():
            continue

        for split_file in modality_dir.rglob("*.txt"):
            items = load_split_file(str(split_file))

            # 새 split 파일 경로 (preprocessed_ 접두사)
            new_split_file = split_file.parent / f"preprocessed_{split_file.name}"

            with open(new_split_file, 'w', encoding='utf-8') as f:
                for img_path, label in items:
                    new_path = convert_path(img_path, output_dir, output_format)
                    f.write(f"{new_path}\t{label}\n")

            print(f"Created: {new_split_file}")


def main():
    parser = argparse.ArgumentParser(description='배터리 이미지 전처리')
    parser.add_argument('--size', type=int, default=DEFAULT_IMAGE_SIZE,
                        help=f'리사이즈 크기 (기본: {DEFAULT_IMAGE_SIZE})')
    parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f'출력 디렉토리 (기본: {DEFAULT_OUTPUT_DIR})')
    parser.add_argument('--data-root', type=str, default=DEFAULT_DATA_ROOT,
                        help=f'데이터 루트 (기본: {DEFAULT_DATA_ROOT})')
    parser.add_argument('--format', type=str, default='PNG', choices=['PNG', 'JPEG'],
                        help='출력 포맷 (기본: PNG)')
    parser.add_argument('--workers', type=int, default=8,
                        help='병렬 처리 워커 수 (기본: 8)')
    parser.add_argument('--modality', type=str, nargs='+', default=['ct', 'rgb'],
                        help='처리할 modality (기본: ct rgb)')
    parser.add_argument('--skip-existing', action='store_true',
                        help='이미 존재하는 파일 건너뛰기')
    parser.add_argument('--update-splits-only', action='store_true',
                        help='split 파일만 업데이트 (이미지 처리 건너뛰기)')

    args = parser.parse_args()

    print("=" * 60)
    print("배터리 이미지 전처리")
    print("=" * 60)
    print(f"이미지 크기: {args.size}x{args.size}")
    print(f"출력 디렉토리: {args.output}")
    print(f"출력 포맷: {args.format}")
    print(f"Modalities: {args.modality}")
    print("=" * 60)

    # Split 파일만 업데이트
    if args.update_splits_only:
        print("\nSplit 파일 업데이트 중...")
        update_split_files(args.data_root, args.output, args.format, args.modality)
        print("완료!")
        return

    # 모든 이미지 경로 수집
    print("\n이미지 경로 수집 중...")
    all_paths = get_all_image_paths(args.data_root, args.modality)
    print(f"총 {len(all_paths)}개 이미지 발견")

    # 처리할 작업 목록 생성
    tasks = []
    for img_path in all_paths:
        src_path = os.path.join(args.data_root, img_path)
        dst_path = convert_path(img_path, args.output, args.format)

        # 이미 존재하면 건너뛰기
        if args.skip_existing and os.path.exists(dst_path):
            continue

        tasks.append((src_path, dst_path))

    print(f"처리할 이미지: {len(tasks)}개")

    if not tasks:
        print("처리할 이미지가 없습니다.")
        update_split_files(args.data_root, args.output, args.format, args.modality)
        return

    # 병렬 처리
    print(f"\n전처리 시작 (workers: {args.workers})...")
    success_count = 0
    fail_count = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                preprocess_image, src, dst, args.size, args.format
            ): (src, dst) for src, dst in tasks
        }

        with tqdm(total=len(tasks), desc="Processing") as pbar:
            for future in as_completed(futures):
                if future.result():
                    success_count += 1
                else:
                    fail_count += 1
                pbar.update(1)

    print(f"\n완료: {success_count}개 성공, {fail_count}개 실패")

    # Split 파일 업데이트
    print("\nSplit 파일 업데이트 중...")
    update_split_files(args.data_root, args.output, args.format, args.modality)

    # 용량 확인
    print("\n저장된 데이터 용량 확인...")
    os.system(f"du -sh {args.output}")

    print("\n전처리 완료!")


if __name__ == "__main__":
    main()
