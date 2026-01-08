"""Split 파일에 있는 이미지만 Linux로 복사하는 스크립트"""
import os
import shutil
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# 설정
PROJECT_ROOT = Path("/home/ubuntu/projects/battery-inspection")
SOURCE_BASE = PROJECT_ROOT / "data"  # 현재 심볼릭 링크 (/mnt/d/...)
TARGET_BASE = Path("/home/ubuntu/battery-data")  # 복사 대상 (Linux)

SPLIT_FILES = [
    PROJECT_ROOT / "training/data/splits/ct/train.txt",
    PROJECT_ROOT / "training/data/splits/ct/val.txt",
    PROJECT_ROOT / "training/data/splits/ct/test.txt",
]

# 병렬 복사 워커 수
NUM_WORKERS = 8


def read_split_files():
    """Split 파일에서 이미지 경로 추출"""
    image_paths = set()

    for split_file in SPLIT_FILES:
        if not split_file.exists():
            print(f"⚠️ Split 파일 없음: {split_file}")
            continue

        with open(split_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if parts:
                    # 'data/103.배터리.../image.jpg' -> '103.배터리.../image.jpg'
                    rel_path = parts[0]
                    if rel_path.startswith('data/'):
                        rel_path = rel_path[5:]  # 'data/' 제거
                    image_paths.add(rel_path)

    return list(image_paths)


def copy_file(rel_path):
    """단일 파일 복사"""
    src = SOURCE_BASE / rel_path
    dst = TARGET_BASE / rel_path

    try:
        if dst.exists():
            return "skip"

        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return "ok"
    except Exception as e:
        return f"error: {e}"


def main():
    print("=" * 60)
    print("Split 파일 기반 이미지 복사 스크립트")
    print("=" * 60)
    print(f"원본: {SOURCE_BASE}")
    print(f"대상: {TARGET_BASE}")
    print()

    # 이미지 경로 수집
    print("📂 Split 파일 읽는 중...")
    image_paths = read_split_files()
    print(f"   총 이미지 수: {len(image_paths):,}개")
    print()

    # 대상 디렉토리 생성
    TARGET_BASE.mkdir(parents=True, exist_ok=True)

    # 이미 복사된 파일 확인
    existing = sum(1 for p in image_paths if (TARGET_BASE / p).exists())
    to_copy = len(image_paths) - existing

    print(f"📊 복사 현황:")
    print(f"   이미 복사됨: {existing:,}개")
    print(f"   복사 필요: {to_copy:,}개")
    print()

    if to_copy == 0:
        print("✅ 모든 파일이 이미 복사되어 있습니다!")
        print()
        print("심볼릭 링크 변경 명령어:")
        print(f"  cd {PROJECT_ROOT}")
        print(f"  rm data")
        print(f"  ln -s {TARGET_BASE} data")
        return

    # 용량 추정
    sample_size_mb = 1.6  # 평균 이미지 크기
    estimated_gb = (to_copy * sample_size_mb) / 1024
    print(f"📦 예상 복사 용량: ~{estimated_gb:.1f}GB")
    print()

    # 확인
    confirm = input("복사를 시작할까요? (y/n): ").strip().lower()
    if confirm != 'y':
        print("취소됨.")
        return

    print()
    print("🚀 복사 시작...")

    # 병렬 복사
    copied = 0
    skipped = 0
    errors = 0

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(copy_file, p): p for p in image_paths}

        with tqdm(total=len(image_paths), desc="복사 중", unit="files") as pbar:
            for future in as_completed(futures):
                result = future.result()
                if result == "ok":
                    copied += 1
                elif result == "skip":
                    skipped += 1
                else:
                    errors += 1
                pbar.update(1)

    print()
    print("=" * 60)
    print("✅ 복사 완료!")
    print(f"   복사됨: {copied:,}개")
    print(f"   스킵됨: {skipped:,}개")
    print(f"   에러: {errors}개")
    print()
    print("📌 다음 단계: 심볼릭 링크 변경")
    print(f"   cd {PROJECT_ROOT}")
    print(f"   rm data")
    print(f"   ln -s {TARGET_BASE} data")
    print()
    print("그 후 학습 재시작:")
    print("   python models/ct_cnn/train.py --config cnn_ct_unified")
    print("=" * 60)


if __name__ == "__main__":
    main()
