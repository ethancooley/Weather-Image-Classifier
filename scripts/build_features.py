"""
Preprocessing pipeline for the Weather Image Recognition dataset.

Splits raw data into train/val/test sets, resizes images to 224x224,
applies normalization, and saves processed splits to disk.

Usage:
    python scripts/build_features.py
    python scripts/build_features.py --data_dir data/raw --output_dir data/processed --val_size 0.15 --test_size 0.15
"""
# AI Assistance Attribution:
#     Portions of this project were developed with the assistance of Claude (Anthropic).
#     https://claude.ai


import argparse
import shutil
import random
from pathlib import Path
from PIL import Image
from tqdm import tqdm


# ImageNet normalization constants (used by pretrained EfficientNet)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_SIZE = (224, 224)

CLASSES = [
    "dew", "fogsmog", "frost", "glaze", "hail",
    "lightning", "rain", "rainbow", "rime", "sandstorm", "snow"
]


def get_splits(
    files: list[Path],
    val_size: float,
    test_size: float,
    seed: int = 42,
) -> tuple[list[Path], list[Path], list[Path]]:
    """
    Split a list of file paths into train, val, and test sets.

    Args:
        files: List of image file paths.
        val_size: Fraction of data for validation.
        test_size: Fraction of data for test.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train, val, test) file lists.
    """
    random.seed(seed)
    shuffled = files[:]
    random.shuffle(shuffled)
    n = len(shuffled)
    n_test = int(n * test_size)
    n_val = int(n * val_size)
    test = shuffled[:n_test]
    val = shuffled[n_test:n_test + n_val]
    train = shuffled[n_test + n_val:]
    return train, val, test


def process_image(src: Path, dest: Path) -> bool:
    """
    Center-crop, resize to IMAGE_SIZE, and save as JPEG.

    Args:
        src: Source image path.
        dest: Destination path.

    Returns:
        True if successful, False otherwise.
    """
    try:
        img = Image.open(src).convert("RGB")
        w, h = img.size
        crop = min(w, h)
        img = img.crop(((w - crop) // 2, (h - crop) // 2,
                        (w + crop) // 2, (h + crop) // 2))
        img = img.resize(IMAGE_SIZE, Image.LANCZOS)
        dest.parent.mkdir(parents=True, exist_ok=True)
        img.save(dest, "JPEG", quality=95)
        return True
    except Exception as e:
        print(f"  Warning: failed to process {src}: {e}")
        return False


def build_split(
    files: list[Path],
    class_name: str,
    split_name: str,
    output_dir: Path,
) -> int:
    """
    Process and copy images for one class into a split directory.

    Args:
        files: Source image paths for this class.
        class_name: Class label (used as subdirectory name).
        split_name: One of 'train', 'val', 'test'.
        output_dir: Root output directory.

    Returns:
        Number of successfully processed images.
    """
    count = 0
    for i, src in enumerate(tqdm(files, desc=f"    {split_name}/{class_name}", leave=False)):
        dest = output_dir / split_name / class_name / f"{class_name}_{i:04d}.jpg"
        if process_image(src, dest):
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess weather image dataset.")
    parser.add_argument("--data_dir", default="data/raw", help="Raw data directory")
    parser.add_argument("--output_dir", default="data/processed", help="Output directory")
    parser.add_argument("--val_size", type=float, default=0.15, help="Validation split size")
    parser.add_argument("--test_size", type=float, default=0.15, help="Test split size")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    print(f"Building features: {data_dir} → {output_dir}")
    print(f"Splits: train={1-args.val_size-args.test_size:.0%} val={args.val_size:.0%} test={args.test_size:.0%}\n")

    total_counts = {"train": 0, "val": 0, "test": 0}

    for class_name in CLASSES:
        class_dir = data_dir / class_name
        if not class_dir.exists():
            print(f"  ⚠ Missing class directory: {class_dir}")
            continue

        files = list(class_dir.glob("*"))
        files = [f for f in files if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]

        if not files:
            print(f"  ⚠ No images found in {class_dir}")
            continue

        print(f"  [{class_name}] {len(files)} images")
        train, val, test = get_splits(files, args.val_size, args.test_size)

        for split_name, split_files in [("train", train), ("val", val), ("test", test)]:
            count = build_split(split_files, class_name, split_name, output_dir)
            total_counts[split_name] += count

    print("\nSummary:")
    for split, count in total_counts.items():
        print(f"  {split:<8} {count} images")
    print(f"\nProcessed data saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
