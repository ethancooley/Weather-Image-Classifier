"""
Dataset acquisition script for the Weather Image Recognition dataset.

AI Assistance Attribution:
    Portions of this project were developed with the assistance of Claude (Anthropic).
    https://claude.ai

Checks whether the dataset is already present in data/raw/. If not, attempts
to download it via the Kaggle API. Falls back to printing manual instructions
if the Kaggle CLI is not configured.

Usage:
    python scripts/make_dataset.py
    python scripts/make_dataset.py --data_dir data/raw
"""

import argparse
import subprocess
import sys
from pathlib import Path


KAGGLE_DATASET = "jehanbhathena/weather-dataset"
CLASSES = [
    "dew", "fogsmog", "frost", "glaze", "hail",
    "lightning", "rain", "rainbow", "rime", "sandstorm", "snow",
]


def dataset_exists(data_dir: Path) -> bool:
    """
    Check whether the raw dataset is already present.

    Args:
        data_dir: Expected root directory for raw class folders.

    Returns:
        True if all class directories exist and contain images.
    """
    if not data_dir.exists():
        return False
    found = [d.name for d in data_dir.iterdir() if d.is_dir()]
    return all(cls in found for cls in CLASSES)


def download_via_kaggle(data_dir: Path) -> bool:
    """
    Attempt to download the dataset using the Kaggle CLI.

    Requires a configured ~/.kaggle/kaggle.json API token.
    See: https://www.kaggle.com/docs/api

    Args:
        data_dir: Directory to download and extract the dataset into.

    Returns:
        True if download succeeded, False otherwise.
    """
    try:
        result = subprocess.run(
            ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET,
             "--unzip", "-p", str(data_dir)],
            check=True,
        )
        return result.returncode == 0
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def print_manual_instructions(data_dir: Path) -> None:
    """Print manual download instructions if Kaggle CLI is unavailable."""
    print("\n  Manual download instructions:")
    print(f"  1. Go to: https://www.kaggle.com/datasets/{KAGGLE_DATASET}")
    print(f"  2. Click 'Download' and extract the zip")
    print(f"  3. Move the extracted folders into: {data_dir.resolve()}")
    print(f"     Expected structure: {data_dir}/dew/, {data_dir}/rain/, etc.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the weather image dataset.")
    parser.add_argument("--data_dir", default="data/raw", help="Target directory for raw data")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"Checking for dataset in {data_dir.resolve()}...")

    if dataset_exists(data_dir):
        total = sum(
            len(list((data_dir / cls).iterdir()))
            for cls in CLASSES
            if (data_dir / cls).exists()
        )
        print(f"  ✓ Dataset already present ({total} images across {len(CLASSES)} classes)")
        return

    print("  Dataset not found. Attempting Kaggle CLI download...")
    if download_via_kaggle(data_dir):
        print("  ✓ Download complete")
    else:
        print("  ✗ Kaggle CLI not available or not configured.")
        print_manual_instructions(data_dir)
        sys.exit(1)


if __name__ == "__main__":
    main()
