"""
Setup script for the weather classifier project.
Downloads and verifies data, creates directory structure.

Usage:
    python setup.py
"""
# AI Assistance Attribution:
#     Portions of this project were developed with the assistance of Claude (Anthropic).
#     https://claude.ai


import os
from pathlib import Path


REQUIRED_DIRS = [
    "data/raw",
    "data/processed",
    "data/outputs",
    "models",
    "notebooks",
]

CLASSES = [
    "dew", "fogsmog", "frost", "glaze", "hail",
    "lightning", "rain", "rainbow", "rime", "sandstorm", "snow"
]


def create_dirs() -> None:
    """Create all required project directories."""
    for d in REQUIRED_DIRS:
        Path(d).mkdir(parents=True, exist_ok=True)
        print(f"  ✓ {d}")


def check_data() -> None:
    """Check whether raw data has been placed in data/raw/."""
    raw = Path("data/raw")
    found = [d.name for d in raw.iterdir() if d.is_dir()] if raw.exists() else []
    missing = [c for c in CLASSES if c not in found]

    if not missing:
        total = sum(len(list((raw / c).iterdir())) for c in CLASSES)
        print(f"  ✓ Dataset found: {total} images across {len(CLASSES)} classes")
    else:
        print(f"  ⚠ Data not found in data/raw/")
        print(f"    Download from: https://www.kaggle.com/datasets/jehanbhathena/weather-dataset")
        print(f"    Extract into data/raw/ so that data/raw/dew/, data/raw/rain/, etc. exist")


def main() -> None:
    print("Setting up weather-classifier project...\n")
    print("Creating directories:")
    create_dirs()
    print("\nChecking dataset:")
    check_data()
    print("\nDone. Next step: python scripts/build_features.py")


if __name__ == "__main__":
    main()
