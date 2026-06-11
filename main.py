"""
Weather Image Classifier — Main Entry Point
============================================
Runs the full pipeline: preprocess → train → evaluate → experiment → serve.

AI Assistance Attribution:
    Portions of this project were developed with the assistance of Claude (Anthropic).
    https://claude.ai

Usage:
    # Run full pipeline
    python main.py --pipeline full

    # Individual steps
    python main.py --pipeline preprocess
    python main.py --pipeline train
    python main.py --pipeline evaluate
    python main.py --pipeline experiment
    python main.py --pipeline serve
"""

import argparse
import subprocess
import sys
from pathlib import Path


STEPS = {
    "preprocess": [
        sys.executable, "scripts/build_features.py",
        "--data_dir", "data/raw",
        "--output_dir", "data/processed",
    ],
    "train_baseline": [
        sys.executable, "scripts/model.py", "--model", "baseline",
        "--data_dir", "data/processed", "--output_dir", "models",
    ],
    "train_svm": [
        sys.executable, "scripts/model.py", "--model", "svm",
        "--data_dir", "data/processed", "--output_dir", "models",
    ],
    "train_efficientnet": [
        sys.executable, "scripts/model.py", "--model", "efficientnet",
        "--data_dir", "data/processed", "--output_dir", "models",
    ],
    "evaluate": [
        sys.executable, "scripts/evaluate.py",
        "--model_path", "models/efficientnet_best.pt",
        "--data_dir", "data/processed",
        "--output_dir", "data/outputs",
    ],
    "gradcam": [
        sys.executable, "scripts/gradcam.py",
        "--model_path", "models/efficientnet_best.pt",
        "--data_dir", "data/processed",
        "--output_dir", "data/outputs",
    ],
    "experiment": [
        sys.executable, "scripts/robustness.py",
        "--model_path", "models/efficientnet_best.pt",
        "--data_dir", "data/processed",
        "--output_dir", "data/outputs",
    ],
    "serve": [
        "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload",
    ],
}


def run_step(name: str, cmd: list[str]) -> None:
    """
    Run a single pipeline step as a subprocess.

    Args:
        name: Human-readable step name for logging.
        cmd: Command list to pass to subprocess.run.

    Raises:
        SystemExit: If the subprocess returns a non-zero exit code.
    """
    print(f"\n{'='*50}")
    print(f"  STEP: {name}")
    print(f"{'='*50}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\n  ✗ Step '{name}' failed with exit code {result.returncode}")
        sys.exit(result.returncode)
    print(f"  ✓ Step '{name}' complete")


def run_pipeline(pipeline: str) -> None:
    """
    Execute one or all pipeline steps.

    Args:
        pipeline: One of 'full', 'preprocess', 'train', 'evaluate', 'experiment', 'serve'.
    """
    if pipeline == "full":
        steps = [
            ("Preprocess data", STEPS["preprocess"]),
            ("Train baseline", STEPS["train_baseline"]),
            ("Train SVM", STEPS["train_svm"]),
            ("Train EfficientNetB3", STEPS["train_efficientnet"]),
            ("Evaluate model", STEPS["evaluate"]),
            ("Generate GradCAM", STEPS["gradcam"]),
            ("Run robustness experiment", STEPS["experiment"]),
        ]
    elif pipeline == "preprocess":
        steps = [("Preprocess data", STEPS["preprocess"])]
    elif pipeline == "train":
        steps = [
            ("Train baseline", STEPS["train_baseline"]),
            ("Train SVM", STEPS["train_svm"]),
            ("Train EfficientNetB3", STEPS["train_efficientnet"]),
        ]
    elif pipeline == "evaluate":
        steps = [
            ("Evaluate model", STEPS["evaluate"]),
            ("Generate GradCAM", STEPS["gradcam"]),
        ]
    elif pipeline == "experiment":
        steps = [("Run robustness experiment", STEPS["experiment"])]
    elif pipeline == "serve":
        steps = [("Serve app", STEPS["serve"])]
    else:
        print(f"Unknown pipeline: {pipeline}")
        sys.exit(1)

    for name, cmd in steps:
        run_step(name, cmd)

    if pipeline != "serve":
        print(f"\n✓ Pipeline '{pipeline}' complete.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Weather Image Classifier — pipeline runner."
    )
    parser.add_argument(
        "--pipeline",
        choices=["full", "preprocess", "train", "evaluate", "experiment", "serve"],
        default="full",
        help="Which pipeline stage to run (default: full)",
    )
    args = parser.parse_args()
    run_pipeline(args.pipeline)


if __name__ == "__main__":
    main()
