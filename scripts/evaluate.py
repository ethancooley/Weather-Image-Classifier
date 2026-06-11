"""
Evaluation script: generates confusion matrix, per-class metrics,
and error analysis for the trained EfficientNetB3 model.

Usage:
    python scripts/evaluate.py --model_path models/efficientnet_best.pt --data_dir data/processed
"""
# AI Assistance Attribution:
#     Portions of this project were developed with the assistance of Claude (Anthropic).
#     https://claude.ai


import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


CLASSES = [
    "dew", "fogsmog", "frost", "glaze", "hail",
    "lightning", "rain", "rainbow", "rime", "sandstorm", "snow"
]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_model(model_path: Path, num_classes: int, device: torch.device) -> nn.Module:
    """Load a saved EfficientNetB3 checkpoint."""
    model = models.efficientnet_b3(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes),
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model.to(device).eval()


def get_predictions(
    model: nn.Module,
    data_dir: Path,
    device: torch.device,
    batch_size: int = 32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run inference on the test set and collect predictions.

    Args:
        model: Trained model.
        data_dir: Path to processed data directory.
        device: CUDA or CPU.
        batch_size: DataLoader batch size.

    Returns:
        Tuple of (y_true, y_pred, y_probs) numpy arrays.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    dataset = datasets.ImageFolder(root=str(data_dir / "test"), transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating"):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            all_probs.extend(probs)

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, output_path: Path) -> None:
    """Save a normalized confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt=".2f", xticklabels=CLASSES, yticklabels=CLASSES,
                cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Normalized Confusion Matrix — EfficientNetB3")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Confusion matrix saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate weather classifier.")
    parser.add_argument("--model_path", default="models/efficientnet_best.pt")
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--output_dir", default="data/outputs")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(Path(args.model_path), len(CLASSES), device)
    y_true, y_pred, y_probs = get_predictions(model, Path(args.data_dir), device)

    report = classification_report(y_true, y_pred, target_names=CLASSES, output_dict=True)
    print(classification_report(y_true, y_pred, target_names=CLASSES))

    with open(output_dir / "classification_report.json", "w") as f:
        json.dump(report, f, indent=2)

    plot_confusion_matrix(y_true, y_pred, output_dir / "confusion_matrix.png")


if __name__ == "__main__":
    main()
