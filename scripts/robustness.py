"""
Robustness experiment: evaluate EfficientNetB3 under systematic image corruptions.

Tests 5 corruption types at 5 severity levels and reports accuracy degradation.
This is the novel contribution over prior work which only evaluates on clean images.

Usage:
    python scripts/robustness.py --model_path models/efficientnet_best.pt --data_dir data/processed
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
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
from tqdm import tqdm


CLASSES = [
    "dew", "fogsmog", "frost", "glaze", "hail",
    "lightning", "rain", "rainbow", "rime", "sandstorm", "snow"
]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# Corruption functions
# ---------------------------------------------------------------------------

def apply_gaussian_blur(img: Image.Image, severity: int) -> Image.Image:
    """Apply Gaussian blur at increasing severity levels (radius 1-5)."""
    radius = severity
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def apply_gaussian_noise(img: Image.Image, severity: int) -> Image.Image:
    """Add Gaussian noise at increasing severity levels (std 10-50)."""
    std = severity * 10
    arr = np.array(img, dtype=np.float32)
    noise = np.random.normal(0, std, arr.shape)
    noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)


def apply_jpeg_compression(img: Image.Image, severity: int) -> Image.Image:
    """Apply JPEG compression at decreasing quality (80 down to 10)."""
    import io
    quality = max(10, 90 - severity * 20)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).copy()


def apply_brightness(img: Image.Image, severity: int) -> Image.Image:
    """Reduce brightness at increasing severity levels."""
    from PIL import ImageEnhance
    factor = max(0.1, 1.0 - severity * 0.18)
    return ImageEnhance.Brightness(img).enhance(factor)


def apply_contrast(img: Image.Image, severity: int) -> Image.Image:
    """Reduce contrast at increasing severity levels."""
    from PIL import ImageEnhance
    factor = max(0.1, 1.0 - severity * 0.18)
    return ImageEnhance.Contrast(img).enhance(factor)


CORRUPTIONS = {
    "gaussian_blur": apply_gaussian_blur,
    "gaussian_noise": apply_gaussian_noise,
    "jpeg_compression": apply_jpeg_compression,
    "brightness": apply_brightness,
    "contrast": apply_contrast,
}
SEVERITIES = [1, 2, 3, 4, 5]


# ---------------------------------------------------------------------------
# Corrupted dataset wrapper
# ---------------------------------------------------------------------------

class CorruptedDataset(Dataset):
    """Wraps an ImageFolder dataset and applies a corruption at runtime."""

    def __init__(
        self,
        base_dataset: datasets.ImageFolder,
        corruption_fn,
        severity: int,
        normalize: transforms.Normalize,
    ) -> None:
        self.base = base_dataset
        self.corruption_fn = corruption_fn
        self.severity = severity
        self.to_tensor = transforms.ToTensor()
        self.normalize = normalize

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self.base.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.corruption_fn(img, self.severity)
        tensor = self.to_tensor(img)
        tensor = self.normalize(tensor)
        return tensor, label


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def load_model(model_path: Path, num_classes: int, device: torch.device) -> nn.Module:
    """Load a saved EfficientNetB3 checkpoint."""
    model = models.efficientnet_b3(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes),
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()
    return model


def evaluate_loader(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Return accuracy on a DataLoader."""
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total


def run_robustness_experiment(
    model: nn.Module,
    data_dir: Path,
    device: torch.device,
    batch_size: int = 32,
) -> dict:
    """
    Evaluate model accuracy across all corruption types and severity levels.

    Args:
        model: Trained EfficientNetB3.
        data_dir: Path to processed data (uses test split).
        device: CUDA or CPU.
        batch_size: DataLoader batch size.

    Returns:
        Nested dict: {corruption_name: {severity: accuracy}}.
    """
    normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    base_dataset = datasets.ImageFolder(root=str(data_dir / "test"))

    # Clean baseline
    clean_loader = DataLoader(
        datasets.ImageFolder(
            root=str(data_dir / "test"),
            transform=transforms.Compose([transforms.ToTensor(), normalize]),
        ),
        batch_size=batch_size, shuffle=False, num_workers=2,
    )
    clean_acc = evaluate_loader(model, clean_loader, device)
    print(f"  Clean accuracy: {clean_acc:.4f}")

    results = {"clean": round(clean_acc, 4)}

    for corruption_name, corruption_fn in CORRUPTIONS.items():
        results[corruption_name] = {}
        print(f"\n  Corruption: {corruption_name}")
        for severity in SEVERITIES:
            dataset = CorruptedDataset(base_dataset, corruption_fn, severity, normalize)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
            acc = evaluate_loader(model, loader, device)
            results[corruption_name][severity] = round(acc, 4)
            print(f"    Severity {severity}: {acc:.4f}")

    return results


def plot_results(results: dict, output_path: Path) -> None:
    """
    Plot accuracy vs. severity for all corruption types.

    Args:
        results: Output from run_robustness_experiment.
        output_path: Path to save the figure.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    clean_acc = results["clean"]
    ax.axhline(y=clean_acc, color="black", linestyle="--", label=f"Clean ({clean_acc:.3f})")

    for corruption_name in CORRUPTIONS:
        if corruption_name not in results:
            continue
        severities = list(results[corruption_name].keys())
        accs = [results[corruption_name][s] for s in severities]
        ax.plot(severities, accs, marker="o", label=corruption_name.replace("_", " ").title())

    ax.set_xlabel("Severity Level")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Model Robustness Under Image Corruptions")
    ax.legend()
    ax.set_xticks(SEVERITIES)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"\n  Plot saved to {output_path}")



def summarize_results(results: dict) -> None:
    """
    Print a human-readable interpretation of robustness experiment results.

    Reports clean accuracy, per-corruption accuracy drop at max severity,
    identifies the most damaging corruption, and flags severity levels where
    accuracy falls below meaningful thresholds.

    Args:
        results: Output from run_robustness_experiment.
    """
    clean_acc = results["clean"]
    print("" + "=" * 50)
    print("  ROBUSTNESS EXPERIMENT SUMMARY")
    print("=" * 50)
    print(f"  Clean test accuracy : {clean_acc:.4f}")
    print()

    drops = {}
    for corruption in CORRUPTIONS:
        if corruption not in results:
            continue
        max_severity_acc = results[corruption][max(results[corruption])]
        drop = clean_acc - max_severity_acc
        drops[corruption] = drop

    # Per-corruption summary
    print("  Accuracy drop at max severity (severity=5):")
    for corruption, drop in sorted(drops.items(), key=lambda x: -x[1]):
        label = corruption.replace("_", " ").title()
        final_acc = clean_acc - drop
        print(f"    {label:<25} {final_acc:.4f}  (drop: -{drop:.4f})")

    # Most damaging corruption
    worst = max(drops, key=drops.get)
    print(f"Most damaging corruption : {worst.replace(chr(95), chr(32)).title()}")
    print(f"  Accuracy drop            : -{drops[worst]:.4f}")

    # Threshold analysis — at what severity does each corruption push accuracy below 50%?
    print("Severity at which accuracy drops below 50%:")
    any_below = False
    for corruption in CORRUPTIONS:
        if corruption not in results:
            continue
        for severity in sorted(results[corruption]):
            if results[corruption][severity] < 0.50:
                print(f"    {corruption.replace(chr(95), chr(32)).title():<25} severity {severity}")
                any_below = True
                break
    if not any_below:
        print("    No corruption drove accuracy below 50% — model is relatively robust.")

    print()

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run robustness experiment.")
    parser.add_argument("--model_path", default="models/efficientnet_best.pt")
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--output_dir", default="data/outputs")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running robustness experiment on {device}...\n")
    model = load_model(Path(args.model_path), len(CLASSES), device)
    results = run_robustness_experiment(model, Path(args.data_dir), device, args.batch_size)

    with open(output_dir / "robustness_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {output_dir}/robustness_results.json")

    plot_results(results, output_dir / "robustness_plot.png")
    summarize_results(results)


if __name__ == "__main__":
    main()
