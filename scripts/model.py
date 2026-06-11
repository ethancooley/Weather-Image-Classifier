"""
Training script for all three model types:
  - baseline: majority class classifier (naive baseline)
  - svm: SVM on flattened/PCA-reduced image features (classical ML)
  - efficientnet: fine-tuned EfficientNetB3 (deep learning)

Usage:
    python scripts/model.py --model baseline --data_dir data/processed
    python scripts/model.py --model svm --data_dir data/processed
    python scripts/model.py --model efficientnet --data_dir data/processed --epochs 20
"""
# AI Assistance Attribution:
#     Portions of this project were developed with the assistance of Claude (Anthropic).
#     https://claude.ai


import argparse
import warnings
import json
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm


CLASSES = [
    "dew", "fogsmog", "frost", "glaze", "hail",
    "lightning", "rain", "rainbow", "rime", "sandstorm", "snow"
]
NUM_CLASSES = len(CLASSES)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def get_transforms(split: str) -> transforms.Compose:
    """
    Return image transforms for a given split.

    Training uses augmentation; val/test use only normalization.

    Args:
        split: One of 'train', 'val', 'test'.

    Returns:
        torchvision Compose transform.
    """
    if split == "train":
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_dataloaders(
    data_dir: Path,
    batch_size: int = 32,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train, val, and test DataLoaders from processed data directory.

    Args:
        data_dir: Path to processed data (contains train/, val/, test/ subdirs).
        batch_size: Batch size for all loaders.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    loaders = {}
    for split in ("train", "val", "test"):
        dataset = datasets.ImageFolder(
            root=str(data_dir / split),
            transform=get_transforms(split),
        )
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=2,
            pin_memory=torch.cuda.is_available(),
        )
    return loaders["train"], loaders["val"], loaders["test"]


# ---------------------------------------------------------------------------
# Naive baseline
# ---------------------------------------------------------------------------

def train_baseline(data_dir: Path, output_dir: Path) -> dict:
    """
    Train a majority-class naive baseline classifier.

    Predicts the most frequent class in the training set for every input.

    Args:
        data_dir: Path to processed data directory.
        output_dir: Path to save model and results.

    Returns:
        Dictionary of evaluation metrics.
    """
    print("Training naive baseline (majority class)...")
    dataset = datasets.ImageFolder(root=str(data_dir / "train"))
    class_counts = np.bincount([label for _, label in dataset.samples])
    majority_class = int(np.argmax(class_counts))
    majority_name = CLASSES[majority_class]
    print(f"  Majority class: {majority_name} ({class_counts[majority_class]} samples)")

    test_dataset = datasets.ImageFolder(root=str(data_dir / "test"))
    y_true = [label for _, label in test_dataset.samples]
    y_pred = [majority_class] * len(y_true)
    acc = accuracy_score(y_true, y_pred)

    results = {
        "model": "baseline",
        "majority_class": majority_name,
        "test_accuracy": round(acc, 4),
    }
    print(f"  Test accuracy: {acc:.4f}")

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to {output_dir}/baseline_results.json")
    return results


# ---------------------------------------------------------------------------
# Classical ML: SVM
# ---------------------------------------------------------------------------

def extract_flat_features(data_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load images and flatten pixel values into feature vectors.

    Args:
        data_dir: Path to processed data directory.
        split: One of 'train', 'val', 'test'.

    Returns:
        Tuple of (X, y) numpy arrays.
    """
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.ImageFolder(root=str(data_dir / split), transform=transform)
    X, y = [], []
    for img, label in tqdm(dataset, desc=f"  Loading {split}"):
        # Cast to float64 to prevent overflow in PCA randomized SVD
        X.append(img.numpy().flatten().astype(np.float64))
        y.append(label)
    return np.array(X, dtype=np.float64), np.array(y)


def train_svm(data_dir: Path, output_dir: Path) -> dict:
    """
    Train an SVM classifier on PCA-reduced flattened image features.

    Uses PCA to reduce dimensionality to 100 components before SVM training.

    Args:
        data_dir: Path to processed data directory.
        output_dir: Path to save model and results.

    Returns:
        Dictionary of evaluation metrics.
    """
    print("Training SVM classifier...")
    X_train, y_train = extract_flat_features(data_dir, "train")
    X_test, y_test = extract_flat_features(data_dir, "test")

    print("  Scaling and applying PCA (n=100)...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # n_oversamples=20 stabilizes the randomized SVD on large feature matrices
    pca = PCA(n_components=100, random_state=42, svd_solver="randomized", n_oversamples=20)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)

    print("  Fitting SVM (RBF kernel)...")
    svm = SVC(kernel="rbf", C=10, gamma="scale", random_state=42)
    svm.fit(X_train, y_train)

    y_pred = svm.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=CLASSES)
    print(f"  Test accuracy: {acc:.4f}")
    print(report)

    results = {"model": "svm", "test_accuracy": round(acc, 4)}
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "svm_results.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(output_dir / "svm_model.pkl", "wb") as f:
        pickle.dump({"svm": svm, "scaler": scaler, "pca": pca}, f)

    print(f"  Model saved to {output_dir}/svm_model.pkl")
    return results


# ---------------------------------------------------------------------------
# Deep learning: EfficientNetB3
# ---------------------------------------------------------------------------

def build_efficientnet(num_classes: int, freeze_backbone: bool = False) -> nn.Module:
    """
    Build a fine-tunable EfficientNetB3 with a custom classifier head.

    Args:
        num_classes: Number of output classes.
        freeze_backbone: If True, freeze all layers except the classifier head.

    Returns:
        PyTorch model.
    """
    model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes),
    )
    return model


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """
    Run one training epoch.

    Args:
        model: PyTorch model.
        loader: Training DataLoader.
        optimizer: Optimizer.
        criterion: Loss function.
        device: CUDA or CPU device.

    Returns:
        Average training loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    for images, labels in tqdm(loader, desc="  Training", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """
    Evaluate model accuracy on a DataLoader.

    Args:
        model: PyTorch model.
        loader: DataLoader to evaluate on.
        device: CUDA or CPU device.

    Returns:
        Accuracy as a float between 0 and 1.
    """
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total


def train_efficientnet(
    data_dir: Path,
    output_dir: Path,
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-4,
) -> dict:
    """
    Fine-tune EfficientNetB3 on the weather dataset.

    Uses a two-phase training strategy: first trains only the classifier head,
    then unfreezes the full backbone for end-to-end fine-tuning.

    Args:
        data_dir: Path to processed data directory.
        output_dir: Path to save model checkpoints and results.
        epochs: Total number of training epochs.
        batch_size: Training batch size.
        lr: Learning rate.

    Returns:
        Dictionary of evaluation metrics.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Training EfficientNetB3 on {device}...")

    train_loader, val_loader, test_loader = get_dataloaders(data_dir, batch_size)
    model = build_efficientnet(NUM_CLASSES, freeze_backbone=True).to(device)
    criterion = nn.CrossEntropyLoss()

    # Phase 1: train head only (5 epochs)
    head_epochs = min(5, epochs // 4)
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=lr)
    print(f"\n  Phase 1: training classifier head ({head_epochs} epochs)")
    for epoch in range(head_epochs):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_acc = evaluate(model, val_loader, device)
        print(f"  Epoch {epoch+1}/{head_epochs} — loss: {loss:.4f} — val_acc: {val_acc:.4f}")

    # Phase 2: unfreeze all and fine-tune
    for param in model.parameters():
        param.requires_grad = True
    optimizer = torch.optim.Adam(model.parameters(), lr=lr / 10)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - head_epochs)

    best_val_acc = 0.0
    print(f"\n  Phase 2: full fine-tuning ({epochs - head_epochs} epochs)")
    for epoch in range(epochs - head_epochs):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_acc = evaluate(model, val_loader, device)
        scheduler.step()
        print(f"  Epoch {epoch+1}/{epochs-head_epochs} — loss: {loss:.4f} — val_acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            output_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), output_dir / "efficientnet_best.pt")

    # Final test evaluation
    model.load_state_dict(torch.load(output_dir / "efficientnet_best.pt", map_location=device))
    test_acc = evaluate(model, test_loader, device)
    print(f"\n  Best val accuracy : {best_val_acc:.4f}")
    print(f"  Test accuracy     : {test_acc:.4f}")

    results = {
        "model": "efficientnet_b3",
        "best_val_accuracy": round(best_val_acc, 4),
        "test_accuracy": round(test_acc, 4),
        "epochs": epochs,
    }
    with open(output_dir / "efficientnet_results.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train weather classifier models.")
    parser.add_argument(
        "--model", required=True, choices=["baseline", "svm", "efficientnet"],
        help="Which model to train"
    )
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--output_dir", default="models")
    parser.add_argument("--epochs", type=int, default=20, help="Epochs (efficientnet only)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    if args.model == "baseline":
        train_baseline(data_dir, output_dir)
    elif args.model == "svm":
        train_svm(data_dir, output_dir)
    elif args.model == "efficientnet":
        train_efficientnet(data_dir, output_dir, args.epochs, args.batch_size, args.lr)


if __name__ == "__main__":
    main()
