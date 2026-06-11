"""
GradCAM visualization script.

Generates GradCAM heatmap overlays for sample images from each class,
showing which image regions the model uses to make predictions.

Usage:
    python scripts/gradcam.py --model_path models/efficientnet_best.pt --data_dir data/processed
"""
# AI Assistance Attribution:
#     Portions of this project were developed with the assistance of Claude (Anthropic).
#     https://claude.ai


import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm


CLASSES = [
    "dew", "fogsmog", "frost", "glaze", "hail",
    "lightning", "rain", "rainbow", "rime", "sandstorm", "snow"
]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# GradCAM implementation
# ---------------------------------------------------------------------------

class GradCAM:
    """
    GradCAM: Gradient-weighted Class Activation Mapping.

    Registers forward and backward hooks on a target layer to compute
    class-discriminative localization maps.

    Reference: Selvaraju et al. (2017), https://arxiv.org/abs/1610.02391
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.activations = None
        self.gradients = None
        self._register_hooks(target_layer)

    def _register_hooks(self, layer: nn.Module) -> None:
        """Register forward and backward hooks on the target layer."""
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        layer.register_forward_hook(forward_hook)
        layer.register_full_backward_hook(backward_hook)

    def generate(
        self,
        input_tensor: torch.Tensor,
        class_idx: int | None = None,
    ) -> np.ndarray:
        """
        Generate a GradCAM heatmap for an input image.

        Args:
            input_tensor: Preprocessed image tensor of shape (1, C, H, W).
            class_idx: Target class index. If None, uses the predicted class.

        Returns:
            Normalized heatmap as a (H, W) numpy array in [0, 1].
        """
        self.model.eval()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        output[0, class_idx].backward()

        # Global average pool gradients over spatial dimensions
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam).squeeze().cpu().numpy()

        # Normalize to [0, 1]
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam


def overlay_heatmap(
    original_img: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.5,
) -> Image.Image:
    """
    Overlay a GradCAM heatmap on the original image.

    Args:
        original_img: Original PIL image.
        heatmap: GradCAM heatmap array in [0, 1].
        alpha: Blend weight for the heatmap overlay.

    Returns:
        Blended PIL image.
    """
    heatmap_resized = np.array(
        Image.fromarray(np.uint8(heatmap * 255)).resize(original_img.size, Image.LANCZOS)
    ) / 255.0
    colormap = matplotlib.colormaps["jet"]
    heatmap_colored = colormap(heatmap_resized)[:, :, :3]
    heatmap_pil = Image.fromarray(np.uint8(heatmap_colored * 255))
    return Image.blend(original_img, heatmap_pil, alpha=alpha)


# ---------------------------------------------------------------------------
# Visualization
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
    return model.to(device)


def generate_class_grid(
    model: nn.Module,
    data_dir: Path,
    output_dir: Path,
    device: torch.device,
    n_samples: int = 2,
) -> None:
    """
    Generate GradCAM overlays for sample images from each class.

    Saves a grid image showing original vs. GradCAM overlay for each class.

    Args:
        model: Trained EfficientNetB3.
        data_dir: Path to processed data (uses test split).
        output_dir: Path to save output visualizations.
        device: CUDA or CPU.
        n_samples: Number of sample images per class.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    dataset = datasets.ImageFolder(root=str(data_dir / "test"), transform=transform)
    raw_dataset = datasets.ImageFolder(root=str(data_dir / "test"))

    # Target the last convolutional block of EfficientNetB3
    target_layer = model.features[-1]
    gradcam = GradCAM(model, target_layer)

    # Group sample indices by class
    class_samples: dict[int, list[int]] = {i: [] for i in range(len(CLASSES))}
    for idx, (_, label) in enumerate(dataset.samples):
        if len(class_samples[label]) < n_samples:
            class_samples[label].append(idx)

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        len(CLASSES), n_samples * 2,
        figsize=(n_samples * 6, len(CLASSES) * 3)
    )

    for row, class_idx in enumerate(range(len(CLASSES))):
        for col, sample_idx in enumerate(class_samples[class_idx]):
            tensor, _ = dataset[sample_idx]
            input_tensor = tensor.unsqueeze(0).to(device)

            # Get prediction and heatmap
            with torch.no_grad():
                logits = model(input_tensor)
            pred_idx = logits.argmax(dim=1).item()
            pred_name = CLASSES[pred_idx]

            # Re-run with grad for GradCAM
            input_tensor.requires_grad_(True)
            heatmap = gradcam.generate(input_tensor, class_idx=pred_idx)

            # Original image
            orig_img, _ = raw_dataset[sample_idx]

            # Plot
            ax_orig = axes[row][col * 2]
            ax_cam = axes[row][col * 2 + 1]

            ax_orig.imshow(orig_img)
            ax_orig.set_title(f"True: {CLASSES[class_idx]}", fontsize=8)
            ax_orig.axis("off")

            overlay = overlay_heatmap(orig_img, heatmap)
            ax_cam.imshow(overlay)
            ax_cam.set_title(f"Pred: {pred_name}", fontsize=8,
                             color="green" if pred_idx == class_idx else "red")
            ax_cam.axis("off")

    plt.suptitle("GradCAM Visualizations — Weather Classifier", fontsize=14)
    plt.tight_layout()
    save_path = output_dir / "gradcam_grid.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"GradCAM grid saved to {save_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate GradCAM visualizations.")
    parser.add_argument("--model_path", default="models/efficientnet_best.pt")
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--output_dir", default="data/outputs")
    parser.add_argument("--n_samples", type=int, default=2)
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Generating GradCAM visualizations on {device}...")

    model = load_model(Path(args.model_path), len(CLASSES), device)
    generate_class_grid(
        model, Path(args.data_dir), Path(args.output_dir), device, args.n_samples
    )


if __name__ == "__main__":
    main()
