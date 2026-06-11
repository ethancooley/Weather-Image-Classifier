"""
Inference and GradCAM utilities for the web application.
"""
# AI Assistance Attribution:
#     Portions of this project were developed with the assistance of Claude (Anthropic).
#     https://claude.ai


from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import matplotlib
import matplotlib.cm as cm
import io
import base64


CLASSES = [
    "dew", "fogsmog", "frost", "glaze", "hail",
    "lightning", "rain", "rainbow", "rime", "sandstorm", "snow"
]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def load_model(model_path: str, device: torch.device) -> nn.Module:
    """Load EfficientNetB3 from a checkpoint path."""
    model = models.efficientnet_b3(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, len(CLASSES)),
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model.to(device).eval()


def predict_with_gradcam(
    model: nn.Module,
    image: Image.Image,
    device: torch.device,
) -> dict:
    """
    Run inference and generate a GradCAM heatmap for a PIL image.

    Args:
        model: Loaded EfficientNetB3.
        image: Input PIL image.
        device: CUDA or CPU.

    Returns:
        Dict with prediction, confidence, top5, and base64-encoded heatmap image.
    """
    activations, gradients = {}, {}

    def forward_hook(module, input, output):
        activations["value"] = output.detach()

    def backward_hook(module, grad_in, grad_out):
        gradients["value"] = grad_out[0].detach()

    handle_f = model.features[-1].register_forward_hook(forward_hook)
    handle_b = model.features[-1].register_backward_hook(backward_hook)

    tensor = TRANSFORM(image.convert("RGB")).unsqueeze(0).to(device)
    tensor.requires_grad_(True)

    output = model(tensor)
    probs = torch.softmax(output, dim=1)[0].detach().cpu().numpy()
    pred_idx = int(np.argmax(probs))

    model.zero_grad()
    output[0, pred_idx].backward()

    # Compute GradCAM
    weights = gradients["value"].mean(dim=(2, 3), keepdim=True)
    cam = (weights * activations["value"]).sum(dim=1).squeeze().cpu().numpy()
    cam = np.maximum(cam, 0)
    if cam.max() > 0:
        cam = cam / cam.max()

    handle_f.remove()
    handle_b.remove()

    # Overlay heatmap on original image
    orig = image.convert("RGB").resize((224, 224))
    heatmap = np.array(Image.fromarray(np.uint8(cam * 255)).resize((224, 224))) / 255.0
    colored = matplotlib.colormaps["jet"](heatmap)[:, :, :3]
    overlay = Image.blend(orig, Image.fromarray(np.uint8(colored * 255)), alpha=0.5)

    buf = io.BytesIO()
    overlay.save(buf, format="PNG")
    heatmap_b64 = base64.b64encode(buf.getvalue()).decode()

    top5_idx = np.argsort(probs)[::-1][:5]
    top5 = [{"class": CLASSES[i], "confidence": round(float(probs[i]), 4)} for i in top5_idx]

    return {
        "prediction": CLASSES[pred_idx],
        "confidence": round(float(probs[pred_idx]), 4),
        "top5": top5,
        "heatmap_b64": heatmap_b64,
    }
