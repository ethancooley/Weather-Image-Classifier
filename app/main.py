"""
FastAPI web application for weather image classification.

Accepts an uploaded image, runs EfficientNetB3 inference,
and returns the prediction, confidence, and a GradCAM heatmap overlay.

Usage:
    uvicorn app.main:app --reload
"""
# AI Assistance Attribution:
#     Portions of this project were developed with the assistance of Claude (Anthropic).
#     https://claude.ai

import os
import io
from pathlib import Path
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from PIL import Image

from app.utils import load_model, predict_with_gradcam, CLASSES

MODEL_PATH = os.getenv("MODEL_PATH", "models/efficientnet_best.pt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None

BASE_DIR = Path(__file__).parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    global model
    if Path(MODEL_PATH).exists():
        model = load_model(MODEL_PATH, device)
        print(f"Model loaded from {MODEL_PATH}")
    else:
        print(f"Warning: model not found at {MODEL_PATH}")
    yield


app = FastAPI(
    title="Weather Classifier",
    description="EfficientNetB3-based weather image classification with GradCAM explainability.",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


@app.get("/", response_class=HTMLResponse)
async def root(request: Request) -> HTMLResponse:
    """Serve the main UI."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    """
    Accept an uploaded image and return classification results with GradCAM.

    Args:
        file: Uploaded image file.

    Returns:
        Dict with prediction, confidence, top5, and base64 heatmap.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    return predict_with_gradcam(model, image, device)
