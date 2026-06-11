# Weather Image Classifier

> EfficientNetB3-based weather classification with GradCAM explainability and systematic robustness evaluation.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)

---

## Overview

This project classifies weather conditions from images into 11 classes using the [Weather Image Recognition dataset](https://www.kaggle.com/datasets/jehanbhathena/weather-dataset) from Kaggle.

**Classes:** dew · fog/smog · frost · glaze · hail · lightning · rain · rainbow · rime · sandstorm · snow

### Novel contributions over prior work

Prior work on this dataset trains small CNNs from scratch and reports top-1 accuracy on clean images only. This project adds:

1. **Robustness evaluation** — systematic accuracy measurement under 5 corruption types (Gaussian blur, Gaussian noise, JPEG compression, brightness reduction, contrast reduction) at 5 severity levels. No prior work on this dataset does this.
2. **GradCAM explainability** — visual heatmaps showing which image regions drive each prediction, integrated directly into the web application.
3. **Production-quality app** — FastAPI backend + custom UI with confidence bars and heatmap overlay, not a basic Streamlit demo.

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/weather-classifier.git
cd weather-classifier
pip install -r requirements.txt
```

### 2. Download dataset

```bash
python scripts/make_dataset.py
```

This attempts to use the Kaggle CLI. If you haven't set it up:
- Go to https://www.kaggle.com/datasets/jehanbhathena/weather-dataset
- Click Download and unzip into `data/raw/`
- Expected: `data/raw/dew/`, `data/raw/rain/`, etc.

### 3. Run full pipeline

```bash
python main.py --pipeline full
```

Or run individual stages:

```bash
python main.py --pipeline preprocess   # build train/val/test splits
python main.py --pipeline train        # train all 3 models
python main.py --pipeline evaluate     # confusion matrix + metrics
python main.py --pipeline experiment   # robustness experiment
python main.py --pipeline serve        # launch web app
```

### 4. Launch app only

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Open http://localhost:8000

---

## Project Structure

```
weather-classifier/
├── README.md                   ← this file
├── requirements.txt            ← dependencies
├── setup.py                    ← verify environment and data
├── main.py                     ← pipeline runner (entry point)
│
├── scripts/
│   ├── make_dataset.py         ← download raw data via Kaggle CLI
│   ├── build_features.py       ← preprocess into train/val/test splits
│   ├── model.py                ← train baseline, SVM, and EfficientNetB3
│   ├── evaluate.py             ← confusion matrix + classification report
│   ├── gradcam.py              ← GradCAM heatmap grid per class
│   └── robustness.py           ← robustness experiment (novel contribution)
│
├── app/
│   ├── main.py                 ← FastAPI app + UI
│   └── utils.py                ← inference + GradCAM helpers
│
├── data/
│   ├── raw/                    ← original Kaggle images (git-ignored)
│   ├── processed/              ← train/val/test splits (git-ignored)
│   └── outputs/                ← confusion matrices, heatmaps, plots (git-ignored)
│
├── models/                     ← saved checkpoints (git-ignored)
├── notebooks/                  ← exploration notebooks (not graded)
└── .gitignore
```

---

## Models

Three models are implemented as required:

| Model | Type | Description |
|---|---|---|
| Majority class | Naive baseline | Always predicts the most frequent training class |
| SVM + PCA | Classical ML | SVM with RBF kernel on PCA-reduced (n=100) flattened pixel features |
| EfficientNetB3 | Deep learning | Fine-tuned from ImageNet weights, two-phase training strategy |

All three are trained via `scripts/model.py` and results saved to `models/`.

---

## Experiment: Robustness Under Image Corruptions

**Motivation:** Prior work evaluates weather classifiers on clean test images only. Real-world deployment involves degraded images — motion blur, compression artifacts, low light. We test how much accuracy drops under controlled corruptions.

**Corruptions tested:** Gaussian blur · Gaussian noise · JPEG compression · Brightness reduction · Contrast reduction

**Severity levels:** 1 (mild) through 5 (severe)

Results are saved to `data/outputs/robustness_results.json` and plotted at `data/outputs/robustness_plot.png`.

---

## Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA recommended for training (CPU works but is slow)

See `requirements.txt` for full dependency list.

---

## Live Demo

🔗 [Deployed app](https://huggingface.co/spaces/ecooley/Weather-Image-Classifier)

Upload any weather photo and get:
- Predicted weather condition + confidence score
- Top 5 class probabilities
- GradCAM heatmap showing what the model focused on

---

## Attribution

Dataset: [Weather Image Recognition](https://www.kaggle.com/datasets/jehanbhathena/weather-dataset) by Jehan Bhathena (Kaggle)

AI assistance: Portions of this project were developed with the assistance of [Claude](https://claude.ai) (Anthropic).
