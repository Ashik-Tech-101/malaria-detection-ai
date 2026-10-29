# Malaria Detection AI

A standalone, local-first research prototype for classifying thin blood-smear cell images as **Parasitized** or **Uninfected**. It includes a two-stage MobileNetV2 transfer-learning pipeline, evaluation plots and metrics, and a Flask inference interface. The app validates uploads, reports missing or incompatible models, and avoids Flask debug mode.

**Live dashboard:** <https://malaria-detection-ai.onrender.com/> — predictions use model weights stored in a separate private GitHub repository.

> **Research use only.** This project is not a medical device and must not be used to diagnose or guide treatment. A classifier trained on a small or unrepresentative subset can fail on real-world images. Results require clinical validation by qualified professionals.

## Project contents

- `app.py` — Flask UI, health and validation-metrics endpoints, image validation, and model inference.
- `train.py` — reproducible stratified split, training-only augmentation, pretrained MobileNetV2 feature extraction and fine-tuning, checkpointing, early stopping, and evaluation artifacts.
- `templates/index.html`, `static/` — responsive microscopy interface with a compact specimen preview, score doughnut, per-image score histogram, validation grouped bars, and a tab-scoped historical trend.
- `requirements.txt` — light inference dependencies.
- `render.yaml`, `requirements-deploy.txt` — Render Blueprint and production web-server dependencies.
- `requirements-training.txt` — additional deep-learning and evaluation dependencies.

## 1. Prepare data

Download the NIH Malaria Cell Images dataset via Kaggle: <https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria>. Extract into this structure (folder names and capitalization matter):

```text
data/raw/
├── Parasitized/   # infected cells, label 0
└── Uninfected/    # healthy cells, label 1
```

The full dataset is commonly distributed with 27,558 images. The 998-image subset described in the project brief is suitable for a demonstration, but will have higher uncertainty and wider validation error. Splitting occurs before any augmentation to prevent augmented copies leaking across train/validation sets. Use patient-level grouping if patient identifiers are available; random image-level splitting can overstate generalization when related cells appear in both sets.

## 2. Environment setup

Python 3.10–3.12 is recommended for TensorFlow package compatibility. Anaconda Navigator is optional; a regular Python virtual environment works too. Avoid administrator mode for package installation.

```powershell
cd "malaria_detection"
$venv = Join-Path $env:LOCALAPPDATA "venvs\hemalens-malaria311"
py -3.11 -m venv $venv
$python = Join-Path $venv "Scripts\python.exe"
& $python -m pip install --upgrade pip
& $python -m pip install -r requirements.txt -r requirements-training.txt
```

TensorFlow's supported versions depend on your Python and operating system. If installation fails, check the current official TensorFlow install guide before selecting a compatible Python version. The original brief's `tenserflow` package spelling is incorrect; install `tensorflow`.

## 3. Train and evaluate

Install the training dependencies in the same Python environment as the app. The first training run needs internet access to download MobileNetV2's ImageNet weights; the malaria dataset itself must be placed under `data/raw/` as described above.

```powershell
$python = Join-Path $env:LOCALAPPDATA "venvs\hemalens-malaria311\Scripts\python.exe"
& $python -m pip install -r requirements-training.txt
& $python train.py
```

Optional environment variables: `INITIAL_EPOCHS` (default 10), `FINE_TUNE_EPOCHS` (default 20), `BATCH_SIZE` (default 32), `MALARIA_IMAGE_SIZE` (default 224), and `MALARIA_DATA_DIR` (default `data/raw`). Training first freezes the pretrained backbone, then fine-tunes its last layers at a lower learning rate. It skips unreadable files, requires at least three valid images per class, and saves the best validation-AUC checkpoint to `models/malaria_cnn.keras`, validation metrics to `reports/metrics.json`, and evaluation plots under `reports/`. Validation metrics are computed on held-out images; they are not an independent clinical validation and must not be presented as established performance claims. The model uses a sigmoid output: values near 1 indicate Uninfected; values near 0 indicate Parasitized.

## 4. Run the web app

```powershell
$python = Join-Path $env:LOCALAPPDATA "venvs\hemalens-malaria311\Scripts\python.exe"
& $python app.py
```

Visit <http://127.0.0.1:5000>. Upload PNG, JPG, JPEG, or WebP images up to 12 MB and 20 megapixels. The dashboard shows an image-level class-probability doughnut, a histogram of the parasitized score across up to 10 analyses in the current browser tab, grouped precision/recall/F1 validation bars with accuracy and ROC AUC from `reports/metrics.json`, and a historical parasitized-score line. Histogram scores are not parasite counts or density: this binary classifier does not detect individual cells or measure parasitemia. Regional or patient history is not collected. Model scores are not necessarily calibrated probabilities, and validation performance is not an independent clinical validation. Without a trained model, the UI remains usable and reports setup status; predictions return HTTP 503. `/health` exposes readiness and `/metrics` serves the saved validation summary without image data. Uploaded images are held in memory for inference and are not saved. Input dimensions are read from the loaded model, preventing accidental resizing/model-shape mismatches.

For local web-only setup, install `requirements.txt`. TensorFlow is only needed for inference after adding a compatible model. Set `MALARIA_MODEL_PATH` to load a model from another path. `MALARIA_IMAGE_SIZE` (default 224) is used only when a model has dynamic spatial input dimensions.

## 5. Deploy the public UI to Render

This public repository intentionally excludes the trained model and all dataset images. Render downloads the model at startup from the separate private repository `Ashik-Tech-101/malaria-model-weights`, verifies its SHA-256, and caches it in `/tmp`. The weights and access token must never be committed to this repository. Prediction stays unavailable until Render has a read-only token for the private model repository.

Deploy using the included `render.yaml`:

1. Sign in to Render and choose **New → Blueprint**.
2. Connect `Ashik-Tech-101/malaria-detection-ai`.
3. Review the service settings and apply the Blueprint.

The Blueprint uses Render's free web-service plan, Python 3.11.9, Gunicorn, TensorFlow, and `/health` as its health check. Free instances can spin down when idle and have limited memory. The model path, private GitHub Contents URL, and SHA-256 are configured in `render.yaml`; the access token is deliberately not stored there.

To enable predictions on a newly created Render service:

1. Create a GitHub fine-grained personal access token limited to the `malaria-model-weights` repository, with **Contents: Read-only** permission. Never paste the token into chat or commit it to source.
2. In Render, open the `malaria-detection-ai` service → **Environment** → **Add Environment Variable**. Set `MALARIA_MODEL_TOKEN` to the token and save changes to deploy.
3. Check `/health` for `"model_loaded": true`, then analyze a specimen on the live dashboard.

The app downloads only over HTTPS from `api.github.com`, caps the model download at 100 MB, verifies the configured SHA-256 before loading, and reports a clear readiness error if private access or TensorFlow is unavailable. Render free instances may not have enough memory for TensorFlow inference; if a deploy restarts or fails from memory pressure, use a higher-memory Render instance before relying on predictions.

## Data and interpretation notes

- `Parasitized` is encoded as 0; `Uninfected` as 1. Do not reverse these labels without updating the sigmoid interpretation in the app.
- ImageNet transfer learning is a starting point, not a guarantee of higher clinical accuracy. Evaluate on patient-separated and external-site data when available; the included random image-level split can overestimate generalization.
- Accuracy alone is insufficient. Review sensitivity/recall, specificity, ROC-AUC, confusion matrix, class balance, and performance across acquisition sites and staining variation.
- The commonly quoted 98.99% accuracy and 0.9995 ROC-AUC in the brief are not verified outcomes for this implementation and are not hard-coded as claims.
- Future work: external-site validation, patient-level splits, calibrated scores, whole-slide cell detection/counting, TFLite deployment, multi-species classification, and prospective clinical evaluation.
