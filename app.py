"""Malaria smear classifier web interface. Model is optional until trained."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.environ.get("MALARIA_MODEL_PATH", BASE_DIR / "models" / "malaria_cnn.keras"))
IMAGE_SIZE = int(os.environ.get("MALARIA_IMAGE_SIZE", "224"))
MAX_UPLOAD_BYTES = 12 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
CLASS_NAMES = ("Parasitized", "Uninfected")
model_input_size = (IMAGE_SIZE, IMAGE_SIZE)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
model = None
model_error = "Model file not found. Train the model or set MALARIA_MODEL_PATH."
inference_lock = Lock()
METRICS_PATH = BASE_DIR / "reports" / "metrics.json"


def validate_model(loaded_model) -> tuple[int, int]:
    input_shape = loaded_model.input_shape
    output_shape = loaded_model.output_shape
    if isinstance(input_shape, list) or isinstance(output_shape, list):
        raise ValueError("Expected one image input and one sigmoid output.")
    if len(input_shape) != 4 or input_shape[-1] not in (3, None):
        raise ValueError("Expected a model input shaped (batch, height, width, 3).")
    if len(output_shape) != 2 or output_shape[-1] != 1:
        raise ValueError("Expected one sigmoid output for the two malaria classes.")
    height, width = input_shape[1:3]
    if height is None or width is None:
        return IMAGE_SIZE, IMAGE_SIZE
    if not isinstance(height, int) or not isinstance(width, int) or min(height, width) < 1:
        raise ValueError("Model image dimensions must be positive integers.")
    return height, width


def load_model() -> None:
    global model, model_error, model_input_size
    if not MODEL_PATH.is_file():
        model = None
        model_error = f"Model not found at {MODEL_PATH}"
        return
    try:
        import tensorflow as tf
        loaded_model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        model_input_size = validate_model(loaded_model)
        model = loaded_model
        model_error = ""
    except Exception as exc:  # TensorFlow is an optional dependency for UI-only setup.
        model = None
        model_error = f"Could not load a compatible model: {exc}"


load_model()


def prepare_image(file_storage) -> np.ndarray:
    with Image.open(file_storage.stream) as source:
        if source.width * source.height > MAX_IMAGE_PIXELS:
            raise ValueError(f"Image exceeds the {MAX_IMAGE_PIXELS:,}-pixel limit.")
        image = source.convert("RGB")
    image = image.resize((model_input_size[1], model_input_size[0]), Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float32)[None, ...] / 255.0


@app.get("/")
def home():
    return render_template("index.html", model_ready=model is not None, model_error=model_error)


@app.get("/metrics")
def metrics():
    if not METRICS_PATH.is_file():
        return jsonify(available=False, error="Validation metrics are not available. Train and evaluate the model first."), 404
    try:
        report = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        classes = {
            class_name: {
                metric: float(report[class_name][metric])
                for metric in ("precision", "recall", "f1-score")
            }
            for class_name in CLASS_NAMES
        }
        accuracy = float(report["accuracy"])
        roc_auc = float(report["roc_auc"])
        validation_count = int(report["split"]["validation"])
        if (
            validation_count < 1
            or any(
                not np.isfinite(value) or not 0 <= value <= 1
                for class_metrics in classes.values()
                for value in class_metrics.values()
            )
            or not np.isfinite(accuracy)
            or not 0 <= accuracy <= 1
            or not np.isfinite(roc_auc)
            or not 0 <= roc_auc <= 1
        ):
            raise ValueError("Metrics contain values outside their valid ranges.")
        return jsonify(
            available=True,
            classes=classes,
            accuracy=accuracy,
            roc_auc=roc_auc,
            validation_count=validation_count,
            split_note=report.get("split", {}).get("note", ""),
            model=report.get("model", "Trained classifier"),
        )
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        app.logger.exception("Could not read the saved validation metrics.")
        return jsonify(available=False, error=f"Validation metrics could not be loaded: {exc}"), 500


@app.get("/health")
def health():
    return jsonify(status="running", model_loaded=model is not None, model_error=model_error)


@app.post("/predict")
def predict():
    if model is None:
        return jsonify(success=False, error="The trained model is not available yet.", detail=model_error), 503
    upload = request.files.get("file")
    if upload is None or not upload.filename:
        return jsonify(success=False, error="Choose a microscopy image first."), 400
    suffix = upload.filename.rsplit(".", 1)[-1].lower() if "." in upload.filename else ""
    if suffix not in ALLOWED_EXTENSIONS:
        return jsonify(success=False, error="Upload a PNG, JPG, JPEG, or WebP image."), 415
    try:
        batch = prepare_image(upload)
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, ValueError) as exc:
        return jsonify(success=False, error=f"Image could not be processed: {exc}"), 400
    try:
        with inference_lock:
            probabilities = np.asarray(model(batch, training=False)).reshape(-1)
            if (
                probabilities.size != 1
                or not np.isfinite(probabilities[0])
                or not 0 <= probabilities[0] <= 1
            ):
                raise ValueError("Model returned an invalid sigmoid probability.")
            p_uninfected = float(probabilities[0])
        p_parasitized = 1.0 - p_uninfected
        predicted = int(p_uninfected >= 0.5)
        return jsonify(
            success=True,
            prediction=CLASS_NAMES[predicted],
            confidence=round(float(max(p_uninfected, p_parasitized)) * 100, 2),
            probabilities={"Parasitized": round(p_parasitized * 100, 2), "Uninfected": round(p_uninfected * 100, 2)},
            analyzed_images=1,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
    except Exception:
        app.logger.exception("Inference failed")
        return jsonify(success=False, error="Prediction failed. Check that the model matches the documented input format."), 500


@app.errorhandler(413)
def too_large(_):
    return jsonify(success=False, error="Image exceeds the 12 MB upload limit."), 413


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "5000")), debug=False)
