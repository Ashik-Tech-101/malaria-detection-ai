"""Fine-tune a pretrained image model on class-folder malaria images.

Expected layout: data/raw/Parasitized/* and data/raw/Uninfected/*.
Images are split before augmentation; augmentation runs only on training batches.
"""
from __future__ import annotations

import importlib.util
import json
import os
import warnings
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
required_modules = {
    "tensorflow": "tensorflow",
    "matplotlib": "matplotlib",
    "numpy": "numpy",
    "seaborn": "seaborn",
    "PIL": "Pillow",
    "sklearn": "scikit-learn",
}
missing_packages = [
    package for module, package in required_modules.items()
    if importlib.util.find_spec(module) is None
]
if missing_packages:
    raise SystemExit(
        "Training dependencies are missing: "
        f"{', '.join(missing_packages)}. Install requirements-training.txt "
        "using a Python version supported by TensorFlow."
    )

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             confusion_matrix, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("MALARIA_DATA_DIR", ROOT / "data" / "raw"))
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
IMAGE_SIZE = int(os.environ.get("MALARIA_IMAGE_SIZE", "224"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "32"))
SEED = 42
CLASS_NAMES = ("Parasitized", "Uninfected")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}

tf.keras.utils.set_random_seed(SEED)


def build_model() -> tuple[tf.keras.Model, tf.keras.Model]:
    inputs = tf.keras.Input((IMAGE_SIZE, IMAGE_SIZE, 3), name="cell_image")
    x = tf.keras.layers.RandomFlip("horizontal_and_vertical")(inputs)
    x = tf.keras.layers.RandomRotation(0.08)(x)
    x = tf.keras.layers.RandomZoom(0.1)(x)
    x = tf.keras.layers.Rescaling(2.0, offset=-1.0)(x)
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    backbone.trainable = False
    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="uninfected_probability")(x)
    model = tf.keras.Model(inputs, outputs, name="malaria_mobilenetv2")
    return model, backbone


def compile_model(model: tf.keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )


def find_valid_images(directory: Path) -> tuple[list[str], list[int]]:
    image_paths: list[str] = []
    labels: list[int] = []
    for label, class_name in enumerate(CLASS_NAMES):
        class_dir = directory / class_name
        if not class_dir.is_dir():
            raise SystemExit(f"Expected class folder: {class_dir}")
        files = sorted(path for path in class_dir.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
        valid_count = 0
        for path in files:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", Image.DecompressionBombWarning)
                    with Image.open(path) as image:
                        image.load()
                image_paths.append(str(path))
                labels.append(label)
                valid_count += 1
            except (UnidentifiedImageError, OSError, Image.DecompressionBombError,
                    Image.DecompressionBombWarning) as exc:
                print(f"Skipping unreadable image {path}: {exc}")
        if valid_count < 2:
            raise SystemExit(
                f"Need at least 2 valid images in {class_dir}; found {valid_count}."
            )
    return image_paths, labels


def main() -> None:
    if IMAGE_SIZE < 32 or BATCH_SIZE < 1:
        raise SystemExit("MALARIA_IMAGE_SIZE must be >= 32 and BATCH_SIZE must be >= 1.")
    if not DATA_DIR.is_dir():
        raise SystemExit(f"Training data directory does not exist: {DATA_DIR}")
    paths, labels = find_valid_images(DATA_DIR)
    if not paths:
        raise SystemExit("No supported images found.")
    if min(np.bincount(labels)) < 3:
        raise SystemExit("Need at least 3 valid images in each class for a stratified train/validation split.")
    train_p, val_p, train_y, val_y = train_test_split(paths, labels, test_size=0.2, stratify=labels, random_state=SEED)
    class_counts = np.bincount(train_y, minlength=len(CLASS_NAMES))
    class_weights = {
        label: len(train_y) / (len(CLASS_NAMES) * count)
        for label, count in enumerate(class_counts)
    }

    def decode(path, label):
        image = tf.io.read_file(path)
        image = tf.io.decode_image(image, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE)) / 255.0
        return image, tf.cast(label, tf.float32)

    def dataset(p, y, training):
        ds = tf.data.Dataset.from_tensor_slices((p, y))
        if training:
            ds = ds.shuffle(len(p), seed=SEED, reshuffle_each_iteration=True)
        return (
            ds.map(decode, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(BATCH_SIZE)
            .prefetch(tf.data.AUTOTUNE)
        )

    train_ds, val_ds = dataset(train_p, train_y, True), dataset(val_p, val_y, False)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "malaria_cnn.keras"
    model, backbone = build_model()
    initial_epochs = int(os.environ.get("INITIAL_EPOCHS", "10"))
    fine_tune_epochs = int(os.environ.get("FINE_TUNE_EPOCHS", "20"))
    if initial_epochs < 1 or fine_tune_epochs < 0:
        raise SystemExit("INITIAL_EPOCHS must be >= 1 and FINE_TUNE_EPOCHS must be >= 0.")

    def callbacks(
        patience: int, best_validation_auc: float | None = None
    ) -> list[tf.keras.callbacks.Callback]:
        return [
            tf.keras.callbacks.ModelCheckpoint(
                model_path,
                monitor="val_auc",
                mode="max",
                save_best_only=True,
                initial_value_threshold=best_validation_auc,
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_auc", mode="max", patience=patience, restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.35, patience=3, min_lr=1e-7
            ),
        ]

    compile_model(model, learning_rate=1e-3)
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=initial_epochs,
        callbacks=callbacks(patience=4),
        class_weight=class_weights,
    )

    if fine_tune_epochs:
        backbone.trainable = True
        for layer in backbone.layers[:-30]:
            layer.trainable = False
        for layer in backbone.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
        compile_model(model, learning_rate=1e-5)
        best_validation_auc = max(history.history["val_auc"])
        fine_tune_history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=fine_tune_epochs,
            callbacks=callbacks(patience=6, best_validation_auc=best_validation_auc),
            class_weight=class_weights,
        )
        for metric, values in history.history.items():
            values.extend(fine_tune_history.history.get(metric, []))

    model = tf.keras.models.load_model(model_path, compile=False)
    probs = model.predict(val_ds, verbose=0).reshape(-1)
    pred = (probs >= 0.5).astype(int); truth = np.asarray(val_y)
    report = classification_report(truth, pred, labels=[0, 1], target_names=list(CLASS_NAMES), output_dict=True, zero_division=0)
    report["roc_auc"] = float(roc_auc_score(truth, probs))
    report["split"] = {
        "train": len(train_y),
        "validation": len(val_y),
        "seed": SEED,
        "note": "Random image-level split; use patient-level grouping when identifiers are available.",
    }
    report["model"] = "MobileNetV2 ImageNet transfer learning with fine-tuning"
    report["image_size"] = IMAGE_SIZE
    (REPORT_DIR / "metrics.json").write_text(json.dumps(report, indent=2))
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    axes[0].plot(history.history["accuracy"], label="Train"); axes[0].plot(history.history["val_accuracy"], label="Validation"); axes[0].set_title("Accuracy"); axes[0].legend()
    axes[1].plot(history.history["loss"], label="Train"); axes[1].plot(history.history["val_loss"], label="Validation"); axes[1].set_title("Loss"); axes[1].legend()
    ConfusionMatrixDisplay(confusion_matrix(truth, pred, labels=[0, 1]), display_labels=list(CLASS_NAMES)).plot(ax=axes[2], colorbar=False); axes[2].set_title("Validation confusion matrix")
    fig.tight_layout(); fig.savefig(REPORT_DIR / "evaluation.png", dpi=180); plt.close(fig)
    fpr, tpr, _ = roc_curve(truth, probs)
    plt.figure(figsize=(6, 5)); sns.set_theme(style="whitegrid"); plt.plot(fpr, tpr, label=f"AUC = {report['roc_auc']:.3f}"); plt.plot([0, 1], [0, 1], "--", color="gray"); plt.xlabel("False positive rate"); plt.ylabel("True positive rate"); plt.title("Validation ROC curve"); plt.legend(); plt.tight_layout(); plt.savefig(REPORT_DIR / "roc_curve.png", dpi=180); plt.close()
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
