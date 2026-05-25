"""
train.py — Stage 1: YOLO11n Meter Box Detector
================================================
Trains YOLO11n to detect the electricity meter display region.
OCR (Stage 2) is intentionally excluded for now.

Dataset layout expected:
    dataset/
        train/images/  + train/labels/
        valid/images/  + valid/labels/
        test/images/   + test/labels/
        data.yaml
"""

import os
import re
import shutil
from ultralytics import YOLO
from config import DATASET_ROOT, SAVED_MODELS_DIR, NO_EPOCHS, IMG_SIZE, BATCH, DEVICE, WORKERS

# ── Config ────────────────────────────────────────────────────────────────────

EPOCHS      = NO_EPOCHS
IMG_SIZE    = IMG_SIZE
BATCH       = BATCH
WORKERS     = WORKERS
DEVICE      = DEVICE 





# ── Step 1: Train ─────────────────────────────────────────────────────────────

def train_detector() -> str:
    """
    Fine-tune YOLO11n on the meter_box dataset.
    Returns the path to the saved best weights.
    """
    print("\n[Train] Starting YOLO11n training...")

    model = YOLO("yolo11n.pt")      # pretrained COCO nano weights

    model.train(
        data=os.path.join(DATASET_ROOT, "data.yaml"),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        workers=WORKERS,
        device=DEVICE,
        name="digital_digit_detector",
        exist_ok=True,              # resume/overwrite if run dir exists
        # Augmentation (useful for small datasets)
        # degrees=10.0,
        # translate=0.1,
        # scale=0.4,
        # flipud=0.0,                 # no vertical flip — meters are upright
        # fliplr=0.0,                 # no horizontal flip — digit order matters
        # mosaic=1.0,
        # mixup=0.1,
    )

    # Save best + last weights to a stable location
    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

    best_src = "runs/detect/digital_digit_detector/weights/best.pt"
    last_src = "runs/detect/digital_digit_detector/weights/last.pt"
    best_dst = os.path.join(SAVED_MODELS_DIR, "digital_digit_best.pt")
    last_dst = os.path.join(SAVED_MODELS_DIR, "digital_digit_last.pt")

    if os.path.exists(best_src):
        shutil.copy(best_src, best_dst)
        print(f"[Train] Best weights → {best_dst}")
    else:
        print("[Train] WARNING: best.pt not found — check training output.")
        best_dst = last_src         # fallback

    if os.path.exists(last_src):
        shutil.copy(last_src, last_dst)
        print(f"[Train] Last weights → {last_dst}")

    return best_dst


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train_detector()
    print("\n✅  Training complete. Weights saved to:", SAVED_MODELS_DIR)
    print("    Next step: run  python check.py <path/to/image.jpg>")