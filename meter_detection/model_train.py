"""
train.py — Stage 1: YOLO11n Meter Box Detector
================================================
Trains YOLO11n to detect analog and digital electricity meter display regions.
Fine-tunes from official YOLO11n pretrained weights (COCO).

Dataset layout expected:
    dataset/
        train/images/  + train/labels/
        valid/images/  + valid/labels/
        test/images/   + test/labels/
        data.yaml
"""

import os
import shutil

from ultralytics import YOLO
from config import DATASET_ROOT, SAVED_MODELS_DIR, NO_EPOCHS, IMG_SIZE, BATCH, DEVICE, WORKERS

# ── Config ────────────────────────────────────────────────────────────────────

EPOCHS   = NO_EPOCHS
IMG_SIZE = IMG_SIZE
BATCH    = BATCH
WORKERS  = WORKERS
DEVICE   = DEVICE          # GPU index (0 = first GPU). Use "cpu" if no GPU.

BASE_MODEL = "yolo11n.pt"   # Official YOLO11n pretrained on COCO — downloaded automatically if not present


# ── Train ─────────────────────────────────────────────────────────────────────

def train_detector() -> str:
    """
    Fine-tune YOLO11n (COCO pretrained) on the analog+digital meter dataset.
    Returns the path to the saved best weights.
    """
    print(f"\n[Train] Loading base model: {BASE_MODEL}")
    print("[Train] Fine-tuning on analog + digital meter dataset...\n")

    model = YOLO(BASE_MODEL)   # starts from COCO weights, not a previous custom model

    model.train(
        data=os.path.join(DATASET_ROOT, "data.yaml"),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        workers=WORKERS,
        device=DEVICE,
        name="meter_box_detector",
        exist_ok=True,              # resume/overwrite if run dir exists
        pretrained=True,            # explicitly use COCO pretrained backbone
        # ── Augmentation (tune for your dataset size) ──
        # degrees=10.0,             # rotation
        # translate=0.1,
        # scale=0.4,
        # flipud=0.0,               # no vertical flip — meters are upright
        # fliplr=0.5,               # horizontal flip is usually safe
        # mosaic=1.0,
        # mixup=0.1,
    )

    # ── Save weights to a stable location ────────────────────────────────────
    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

    best_src = "runs/detect/meter_box_detector/weights/best.pt"
    last_src = "runs/detect/meter_box_detector/weights/last.pt"
    best_dst = os.path.join(SAVED_MODELS_DIR, "meter_box_analog_digital_best.pt")
    last_dst = os.path.join(SAVED_MODELS_DIR, "meter_box_analog_digital_last.pt")

    if os.path.exists(best_src):
        shutil.copy(best_src, best_dst)
        print(f"\n[Train] Best weights → {best_dst}")
    else:
        print("\n[Train] WARNING: best.pt not found — check training output.")
        best_dst = last_src     # fallback to last.pt

    if os.path.exists(last_src):
        shutil.copy(last_src, last_dst)
        print(f"[Train] Last weights → {last_dst}")

    return best_dst


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  YOLO11n — Analog & Digital Meter Detector Training")
    print("=" * 55)
    print(f"  Base model  : {BASE_MODEL}  (COCO pretrained)")
    print(f"  Dataset     : {DATASET_ROOT}")
    print(f"  Epochs      : {EPOCHS}")
    print(f"  Image size  : {IMG_SIZE}")
    print(f"  Batch size  : {BATCH}")
    print(f"  Device      : {DEVICE}")
    print("=" * 55)

    train_detector()

    print("\n✅  Training complete.")
    print(f"    Weights saved to : {SAVED_MODELS_DIR}/")
    print("    Next step        : python check.py <path/to/image.jpg>")