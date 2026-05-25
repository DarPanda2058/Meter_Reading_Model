"""
api.py — Meter Detection & OCR API  (v4.0)
===========================================
The uploaded image is passed directly to the digit-YOLO model.
No Stage-1 meter-box detection — the user is expected to send
an already-cropped meter bar image.

Endpoints:
    POST /detect   — Read digit values from the uploaded meter image.
    GET  /         — Health check.
"""

import io
import cv2
import numpy as np
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from ultralytics import YOLO
from enum import Enum

# ── Config ────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent

DRUM_DIGIT_MODEL_PATH    = BASE_DIR / "saved_models" / "drum_digit_best.pt"
DIGITAL_DIGIT_MODEL_PATH = BASE_DIR / "saved_models" / "digital_digit_best.pt"

DIGIT_CONF = 0.25   # confidence threshold for digit detection
DIGIT_IOU  = 0.45   # NMS IoU threshold

# ── App & model setup ─────────────────────────────────────────────────────────

app = FastAPI(
    title="Meter Digit OCR API",
    description=(
        "Send a meter image directly to /detect.\n"
        "Digit-YOLO reads all digits, sorts L→R, applies red-drum decimal logic."
    ),
    version="4.0.0",
)


class MeterType(str, Enum):
    drum    = "drum"
    digital = "digital"


def _load_yolo(path: Path, label: str) -> YOLO:
    if not path.exists():
        raise RuntimeError(f"{label} weights not found at: {path}")
    try:
        model = YOLO(str(path))
        print(f"[Startup][OK] {label} loaded from: {path}")
        return model
    except Exception as exc:
        raise RuntimeError(f"Failed to load {label}: {exc}")


print("[Startup] Loading models...")
try:
    drum_detector = _load_yolo(DRUM_DIGIT_MODEL_PATH, "Drum-digit YOLO")
except RuntimeError as exc:
    print(f"[Startup][ERROR] {exc}")
    raise SystemExit(1)

try:
    digital_detector = _load_yolo(DIGITAL_DIGIT_MODEL_PATH, "Digital-digit YOLO")
except RuntimeError as exc:
    print(f"[Startup][WARN] {exc} — /detect with meter_type=digital will return 503.")
    digital_detector = None

print("[Startup] Ready.")


# ── Main endpoint ─────────────────────────────────────────────────────────────

@app.post("/detect")
async def detect_meter(
    file: UploadFile = File(...),
    meter_type: MeterType = Form(...),
):
    """
    Accepts a meter image + meter_type ('drum' or 'digital').
    The image is passed directly to the digit-YOLO model — no prior
    meter-box cropping is performed.
    Returns the annotated image as JPEG; reading is in response headers.
    """
    if meter_type == MeterType.digital and digital_detector is None:
        raise HTTPException(503, "Digital-digit model is not yet trained / loaded.")

    # Decode upload
    img = await _decode_upload(file)

    # Run digit detection directly on the uploaded image
    if meter_type == MeterType.drum:
        out_img, reading, complete, red_present, red_x = _run_digit_yolo(
            drum_detector, img, "debug_drum.jpg"
        )
    else:
        out_img, reading, complete, red_present, red_x = _run_digit_yolo(
            digital_detector, img, "debug_digital.jpg"
        )

    # Encode & return
    ok, encoded = cv2.imencode(".jpg", out_img)
    if not ok:
        raise HTTPException(500, "Failed to encode output image.")

    headers = {
        "X-Meter-Type"      : meter_type.value,
        "X-Meter-Reading"   : reading,
        "X-Reading-Complete": str(complete),
        "X-Red-Drum-Present": str(red_present),
        **({"X-Red-Drum-X": str(red_x)} if red_x is not None else {}),
    }
    return StreamingResponse(
        io.BytesIO(encoded.tobytes()),
        media_type="image/jpeg",
        headers=headers,
    )


@app.get("/")
def health_check():
    return {"status": "running", "message": "POST image + meter_type to /detect"}


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _decode_upload(file: UploadFile) -> np.ndarray:
    try:
        data = await file.read()
        arr  = np.frombuffer(data, np.uint8)
        img  = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("cv2.imdecode returned None")
    except Exception as exc:
        raise HTTPException(400, f"Invalid image: {exc}")
    return img


def _run_digit_yolo(
    model: YOLO,
    img: np.ndarray,
    debug_path: str,
) -> tuple[np.ndarray, str, bool, bool, int | None]:
    """
    Run digit-YOLO on the full image.
    - Sort detections left → right.
    - Tag red-drum boxes via HSV colour analysis (>15% red pixels).
    - Detections left of the leftmost red box  → integer part.
    - Detections from the leftmost red box onward → decimal part.
    - No red detected → all digits are integer part, decimal defaults to '.0'.
    - Leading zeros are stripped from the integer part.
    """
    results     = model.predict(source=img, conf=DIGIT_CONF, iou=DIGIT_IOU, verbose=False)
    boxes       = results[0].boxes
    annotated   = results[0].plot()

    if boxes is None or len(boxes) == 0:
        print(f"[WARN] No detections — {debug_path}")
        cv2.imwrite(debug_path, annotated)
        return annotated, "?.?", False, False, None

    # Parse detections
    detections = []
    for box in boxes:
        cls_id = int(box.cls[0])
        conf   = float(box.conf[0])
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
        detections.append({
            "digit"   : model.names[cls_id],
            "conf"    : conf,
            "x_centre": (x1 + x2) / 2,
            "x1": x1, "x2": x2,
            "y1": y1, "y2": y2,
        })

    # Tag red boxes
    red_present, red_x, _ = _tag_red_detections(img, detections)

    # Sort left → right
    detections.sort(key=lambda d: d["x_centre"])

    # Split at red boundary
    if red_present and red_x is not None:
        int_digits = [d["digit"] for d in detections if d["x_centre"] <  red_x]
        dec_digits = [d["digit"] for d in detections if d["x_centre"] >= red_x]
    else:
        int_digits = [d["digit"] for d in detections]
        dec_digits = []

    # Strip leading zeros from integer part; decimal defaults to '0'
    int_str  = "".join(int_digits).lstrip("0") or "0"
    dec_str  = "".join(dec_digits) if dec_digits else "0"
    reading  = f"{int_str}.{dec_str}"
    complete = "?" not in reading

    print(f"[DEBUG] Reading: {reading}  (red_present={red_present}, red_x={red_x})")
    print("[DEBUG] Detections: "
          + ", ".join(
              f"{d['digit']}@x={d['x_centre']:.0f}"
              f"(conf={d['conf']:.2f},red={d.get('is_red',False)})"
              for d in detections
          ))

    cv2.imwrite(debug_path, annotated)
    return annotated, reading, complete, red_present, red_x


def _tag_red_detections(
    img: np.ndarray,
    detections: list[dict],
) -> tuple[bool, int | None, list[float]]:
    """
    Sample each bounding-box region in HSV.
    A box is 'red drum' if >15% of its pixels are red.
    Mutates each detection dict with 'is_red'.
    Returns (any_red_found, leftmost_red_x_pixel, list_of_red_x_centres).
    """
    hsv      = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask_lo  = cv2.inRange(hsv, np.array([0,   100, 80]), np.array([10,  255, 255]))
    mask_hi  = cv2.inRange(hsv, np.array([160, 100, 80]), np.array([180, 255, 255]))
    red_mask = cv2.bitwise_or(mask_lo, mask_hi)

    h_img, w_img   = img.shape[:2]
    red_x_centres: list[float] = []

    for det in detections:
        bx1  = max(0, int(det["x1"]))
        bx2  = min(w_img, int(det["x2"]))
        by1  = max(0, int(det["y1"]))
        by2  = min(h_img, int(det["y2"]))
        area = (bx2 - bx1) * (by2 - by1)

        if area == 0:
            det["is_red"] = False
            continue

        red_px        = int(red_mask[by1:by2, bx1:bx2].sum() // 255)
        det["is_red"] = (red_px / area) > 0.15

        if det["is_red"]:
            red_x_centres.append(det["x_centre"])
            print(f"[DEBUG] Red box: digit={det['digit']}  "
                  f"red_ratio={red_px/area:.2f}  x_centre={det['x_centre']:.0f}")

    if red_x_centres:
        return True, int(min(red_x_centres)), red_x_centres
    return False, None, []


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)