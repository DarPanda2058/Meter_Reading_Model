import argparse
import cv2
from ultralytics import YOLO
from config import MODEL_PATH, DATA_YAML_PATH, TEST_IMAGES
# Override class names here if model.names still shows old labels (e.g. 'meter')
CLASS_NAMES = {0: "analog", 1: "digital"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def resolve_names(model) -> dict:
    """
    Use CLASS_NAMES override if the model's baked-in names look wrong
    (e.g. still says 'meter' from a previously trained checkpoint).
    """
    baked = {k: v.lower() for k, v in model.names.items()}
    known_valid = {"analog", "digital"}

    if not any(n in known_valid for n in baked.values()):
        print(f"[WARN] model.names looks stale ({baked}) — using CLASS_NAMES override.")
        return CLASS_NAMES

    return baked


def get_meter_type(results, names: dict):
    """
    Extracts the dominant meter type from prediction results.
    Returns (meter_type, confidence, all_detections).
    """
    if not results or len(results[0].boxes) == 0:
        return "unknown", 0.0, []

    detections = []
    for box in results[0].boxes:
        class_id   = int(box.cls[0])
        confidence = float(box.conf[0])
        label      = names.get(class_id, f"class_{class_id}").lower()
        detections.append((label, confidence))

    best_label, best_conf = max(detections, key=lambda x: x[1])
    return best_label, best_conf, detections


def print_meter_result(meter_type, confidence, all_detections, names):
    print("\n" + "=" * 40)
    print("       METER TYPE CLASSIFICATION")
    print("=" * 40)

    if meter_type == "unknown":
        print("  ⚠  No meter detected in this image.")
        print(f"     (Model classes: {list(names.values())})")
    else:
        icon = "🔵" if meter_type == "digital" else "🔴"
        print(f"  {icon}  Meter Type  : {meter_type.upper()}")
        print(f"      Confidence  : {confidence * 100:.1f}%")

        if len(all_detections) > 1:
            print(f"\n  All detections ({len(all_detections)} total):")
            for label, conf in sorted(all_detections, key=lambda x: x[1], reverse=True):
                print(f"    - {label.upper():10s}  {conf * 100:.1f}%")

    print("=" * 40 + "\n")


# ── Evaluate ──────────────────────────────────────────────────────────────────

def evaluate_model():
    print(f"Loading model from {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)
    names = resolve_names(model)
    print(f"Class names in use: {names}")

    print(f"Evaluating on test dataset: {DATA_YAML_PATH}")
    metrics = model.val(
        data=DATA_YAML_PATH,
        split='test',
        project='.',
        name='model_check_images',
        exist_ok=True,
        save=True
    )

    print("\n--- Evaluation Results ---")
    print(f"mAP50-95 : {metrics.box.map:.4f}")
    print(f"mAP50    : {metrics.box.map50:.4f}")
    print(f"mAP75    : {metrics.box.map75:.4f}")

    print("\nRunning inference on test images...")
    model.predict(
        source=TEST_IMAGES,
        project='.',
        name='model_check_images',
        exist_ok=True,
        save=True
    )
    print("Images saved to 'model_check_images' directory.")


# ── Single image predict ──────────────────────────────────────────────────────

def predict_single_image(img_path, conf_thresh):
    print(f"Loading model from {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)
    names = resolve_names(model)
    print(f"Class names in use: {names}")

    print(f"Running inference on: {img_path}  (conf: {conf_thresh})")
    results = model.predict(source=img_path, conf=conf_thresh)

    meter_type, confidence, all_detections = get_meter_type(results, names)
    print_meter_result(meter_type, confidence, all_detections, names)

    if results:
        annotated_img = results[0].plot()

        if meter_type != "unknown":
            label_text = f"{meter_type.upper()}  {confidence * 100:.1f}%"
            color = (255, 100, 0) if meter_type == "digital" else (0, 80, 255)  # BGR
            cv2.putText(
                annotated_img, label_text,
                (15, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                color, 3, cv2.LINE_AA
            )

        window_name = f"Meter Detection — {meter_type.upper()} (press any key to close)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        h, w = annotated_img.shape[:2]
        max_dim = 800
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            cv2.resizeWindow(window_name, int(w * scale), int(h * scale))
        else:
            cv2.resizeWindow(window_name, w, h)

        cv2.imshow(window_name, annotated_img)
        print("Press ANY KEY in the image window to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate or predict with the analog/digital meter model.")
    parser.add_argument('-i', '--image', type=str, help="Path to a single image for prediction.")
    parser.add_argument('-c', '--conf',  type=float, default=0.25, help="Confidence threshold.")
    args = parser.parse_args()

    if args.image:
        predict_single_image(args.image, args.conf)
    else:
        evaluate_model()