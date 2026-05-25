import argparse
import cv2
from ultralytics import YOLO
from config import DATASET_ROOT, SAVED_MODELS_DIR, NO_EPOCHS, IMG_SIZE, BATCH, DEVICE, WORKERS


def evaluate_model():
    # Path to the best trained model weights
    model_path = SAVED_MODELS_DIR + "/" + "digital_digit_best.pt"
    
    # Path to the dataset configuration file
    data_yaml_path = DATASET_ROOT + "/data.yaml"
    
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)
    
    print(f"Evaluating model on test dataset using {data_yaml_path}...")
    # Run validation on the test split, saving results to model_check_images
    metrics = model.val(
        data=data_yaml_path, 
        split='test',
        project='.', 
        name='model_check_images', 
        exist_ok=True,
        save=True
    )
    
    # Print out some key metrics
    print("\n--- Evaluation Results ---")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP75: {metrics.box.map75:.4f}")
    
    print("\nRunning inference to save all individual detected images...")
    # Run prediction to save individual detected images
    test_images_path = DATASET_ROOT + "/test/images"
    model.predict(
        source=test_images_path,
        project='.',
        name='model_check_images',
        exist_ok=True, # Will save inside model_check_images/
        save=True
    )
    
    print("\nImages with detected bounding boxes and evaluation results have been saved to the 'model_check_images' directory.")

def predict_single_image(img_path, conf_thresh):
    model_path = SAVED_MODELS_DIR + "/" + "digital_digit_best.pt"
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)
    
    print(f"Running inference on {img_path} with confidence {conf_thresh}...")
    
    # Run prediction and get results
    results = model.predict(source=img_path, conf=conf_thresh)
    
    if results:
        # Generate the annotated image with bounding boxes
        annotated_img = results[0].plot()
        
        window_name = "Detected Drum Digits (Press any key to exit)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        # Get image dimensions
        h, w = annotated_img.shape[:2]
        
        # Calculate a reasonable size to fit on most screens (e.g., max 800x800)
        max_dim = 800
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            cv2.resizeWindow(window_name, new_w, new_h)
        else:
            cv2.resizeWindow(window_name, w, h)
            
        # Display using OpenCV so we can control when it closes
        cv2.imshow(window_name, annotated_img)
        print("\nPopup opened! Press ANY KEY while the image window is active to close it.")
        cv2.waitKey(0) # Waits indefinitely until a key is pressed
        cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate YOLO model or predict on a single image.")
    parser.add_argument('-i', '--image', type=str, help="Path to a single image for prediction.")
    parser.add_argument('-c', '--conf', type=float, default=0.25, help="Confidence threshold for prediction.")
    
    args = parser.parse_args()
    
    if args.image:
        predict_single_image(args.image, args.conf)
    else:
        evaluate_model()
