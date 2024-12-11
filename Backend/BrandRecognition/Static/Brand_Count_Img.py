import cv2
import numpy as np
from ultralytics import YOLO
from Database.mongo import DatabaseManager
import threading


db_manager = DatabaseManager()

def preprocess_image(image, input_size=(640, 640), augment=False):
    """
    Advanced image preprocessing for multi-brand object detection.
    
    Args:
        image (numpy.ndarray): Input image in BGR format.
        input_size (tuple): Target image dimensions (width, height).
        augment (bool): Apply data augmentation for training.
    
    Returns:
        numpy.ndarray: Preprocessed image in RGB format.
    """

    # Convert image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize and enhance
    resized_image = cv2.resize(image_rgb, input_size, interpolation=cv2.INTER_NEAREST)
    enhanced_image = cv2.convertScaleAbs(resized_image, alpha=1.65, beta=1.75)
    
    return enhanced_image

def detect_grocery_items(image, model_path=None, threshold=0.5):
    
    """
    Detect grocery items in an image.
    
    Args:
        image (numpy.ndarray): Input image in BGR format.
        model_path (str, optional): Path to YOLO model weights.
        threshold (float, optional): Confidence threshold for detection.
    
    Returns:
        tuple: Annotated image, summary table, and status message
    """
    
    
    # Validate image input
    if image is None or image.size == 0:
        return None, [], "Invalid image input"
    
    # Use default model path if not provided
    if model_path is None:
        model_path = 'Weights/kitkat_s.pt'
    
    # Load YOLO model
    try:
        model = YOLO(model_path)
    except Exception as e:
        return None, [], f"Model loading error: {str(e)}"
    
    # Preprocess image
    processed_image = preprocess_image(image)
    
    # Detect objects
    try:
        results = model(processed_image)
        
        # If no results, return early
        if len(results[0].boxes) == 0:
            return processed_image, [], "No items detected"
        
        # Annotate image
        annotated_image = results[0].plot()
        
        # Process detection results
        class_ids = results[0].boxes.cls.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()
        
        # Aggregate results
        class_counts = {}
        class_confidences = {}

        # Iterate over detected objects
        for i, class_id in enumerate(class_ids):
            
            # Get class name and confidence
            confidence = confidences[i]
            
            # Filter by confidence threshold
            if confidence >= threshold:
                
                # Get class name
                class_name = model.names[int(class_id)]
                class_counts[class_name] = class_counts.get(class_name, 0) + 1

                # Store confidence values
                if class_name not in class_confidences:
                    class_confidences[class_name] = []
                    
                # Add confidence value
                class_confidences[class_name].append(confidence)
        
        # Create summary table
        summary_table = [
            [class_name, count, f"{np.mean(class_confidences[class_name]):.2f}"] 
            for class_name, count in class_counts.items()
        ]
        
        def update_db():
            for brand, count in class_counts.items():
                db_manager.add_brand_record(brand, count)
        
        # Convert back to RGB for display
        annotated_image_rgb = annotated_image[:, :, ::-1]
        
        db_thread = threading.Thread(target=update_db)
        db_thread.start()
        
        return annotated_image_rgb, summary_table, "Objects Recognised Successfully 🥳"
    
    except Exception as e:
        return None, [], f"Detection error: {str(e)}"


# Optional: Batch processing function
def batch_detect_grocery_items(images, model_path=None, threshold=0.4):
    """
    Detect grocery items in multiple images.
    
    Args:
        images (list): List of input images in BGR format.
        model_path (str, optional): Path to YOLO model weights.
        threshold (float, optional): Confidence threshold for detection.
    
    Returns:
        list: List of detection results for each image
    """
    return [detect_grocery_items(img, model_path, threshold) for img in images]