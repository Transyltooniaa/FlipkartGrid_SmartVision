import gradio as gr
import cv2
import numpy as np
from ultralytics import YOLO
import Backend.Fruit_Freshness.Banana.run as brn
import Backend.Fruit_Freshness.Apples.run as arn


# Define the freshness function
def freshness(img):
    # Convert Gradio image (PIL.Image) to OpenCV format
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Load the model
    model = YOLO("Weights/freshness.pt", verbose=False)

    # Run inference
    results = model(img)

    # Class names mapping
    class_names = {
        0: 'fresh apple', 1: 'fresh banana', 2: 'fresh bell pepper', 3: 'fresh carrot', 4: 'fresh cucumber',
        5: 'fresh mango', 6: 'fresh orange', 7: 'fresh potato', 8: 'fresh strawberry', 9: 'fresh tomato',
        10: 'rotten apple', 11: 'rotten banana', 12: 'rotten bell pepper', 13: 'rotten carrot', 14: 'rotten cucumber',
        15: 'rotten mango', 16: 'rotten orange', 17: 'rotten potato', 18: 'rotten strawberry', 19: 'rotten tomato'
    }

    output_text = ""

    for i in range(len(results)):
        # Get detected class and bounding box
        detected_class = int(results[i].boxes.cls[0])
        class_name = class_names[detected_class]
        coordinates = results[i].boxes.xyxy

        # Process each bounding box
        for j, box in enumerate(coordinates):
            x1, y1, x2, y2 = map(int, box[:4])
            crop_img = img[y1:y2, x1:x2]

            if detected_class in {1, 11}:  # Banana classes
                score = 0
                result = brn.run(crop_img, score)
                output_text += f"Detected: {class_name}, Banana Freshness: {result}\n"

            elif detected_class in {0, 10}:  # Apple classes
                score = 10
                fresh_flag = False
                stale_flag = False
                result = arn.run(crop_img, stale_flag, fresh_flag, score)
                output_text += f"Detected: {class_name}, Apple Freshness: {result}\n"

    return output_text


