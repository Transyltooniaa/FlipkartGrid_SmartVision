# Import necessary libraries
import cv2
from paddleocr import PaddleOCR, draw_ocr
import csv
import google.generativeai as genai
import pandas as pd


# Define paths
ocr = PaddleOCR(use_angle_cls=True, lang='en')
GOOGLE_API_KEY = "AIzaSyBk0EiGzaWDq4GeowksCUPkze9F58Ewkkg"
genai.configure(api_key=GOOGLE_API_KEY)


# Function to add branding to a frame
def add_branding(frame, text="Abhinav Video OCR", position=(50, 50), font_scale=2, font_thickness=3,
                 text_color=(255, 255, 255), bg_color=(0, 0, 0)):
    
    overlay = frame.copy()
    alpha = 0.6  # Transparency factor

    # Get the width and height of the text box
    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
    x, y = position

    # Draw a rectangle and put the text on it
    cv2.rectangle(overlay, (x, y + 10), (x + text_width, y - text_height - 10), bg_color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness)

    return frame

# Function to preprocess the frame for OCR
def preprocess_frame(frame, resize_width=600):
    resized = cv2.resize(frame, (resize_width, int(frame.shape[0] * (resize_width / frame.shape[1]))))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    return gray, resized
def parse_gemini_response(response_text):
    parsed_data = {
        "Manufacturing Date": "",
        "Expiry Date": "",
        "MRP Details": ""
    }
    for line in response_text.split("\n"):
        if line.startswith("Manufacturing Date:"):
            parsed_data["Manufacturing Date"] = line.split("Manufacturing Date:")[1].strip()
        elif line.startswith("Expiry Date:"):
            parsed_data["Expiry Date"] = line.split("Expiry Date:")[1].strip()
        elif line.startswith("MRP Details:"):
            parsed_data["MRP Details"] = line.split("MRP Details:")[1].strip()
    return parsed_data

# Function to call Gemini LLM for date predictions
def call_gemini_llm_for_dates(text):
    # Use the previously set up Gemini model for predictions
    model = genai.GenerativeModel('models/gemini-1.5-flash')
    prompt = f"""
    You are provided with extracted words from a product's packaging. Based on this text, your task is to predict the manufacturing and expiry dates of the product. 

    Please follow these rules:
    - If only one date is present, consider it to be the expiry date.
    - Ignore any noise or irrelevant information.
    - Predict the most logical manufacturing and expiry dates based on the context provided.
    - Output the dates strictly in the format: 
        Manufacturing Date: DD/MM/YYYY
        Expiry Date: DD/MM/YYYY
    - Do not generate any other information or text besides the two dates.

    Here is the extracted text:
    {text}
    """

    # Send the prompt to Gemini model and get the response
    response = model.generate_content(prompt)
    print(response.text)

    return response.text.strip()



# Gradio function to process the video
def gradio_video_ocr_processing(video_file):
    input_video_path = video_file
    output_video_path = "annotated_video.mp4"
    output_text_file = "detected_words.csv"

    print("[DEBUG] Starting video processing.")
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("[ERROR] Cannot open video file.")
        return None, "Error: Cannot open video file."

    input_frame_rate = cap.get(cv2.CAP_PROP_FPS)
    print(f"[DEBUG] Input video frame rate: {input_frame_rate} FPS.")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None
    frame_skip = 2
    resize_width = 600
    detected_words = [["Frame", "Word", "Confidence", "X", "Y", "Width", "Height"]]
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("[DEBUG] End of video stream.")
            break

        if frame_count % frame_skip != 0:
            frame_count += 1
            continue

        # Preprocess frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized_frame = cv2.resize(gray, (resize_width, int(frame.shape[0] * resize_width / frame.shape[1])))
        print(f"[DEBUG] Processing frame {frame_count}.")

        # OCR processing with PaddleOCR
        # OCR processing with PaddleOCR
        results = ocr.ocr(resized_frame)
        if results[0] is not None:
            for line in results[0]:
                word, confidence = line[1][0], float(line[1][1])
                if confidence > 0.7:
                    bbox = line[0]
                    x_min, y_min = int(bbox[0][0]), int(bbox[0][1])
                    x_max, y_max = int(bbox[2][0]), int(bbox[2][1])
                    detected_words.append([frame_count, word, confidence, x_min, y_min, x_max - x_min, y_max - y_min])

                    # Annotate the frame
                    frame = cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    frame = cv2.putText(frame, f"{word} ({confidence:.2f})", (x_min, y_min - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            print(f"[DEBUG] No text detected in frame {frame_count}.")

        frame = add_branding(frame)
        if out is None:
            out = cv2.VideoWriter(output_video_path, fourcc, input_frame_rate // frame_skip,
                                  (frame.shape[1], frame.shape[0]))
        out.write(frame)
        frame_count += 1

    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()

    # Save detected words to CSV
    with open(output_text_file, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerows(detected_words)
    print(f"[INFO] Detected words saved to {output_text_file}.")
    print(f"[INFO] Annotated video saved to {output_video_path}.")

    # Generate Gemini response
    ocr_results_df = pd.read_csv(output_text_file)
    detected_text = " ".join(ocr_results_df['Word'].dropna())
    gemini_response = call_gemini_llm_for_dates(detected_text)
    parsed_output = parse_gemini_response(gemini_response)

    print("[DEBUG] Gemini response generated.")
    return output_video_path, gemini_response, parsed_output

