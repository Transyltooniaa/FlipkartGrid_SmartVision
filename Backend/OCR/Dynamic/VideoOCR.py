# Import necessary libraries
import cv2
from paddleocr import PaddleOCR
import csv
import numpy as np
import google.generativeai as genai
import pandas as pd
from datetime import datetime

def frame_similarity_detection(video_path, scale_factor=0.45, target_frames=120):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Get original width
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Get original height
    print(f"Total number of frames in the video: {total_frames}")
    print(f"Original video resolution: {original_width}x{original_height}")

    # If the total frames are less than the target, handle it gracefully
    if total_frames <= target_frames:
        print(f"Total frames ({total_frames}) are less than or equal to the target frames ({target_frames}). "
              "All frames will be considered non-similar.")
        non_similar_frames = list(range(1, total_frames + 1))  # Consider all frames as non-similar

        # Release resources and return the non-similar frames
        cap.release()
        return non_similar_frames

    # Initialize variables
    prev_frame = None
    frame_count = 0
    non_similar_frames = []
    frame_differences = []  # List to store the sum of frame differences

    # Resize the frame dimensions for processing (not output)
    resized_width = int(original_width * scale_factor)
    resized_height = int(original_height * scale_factor)

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Break the loop if no more frames are available
        
        frame_count += 1

        # Resize the frame to reduce resolution for faster processing
        resized_frame = cv2.resize(frame, (resized_width, resized_height))

        # Convert frame to grayscale (for faster processing)
        gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)

        if prev_frame is not None:
            # Compute absolute difference between current and previous frame
            frame_diff = cv2.absdiff(prev_frame, gray_frame)

            # Calculate the sum of differences
            diff_sum = np.sum(frame_diff)
            frame_differences.append(diff_sum)  # Store the difference sum

        # Set the current frame as the previous frame for the next iteration
        prev_frame = gray_frame

    # Release video capture to free memory
    cap.release()

    # Determine threshold dynamically to get close to target frames
    frame_differences.sort(reverse=True)  # Sort differences in descending order
    if len(frame_differences) >= target_frames:
        threshold = frame_differences[target_frames - 1]  # Get the threshold for the 120th largest difference
    else:
        threshold = frame_differences[-1] if frame_differences else 0  # Fallback to smallest difference

    print(f"Calculated threshold for approximately {target_frames} frames: {threshold}")

    # Reopen the video to process frames again with the determined threshold
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    prev_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        resized_frame = cv2.resize(frame, (resized_width, resized_height))
        gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)

        if prev_frame is not None:
            # Compute absolute difference between current and previous frame
            frame_diff = cv2.absdiff(prev_frame, gray_frame)

            # Calculate the sum of differences
            diff_sum = np.sum(frame_diff)

            # If the difference is above the threshold, consider frames as non-similar
            if diff_sum > threshold:
                non_similar_frames.append(frame_count)  # Save the frame number

        # Set the current frame as the previous frame for the next iteration
        prev_frame = gray_frame

    # If no non-similar frames were detected, add the first frame
    if not non_similar_frames and total_frames > 0:
        non_similar_frames.append(1)  # Consider the first frame as non-similar

    # Print the list of frames that are not similar
    if non_similar_frames:
        print(f"Frames not similar (above dynamic threshold of {threshold}): {non_similar_frames}")
    else:
        print("All frames are similar. One frame has been included.")

    print(f"Total non-similar frames: {len(non_similar_frames)}")

    return non_similar_frames




# Define paths
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# GOOGLE_API_KEY = os.getenv("GEMINI_API")

GOOGLE_API_KEY = "AIzaSyBk0EiGzaWDq4GeowksCUPkze9F58Ewkkg"
genai.configure(api_key=GOOGLE_API_KEY)

# Adjusted branding function to map back to original resolution
def add_branding(frame, text="Annotated Video OCR", position=(50, 50), font_scale=2, font_thickness=3,
                 text_color=(255, 255, 255), bg_color=(0, 0, 0), original_resolution=None):
    
    # Use the original resolution for branding position
    if original_resolution:
        # Map position back to the original resolution
        original_width, original_height = original_resolution
        x, y = position
        x = int(x * (original_width / frame.shape[1]))
        y = int(y * (original_height / frame.shape[0]))
    
    overlay = frame.copy()
    alpha = 0.6  # Transparency factor

    # Get the width and height of the text box
    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
    x_end = x + text_width + 10  # Add padding to the right
    y_end = y + text_height + 10  # Add padding to the bottom

    # Ensure that the rectangle and text are within the frame boundaries
    if x_end > frame.shape[1]:  # Check for overflow horizontally
        x = frame.shape[1] - text_width - 10
        x_end = frame.shape[1]  # Adjust the end point of the rectangle
    if y_end > frame.shape[0]:  # Check for overflow vertically
        y = frame.shape[0] - text_height - 10
        y_end = frame.shape[0]  # Adjust the end point of the rectangle

    # Draw a filled rectangle for background
    cv2.rectangle(overlay, (x, y), (x_end, y_end), bg_color, -1)
    
    # Add the overlay (with transparency)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    # Draw the text
    cv2.putText(frame, text, (x + 5, y + text_height + 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness)

    return frame


# Function to preprocess the frame for OCR
def preprocess_frame(frame, resize_width=600, resize_height=None, grayscale=True):
    # Store original resolution
    original_height, original_width = frame.shape[:2]
    print("[INFO] Original Height: ", original_height, "[INFO] Original Width: ", original_width)

    # If resize_height is provided, resize both width and height independently
    if resize_height is not None:
        resized = cv2.resize(frame, (resize_width, resize_height))
    else:
        # Otherwise, resize only based on the width to maintain aspect ratio
        resized = cv2.resize(frame, (resize_width, int(frame.shape[0] * (resize_width / frame.shape[1]))))

    # Convert to grayscale if the grayscale flag is True
    if grayscale:
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    
    # Return both the resized frame and the original resolution for later use
    return resized, (original_width, original_height)


def parse_gemini_response(response_text):
    def standardize_date(date_str):
        """Convert date into DD/MM/YYYY format."""
        try:
            if "/" in date_str:
                parts = date_str.split("/")
                # If the format is MM/YYYY, append '01' as the day
                if len(parts) == 2:
                    month = datetime.strptime(parts[0], "%b").month if len(parts[0]) == 3 else int(parts[0])
                    return f"01/{month:02d}/{parts[1]}"
                # If the format is DD/MM/YYYY, return as is
                elif len(parts) == 3:
                    day, month, year = parts
                    return f"{int(day):02d}/{int(month):02d}/{int(year)}"
            return date_str  # Return as is if it doesn't match expected patterns
        except Exception:
            return date_str  # Fallback to original string if parsing fails

    parsed_data = {
        "Manufacturing Date": "",
        "Expiry Date": "",
        "MRP": ""
    }

    for line in response_text.split("\n"):
        if line.startswith("Manufacturing Date:"):
            raw_date = line.split("Manufacturing Date:")[1].strip()
            parsed_data["Manufacturing Date"] = standardize_date(raw_date)
        elif line.startswith("Expiry Date:"):
            raw_date = line.split("Expiry Date:")[1].strip()
            parsed_data["Expiry Date"] = standardize_date(raw_date)
        elif line.startswith("MRP:"):
            parsed_data["MRP"] = line.split("MRP:")[1].strip()

    return parsed_data



# Function to call Gemini LLM for date predictions
def call_gemini_llm_for_dates(text):
    # Use the previously set up Gemini model for predictions
    model = genai.GenerativeModel('models/gemini-1.5-flash')
    prompt = f"""
    You are provided with extracted words from a product's packaging. Based on this text, your task is to predict the manufacturing and expiry dates of the product, and extract the MRP details.

    Please follow these rules:
    - If only one date is present, consider it to be the expiry date.
    - If the dates are detected as only Month and Year, provide them in the format MM/YYYY.
    - Ignore any noise or irrelevant information.
    - Predict the most logical manufacturing and expiry dates based on the context provided.
    - For MRP:
        - Extract the value listed as the MRP, considering symbols like "₹", "Rs.", or "MRP".
        - If no MRP is detected, output "MRP: Not available".
    - Output the details strictly in the format:
        Manufacturing Date: DD/MM/YYYY or MM/YYYY
        Expiry Date: DD/MM/YYYY or MM/YYYY
        MRP: ₹<value> or "Not available"
    - Do not generate any other information or text besides the requested details.

    Here is the extracted text:
    {text}
    """



    # Send the prompt to Gemini model and get the response
    response = model.generate_content(prompt)
    print(response.text)

    return response.text.strip()





def gradio_video_ocr_processing(video_file):
    input_video_path = video_file
    output_video_path = "annotated_video.mp4"
    output_text_file = "detected_words.csv"

    print("[DEBUG] Starting video processing.")

    # Step 1: Frame similarity detection
    print("[DEBUG] Detecting non-similar frames.")
    non_similar_frames  = frame_similarity_detection(input_video_path)

    # Step 2: OCR processing and saving the results
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("[ERROR] Cannot open video file.")
        return None, "Error: Cannot open video file."

    input_frame_rate = cap.get(cv2.CAP_PROP_FPS)
    print(f"[DEBUG] Input video frame rate: {input_frame_rate} FPS.")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None

    frame_skip = 2
  
    detected_words = [["Frame", "Word", "Confidence", "X", "Y", "Width", "Height"]]
    frame_count = 0
    resize_width=600    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("[DEBUG] End of video stream.")
            break

        # Only process non-similar frames
        if frame_count not in non_similar_frames:
            frame_count += 1
            continue  # Skip similar frames

        # Preprocess frame
        resized_frame, original_resolution = preprocess_frame(frame, resize_width)

        print(f"[DEBUG] Processing frame {frame_count}.")

        # OCR processing with PaddleOCR
        results = ocr.ocr(resized_frame)
        if results[0] is not None:
          for line in results[0]:
              word, confidence = line[1][0], float(line[1][1])
              if confidence > 0.7:
                  bbox = line[0]
                  
                  # Get bounding box coordinates in the resized frame
                  x_min_resized, y_min_resized = int(bbox[0][0]), int(bbox[0][1])
                  x_max_resized, y_max_resized = int(bbox[2][0]), int(bbox[2][1])
                  
                  original_width, original_height=original_resolution
                  resized_height = (original_height/original_width)*resize_width
                  # Rescale the bounding box back to the original resolution
                  x_min = int(x_min_resized * (original_width / resize_width))
                  y_min = int(y_min_resized * (original_height / resized_height))
                  x_max = int(x_max_resized * (original_width / resize_width))
                  y_max = int(y_max_resized * (original_height / resized_height))
                  
                  detected_words.append([frame_count, word, confidence, x_min, y_min, x_max - x_min, y_max - y_min])

                  # Annotate the frame with the detected text box on the original resolution
                  frame = cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                  frame = cv2.putText(frame, f"{word} ({confidence:.2f})", (x_min, y_min - 10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
          print(f"[DEBUG] No text detected in frame {frame_count}.")
        frame = add_branding(frame, original_resolution=original_resolution)

        # Add branding to the frame using the original resolution for correct placement
        

        if out is None:
            out = cv2.VideoWriter(output_video_path, fourcc, input_frame_rate, 
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
    ocr_results_df_clean = ocr_results_df.drop_duplicates(subset='Word', keep='first')  # Clean the duplicates in "Word" column

    detected_text = " ".join(ocr_results_df_clean['Word'].dropna())
    gemini_response = call_gemini_llm_for_dates(detected_text)
    parsed_output = parse_gemini_response(gemini_response)

    print("[DEBUG] Gemini response generated.")
    return output_video_path, gemini_response, parsed_output

