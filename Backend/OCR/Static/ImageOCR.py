import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from paddleocr import PaddleOCR
import gradio as gr
import google.generativeai as genai
from datetime import datetime
import re
from ultralytics import YOLO


GOOGLE_API_KEY = "AIzaSyBk0EiGzaWDq4GeowksCUPkze9F58Ewkkg"
genai.configure(api_key=GOOGLE_API_KEY)


def new_draw_bounding_boxes(image):
    """Draw bounding boxes around detected text in the image and display it."""
    try:
        # Check the input type and load the image
        if isinstance(image, str):
            img = Image.open(image)
            np_img = np.array(img)  # Convert to NumPy array
            print("[DEBUG] Loaded image from file path.")
        elif isinstance(image, Image.Image):
            np_img = np.array(image)  # Convert PIL Image to NumPy array
            print("[DEBUG] Converted PIL Image to NumPy array.")
        else:
            raise ValueError("Input must be a file path or a PIL Image object.")

        # Perform OCR on the array
        ocr_result = ocr.ocr(np_img, cls=True)  # Ensure this line is error-free
        print("[DEBUG] OCR Result:\n", ocr_result)

        # Create a figure to display the image
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        ax = plt.gca()
        all_text_data = []

        # Iterate through the OCR results and draw boxes
        for idx, line in enumerate(ocr_result[0]):
            box = line[0]  # Get the bounding box coordinates
            text = line[1][0]  # Extracted text
            print(f"[DEBUG] Box {idx + 1}: {text}")  # Debug print
            all_text_data.append(text)

            # Draw the bounding box
            polygon = plt.Polygon(box, fill=None, edgecolor='red', linewidth=2)
            ax.add_patch(polygon)

            # Add text label with a small offset for visibility
            x, y = box[0][0], box[0][1]
            ax.text(x, y - 5, f"{idx + 1}: {text}", color='blue', fontsize=12, ha='left')

        plt.axis('off')  # Hide axes
        plt.title("Detected Text with Bounding Boxes", fontsize=16)  # Add a title
        plt.show()

        return all_text_data

    except Exception as e:
        print(f"[ERROR] Error in new_draw_bounding_boxes: {e}")
        return []
    


def gemini_context_correction(text):
    """Use Gemini API to refine noisy OCR results and extract MRP details."""
    model = genai.GenerativeModel('models/gemini-1.5-flash')

    response = model.generate_content(
        f"Identify and extract manufacturing, expiration dates, and MRP from the following text. "
        f"The dates may be written in dd/mm/yyyy format or as <Month_name> <Year> or <day> <Month_Name> <Year>. "
        f"The text may contain noise or unclear information. If only one date is provided, assume it is the Expiration Date. "
        f"Additionally, extract the MRP (e.g., 'MRP: ₹99.00', 'Rs. 99/-'). "
        f"Format the output as:\n"
        f"Manufacturing Date: <MFG Date> Expiration Date: <EXP Date> MRP: <MRP Value>"
        f"Here is the text: {text}"
    )

    return response.text

def validate_dates_with_gemini(mfg_date, exp_date):
    """Use Gemini API to validate and correct the manufacturing and expiration dates."""
    model = genai.GenerativeModel('models/gemini-1.5-flash')
    response = model.generate_content = (
        f"Input Manufacturing Date: {mfg_date}, Expiration Date: {exp_date}. "
        f"If either date is '-1', leave it as is. "
        f"1. If the expiration date is earlier than the manufacturing date, swap them. "
        f"2. If both dates are logically incorrect, suggest new valid dates based on typical timeframes. "
        f"Always respond ONLY in the format:\n"
        f"Manufacturing Date: <MFG Date>, Expiration Date: <EXP Date>"
    )

    # Check if the response contains valid parts
    if response.parts:
        # Process the response to extract final dates
        final_dates = response.parts[0].text.strip()
        return final_dates

    # Return a message or a default value if no valid parts are found
    return "Invalid response from Gemini API."


def extract_and_validate_with_gemini(refined_text):
    """
    Use Gemini API to extract, validate, correct, and swap dates in 'yyyy/mm/dd' format if necessary.
    """
    model = genai.GenerativeModel('models/gemini-1.5-flash')

    # Generate content using Gemini with the refined prompt
    response = model.generate_content(
        f"The extracted text is:\n'{refined_text}'\n\n"
        f"1. Extract the 'Manufacturing Date', 'Expiration Date', and 'MRP' from the above text. "
        f"Ignore unrelated data.\n"
        f"2. If a date or MRP is missing or invalid, return -1 for that field.\n"
        f"3. If the 'Expiration Date' is earlier than the 'Manufacturing Date', swap them.\n"
        f"4. Ensure both dates are in 'dd/mm/yyyy' format. If the original dates are not in this format, convert them. "
        f"However, if the dates are in 'mm/yyyy' format (without a day), leave them as is and return in 'mm/yyyy' format. "
        f"If the dates do not have a day, return them in 'mm/yyyy' format.\n"
        f"5. MRP should be returned in the format 'INR <amount>'. If not found or invalid, return 'INR -1'.\n"
        f"Respond ONLY in this exact format:\n"
        f"Manufacturing Date: <MFG Date>\n"
        f"Expiration Date: <EXP Date>\n"
        f"MRP: <MRP>"
    )

    # Validate the response and extract dates
    if hasattr(response, 'parts') and response.parts:
        final_dates = response.parts[0].text.strip()
        print(f"[DEBUG] Gemini Response: {final_dates}")

        # Extract the dates from the response
        mfg_date_str, exp_date_str, mrp_str = parse_gemini_response(final_dates)

        # Process and swap if necessary
        if mfg_date_str != "-1" and exp_date_str != "-1":
            # Handle dates with possible 'mm/yyyy' format
            mfg_date = parse_date(mfg_date_str)
            exp_date = parse_date(exp_date_str)

            # Swap if Expiration Date is earlier than Manufacturing Date
            swapping_statement = ""
            if exp_date < mfg_date:
                print("[DEBUG] Swapping dates.")
                mfg_date, exp_date = exp_date, mfg_date
                swapping_statement = "Corrected Dates: \n"

            # Return the formatted swapped dates
            return swapping_statement + (
                f"Manufacturing Date: {format_date(mfg_date)}, "
                f"Expiration Date: {format_date(exp_date)}\n"
                f"MRP: {mrp_str}"
            )

        # If either date is -1, return them as-is
        return final_dates

    # Handle invalid responses gracefully
    print("[ERROR] Invalid response from Gemini API.")
    return "Invalid response from Gemini API."

def parse_gemini_response(response_text):
    """
    Helper function to extract Manufacturing Date and Expiration Date from the response text.
    """
    try:
        # Split and extract the dates and MRP
        parts = response_text.split(", ")
        mfg_date_str = parts[0].split(": ")[1].strip()
        exp_date_str = parts[1].split(": ")[1].strip()
        mrp_str = parts[2].split(": ")[1].strip() if len(parts) > 2 else "INR -1"  # Extract MRP
        return mfg_date_str, exp_date_str, mrp_str
    except IndexError:
        print("[ERROR] Failed to parse Gemini response.")
        return "-1", "-1", "INR -1"

def parse_date(date_str):
    """Parse date string to datetime object considering possible formats."""
    if '/' in date_str:  # If the date has slashes, we can parse it
        parts = date_str.split('/')
        if len(parts) == 3:  # dd/mm/yyyy
            return datetime.strptime(date_str, "%d/%m/%Y")
        elif len(parts) == 2:  # mm/yyyy
            return datetime.strptime(date_str, "%m/%Y")
    return datetime.strptime(date_str, "%d/%m/%Y")  # Default fallback

def format_date(date):
    """Format date back to string."""
    if date.day == 1:  # If day is defaulted to 1, return in mm/yyyy format
        return date.strftime('%m/%Y')
    return date.strftime('%d/%m/%Y')


def extract_date(refined_text, date_type):
    """Extract the specified date type from the refined text."""
    if date_type in refined_text:
        try:
            # Split the text and find the date for the specified type
            parts = refined_text.split(',')
            for part in parts:
                if date_type in part:
                    return part.split(':')[1].strip()  # Return the date value
        except IndexError:
            return '-1'  # Return -1 if the date is not found
    return '-1'  # Return -1 if the date type is not in the text




def extract_details_from_validated_output(validated_output):
    """Extract manufacturing date, expiration date, and MRP from the validated output."""
    # Pattern to match the specified format exactly
    pattern = (
        r"Manufacturing Date:\s*([\d\/]+)\s*"
        r"Expiration Date:\s*([\d\/]+)\s*"
        r"MRP:\s*INR\s*([\d\.]+)"
    )

    print("[DEBUG] Validated Output:", validated_output)  # Debug print for input

    match = re.search(pattern, validated_output)

    if match:
        mfg_date = match.group(1)  # Extract Manufacturing Date
        exp_date = match.group(2)   # Extract Expiration Date
        mrp = f"INR {match.group(3)}"  # Extract MRP with INR prefix

        print("[DEBUG] Extracted Manufacturing Date:", mfg_date)  # Debug print for extracted values
        print("[DEBUG] Extracted Expiration Date:", exp_date)
        print("[DEBUG] Extracted MRP:", mrp)
    else:
        print("[ERROR] No match found for the specified pattern.")  # Debug print for errors
        mfg_date, exp_date, mrp = "Not Found", "Not Found", "INR -1"

    return [
        ["Manufacturing Date", mfg_date],
        ["Expiration Date", exp_date],
        ["MRP", mrp]
    ]

def new_draw_bounding_boxes(image):
    """Draw bounding boxes around detected text in the image and display it."""
    # If the input is a string (file path), open the image
    if isinstance(image, str):
        img = Image.open(image)
        np_img = np.array(img)  # Convert to NumPy array
        ocr_result = ocr.ocr(np_img, cls=True)  # Perform OCR on the array
    elif isinstance(image, Image.Image):
        np_img = np.array(image)  # Convert PIL Image to NumPy array
        ocr_result = ocr.ocr(np_img, cls=True)  # Perform OCR on the array
    else:
        raise ValueError("Input must be a file path or a PIL Image object.")

    # Create a figure to display the image
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    ax = plt.gca()
    all_text_data = []

    # Iterate through the OCR results and draw boxes
    for idx, line in enumerate(ocr_result[0]):
        box = line[0]  # Get the bounding box coordinates
        text = line[1][0]  # Extracted text
        print(f"[DEBUG] Box {idx + 1}: {text}")  # Debug print
        all_text_data.append(text)

        # Draw the bounding box
        polygon = plt.Polygon(box, fill=None, edgecolor='red', linewidth=2)
        ax.add_patch(polygon)

        # Add text label with a small offset for visibility
        x, y = box[0][0], box[0][1]
        ax.text(x, y - 5, f"{idx + 1}: {text}", color='blue', fontsize=12, ha='left')

    plt.axis('off')  # Hide axes
    plt.title("Detected Text with Bounding Boxes", fontsize=16)  # Add a title
    plt.show()

    return all_text_data
# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')

def detect_and_ocr(image):
    model = YOLO('Weights/OCR.pt')

    """Detect objects using YOLO, draw bounding boxes, and perform OCR."""
    # Convert input image from PIL to OpenCV format
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Run inference using YOLO model
    results = model(image)
    boxes = results[0].boxes.xyxy.cpu().numpy()  # Extract bounding box coordinates

    extracted_texts = []
    for (x1, y1, x2, y2) in boxes:
        # Draw bounding box on the original image
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        # Perform OCR on the detected region using the original image and bounding box coordinates
        region = image[int(y1):int(y2), int(x1):int(x2)]
        ocr_result = ocr.ocr(region, cls=True)

        # Check if ocr_result is None or empty
        if ocr_result and isinstance(ocr_result, list) and ocr_result[0]:
            for idx, line in enumerate(ocr_result[0]):
                box = line[0]  # Get the bounding box coordinates
                text = line[1][0]  # Extracted text
                print(f"[DEBUG] Box {idx + 1}: {text}")  # Debug output
                extracted_texts.append(text)
        else:
            # Handle case when OCR returns no result
            print(f"[DEBUG] No OCR result for region: ({x1}, {y1}, {x2}, {y2}) or OCR returned None")
            extracted_texts.append("No OCR result found")  # Append a message to indicate no result

    # Convert image to RGB for Gradio display
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Join all extracted texts into a single string
    result_text = "\n".join(str(text) for text in extracted_texts)

    # Call the Gemini context correction function
    refined_text = gemini_context_correction(result_text)
    print("[DEBUG] Gemini Refined Text:\n", refined_text)

    # Validate and correct dates
    validated_output = extract_and_validate_with_gemini(refined_text)

    print("[DEBUG] Validated Output from Gemini:\n", validated_output)

    # Return image with bounding boxes and results
    return image_rgb, result_text, refined_text, validated_output

def further_processing(image, previous_result_text):
    bounding_boxes_list = new_draw_bounding_boxes(image)
    print("[DEBUG] ", bounding_boxes_list, type(bounding_boxes_list))
    combined_text = previous_result_text
    for text in bounding_boxes_list:
        combined_text += text
        combined_text += "\n"
    print("[DEBUG] combined text", combined_text)
    # Call Gemini for context correction and refinement
    refined_output = gemini_context_correction(combined_text)
    print("[DEBUG] Gemini Refined Output:\n", refined_output)

    return refined_output   # Return refined output for display

def handle_processing(validated_output):
    """Decide whether to proceed with further processing."""
    # Extract the manufacturing date, expiration date, and MRP from the string
    try:
        mfg_date_str = validated_output.split("Manufacturing Date: ")[1].split("\n")[0].strip()
        exp_date_str = validated_output.split("Expiration Date: ")[1].split("\n")[0].strip()
        mrp_str = validated_output.split("MRP: ")[1].strip()

        # Check for invalid manufacturing date formats
        if mfg_date_str == "-1":
            mfg_date = -1
        else:
            # Attempt to parse the manufacturing date
            if '/' in mfg_date_str:  # If it's in dd/mm/yyyy or mm/yyyy format
                mfg_date = mfg_date_str
            else:
                mfg_date = -1

        # Check for invalid expiration date formats
        if exp_date_str == "-1":
            exp_date = -1
        else:
            # Attempt to parse the expiration date
            if '/' in exp_date_str:  # If it's in dd/mm/yyyy or mm/yyyy format
                exp_date = exp_date_str
            else:
                exp_date = -1

        # Check MRP validity
        if mrp_str == "INR -1":
            mrp = -1
        else:
            # Ensure MRP is in the correct format
            if mrp_str.startswith("INR "):
                mrp = mrp_str.split("INR ")[1].strip()
            else:
                mrp = -1

        print("Further processing: ", mfg_date, exp_date, mrp)

    except IndexError as e:
        print(f"[ERROR] Failed to parse validated output: {e}")
        return gr.update(visible=False)  # Hide button on error

    # Check if all three values are invalid (-1)
    if mfg_date == -1 and exp_date == -1 and mrp == -1:
        print("[DEBUG] Showing the 'Further Processing' button.")  # Debug print
        return gr.update(visible=True)  # Show 'Further Processing' button

    print("[DEBUG] Hiding the 'Further Processing' button.")  # Debug print
    return gr.update(visible=False)  # Hide button if all values are valid



