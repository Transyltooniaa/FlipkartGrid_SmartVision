import gradio as gr
from Frontend.OCR.ImageOCR import createStaticOcrInterface
from Frontend.OCR.VideoOCR import createVideoOcrInterface

def create_OCR_Interface():
    """
    Create an interface for OCR in an image or video.

    The interface allows users to upload an image or video, and the model will return the detected text.

    Returns:
        gr.Interface: A Gradio interface object.
    """
    with gr.Blocks() as ocr_interface:
        gr.Markdown("# Flipkart Grid Robotics Track - OCR Interface")

        with gr.Tabs():
            # Image OCR Tab
            with gr.TabItem("Image OCR"):
                createStaticOcrInterface()

            # Video OCR Tab
            with gr.TabItem("Video OCR"):
                createVideoOcrInterface()

    return ocr_interface

Ocr = create_OCR_Interface()