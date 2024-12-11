import gradio as gr
from Backend.OCR.Dynamic.VideoOCR import gradio_video_ocr_processing
def createVideoOcrInterface():
    """
    Create an interface for OCR in a video.

    The interface allows users to upload a video, and the model will return an annotated video with bounding boxes and detected text.
    """
    return gr.Interface(
        fn=gradio_video_ocr_processing, 
        inputs=gr.Video(label="Upload Video File (.mp4)"), 
        outputs=[
            gr.Video(label="Annotated Video"),
            gr.Textbox(label="Gemini Full Response"),
            gr.JSON(label="Parsed Output")
        ],
        title="OCR in Video",
        description=(
            "Upload a video for OCR. The model will process the video "
            "and return an annotated version with bounding boxes and detected text."
            "It will also process the text with Gemini LLM for manufacturing, expiry date and MRP predictions."
        ),
        examples=None,  
        allow_flagging="never", 
        live=False,  
    )