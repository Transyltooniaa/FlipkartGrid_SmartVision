import gradio as gr
from Backend.BrandRecognition.Dynamic.Brand_Count_Vid import annotate_video


def create_video_interface():
    """
    Create an interface for object detection in a video.

    The interface allows users to upload a video, and the model will return an annotated video with bounding boxes and item quantities.

    Returns:
        gr.Interface: A Gradio interface object.
    """
    return gr.Interface(
        fn=annotate_video, 
        inputs=gr.Video(label="Upload Video"), 
        outputs=[
            gr.Video(label="Annotated Video"),  # Display the annotated video
            gr.Dataframe(headers=["Item", "Quantity"], label="Detected Items and Quantities"),  # Display detected items
            gr.Textbox(label="Status")  # Display status message
        ],
        title="Grocery Item Detection in Video",
        description=(
            "Upload a video for grocery item detection. The model will process the video "
            "and return an annotated version with bounding boxes and detected item quantities."
        ),
        examples=None,  
        allow_flagging="never", 
        live=False,  
    )

