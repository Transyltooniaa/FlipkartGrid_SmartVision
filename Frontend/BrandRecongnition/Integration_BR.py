import gradio as gr
from Frontend.BrandRecongnition.Image import create_image_interface
from Frontend.BrandRecongnition.Video import create_video_interface

def create_brand_recog_interface():
    """
    Create an interface for brand recognition in images and videos.

    The interface uses gradio Tabs to switch between an image and video interface.
    The image interface accepts an image and returns an annotated image, item quantities, and average confidence scores.
    The video interface accepts a video and returns an annotated video with bounding boxes and item quantities.

    Returns:
        gr.Interface
            An Interface object that can be launched to accept user input.
    """
    with gr.Blocks() as demo:
        gr.Markdown("# Flipkart Grid Robotics Track - Brand Recognition Interface")

        with gr.Tabs():
            with gr.Tab("Image"):
                create_image_interface()
            with gr.Tab("Video"):
                create_video_interface()
    return demo

Brand_recog = create_brand_recog_interface()
