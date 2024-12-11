import gradio as gr
from Backend.BrandRecognition.Static.Brand_Count_Img import detect_grocery_items
from Backend.BrandRecognition.Static.Brand_Count_Img import batch_detect_grocery_items

## Layout for Image interface
def create_image_interface():
    
    """
    Create an interface for object detection in an image.

    The interface allows users to upload an image, and the model will return an annotated image, item quantities, and average confidence scores.

    Parameters
    ----------
    None

    Returns
    -------
    gr.Interface
        An Interface object that can be launched to accept user input.

    """
    return gr.Interface(
        fn=detect_grocery_items,
        inputs=gr.Image(label="Upload Image", height=400, width=400),
        outputs=[
            gr.Image(label="Image with Bounding Boxes", height=400, width=400),
            gr.Dataframe(headers=["Item", "Quantity", "Avg Confidence"], label="Detected Items and Quantities", elem_id="summary_table"),
            gr.Textbox(label="Status", elem_id="status_message")
        ],
        title="Grocery Item Detection in an Image",
        description="Upload an image for object detection. The model will return an annotated image, item quantities, and average confidence scores.",
        css=".gr-table { font-size: 16px; text-align: left; width: 50%; margin: auto; } #summary_table { margin-top: 20px; }"
    )
