
import gradio as gr
from Backend.Fruit_Freshness.Freshness_main import freshness


def image_freshness_interface():
    return gr.Interface(
        fn=freshness,  
        inputs=gr.Image(type="pil", label="Upload an Image"),  # Removed tool argument
        #textbox outputs
        outputs="text",
        title="Banana Freshness Classifier",
        description="Upload an image of a banana to classify its freshness.",
        css="#component-0 { width: 300px; height: 300px; }"  # Keep your CSS for fixe
    )
    
