import gradio as gr
from Frontend.FruitFreshness.Image import image_freshness_interface


def create_fruit_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# Flipkart Grid Robotics Track - Fruits Interface")
        with gr.Tabs():
            with gr.Tab("Image Freshness"):
                image_freshness_interface()  # Call the image freshness interface
    return demo


Fruit = create_fruit_interface()
