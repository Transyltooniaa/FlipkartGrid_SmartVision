import gradio as gr
from Frontend.BrandRecongnition.Integration_BR import Brand_recog
from Frontend.FruitFreshness.Integration_FR import Fruit
from Frontend.db.db import db_interface
from Frontend.OCR.Integration_OCR import Ocr

# Define custom CSS to enlarge elements
custom_css = """
* {
    font-size: 18px !important;  /* Increase font size globally */
}
button {
    font-size: 20px !important; /* Enlarge button text */
}
.gr-container {
    transform: scale(1.5);      /* Scale entire interface */
}
"""

def create_tabbed_interface():
    return gr.TabbedInterface(
        [Brand_recog, Fruit, Ocr, db_interface],
        ["Brand Recognition", "Freshness Detection", "OCR Interface", "Database Records"],
    )

with gr.Blocks(css=custom_css) as tabbed_interface:
    create_tabbed_interface()

if __name__ == "__main__":
    tabbed_interface.queue()
    tabbed_interface.launch(debug=True)
