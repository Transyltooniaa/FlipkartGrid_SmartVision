import gradio as gr
from Backend.OCR.Static.ImageOCR import detect_and_ocr, extract_details_from_validated_output, further_processing,handle_processing

def createStaticOcrInterface():
    with gr.Blocks() as ocr_interface:
        gr.Markdown("# OCR Interface")

        with gr.Tabs():
            # Upload and Detection Tab
            with gr.TabItem("Upload & Detection"):
                with gr.Row():
                    input_image = gr.Image(type="pil", label="Upload Image", height=400, width=400)
                    output_image = gr.Image(label="Image with Bounding Boxes", height=400, width=400)

                btn = gr.Button("Analyze Image & Extract Text")

            # OCR Results Tab
            with gr.TabItem("OCR Results"):
                with gr.Row():
                    extracted_textbox = gr.Textbox(label="Extracted OCR Text", lines=5)
                with gr.Row():
                    refined_textbox = gr.Textbox(label="Refined Text from Gemini", lines=5)
                with gr.Row():
                    validated_textbox = gr.Textbox(label="Validated Output", lines=5)

                # Data table for Manufacturing Date, Expiration Date, and MRP
                with gr.Row():
                    detail_table = gr.Dataframe(
                        headers=["Label", "Value"],
                        value=[["", ""], ["", ""], ["", ""]],  # Initialize with empty values
                        label="Manufacturing, Expiration Dates & MRP",
                        datatype=["str", "str"],
                        interactive=False,
                    )

                further_button = gr.Button("Comprehensive OCR", visible=False)

        # Detect and OCR button click event
        btn.click(
            detect_and_ocr,
            inputs=[input_image],
            outputs=[output_image, extracted_textbox, refined_textbox, validated_textbox]
        )

        # Update the table when validated_textbox changes
        validated_textbox.change(
            lambda validated_output: extract_details_from_validated_output(validated_output),
            inputs=[validated_textbox],
            outputs=[detail_table]
        )

        # Further processing button click event
        further_button.click(
            further_processing,
            inputs=[input_image, extracted_textbox],
            outputs=refined_textbox
        )

        # Monitor validated output to control button visibility
        refined_textbox.change(
            handle_processing,
            inputs=[validated_textbox],
            outputs=[further_button]
        )

        further_button.click(
            lambda: gr.update(visible=False),
            outputs=[validated_textbox]
        )

    return ocr_interface