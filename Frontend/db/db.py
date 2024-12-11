import gradio as gr
import json
from Database.mongo import DatabaseManager
import traceback

db_manager = DatabaseManager()

def create_gradio_interface():

    def safe_execute(func, *args, **kwargs):
        """
        Wrapper to handle exceptions gracefully
        """
        try:
            return func(*args, **kwargs)
        except Exception as e:
            return f"Error occurred: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
    # Brand Record Methods

    def get_brand_records(filter_brand='', sort_by='timestamp', ascending=False):
        filter_criteria = {}
        if filter_brand:
            filter_criteria['brand'] = {'$regex': filter_brand, '$options': 'i'}
        
        return safe_execute(db_manager.get_brand_records, filter_criteria, sort_by, ascending)
    
    def get_freshness_records(filter_freshness='', sort_by='timestamp', ascending=False):
        filter_criteria = {}
        if filter_freshness:
            filter_criteria['Eatable'] = {'$regex': filter_freshness, '$options': 'i'}
            
        return safe_execute(db_manager.get_freshness_records, filter_criteria, sort_by, ascending)
            

    # Analysis Methods
    def analyze_brand_trends(time_period):
        return safe_execute(db_manager.analyze_brand_trends, int(time_period))

    # Advanced Search
    def perform_advanced_search(collection, search_criteria, projection):
        try:
            # Parse search criteria and projection as JSON
            search_dict = json.loads(search_criteria) if search_criteria else {}
            proj_dict = json.loads(projection) if projection else {}
            
            return safe_execute(db_manager.advanced_search, collection, search_dict, proj_dict)
        except json.JSONDecodeError:
            return "Invalid JSON format for search criteria or projection"

    # Export Methods
    def export_collection(collection_name):
        return safe_execute(db_manager.export_collection_to_json, collection_name)

    # Gradio Interface Construction
    with gr.Blocks(title="Database Management System", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 📊 Advanced Database Management System")
        
        with gr.Tab("Brand Records"):
            with gr.Row():

                with gr.Column():
                    gr.Markdown("## 📋 Get Brand Records")
                    filter_brand = gr.Textbox(label="Filter by Brand (optional)")
                    sort_by = gr.Dropdown(
                        ["timestamp", "count"], 
                        label="Sort By", 
                        value="timestamp"
                    )
                    ascending = gr.Checkbox(label="Ascending Order")
                    get_brands_btn = gr.Button("Retrieve Brand Records")
                    brand_records_output = gr.JSON(label="Brand Records")

          
            get_brands_btn.click(
                get_brand_records, 
                inputs=[filter_brand, sort_by, ascending], 
                outputs=brand_records_output
            )

        

        with gr.Tab("Freshness Records"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("## 🍎 Get Freshness Record")
                    filter_freshness = gr.Textbox(label="Filter by Freshness (optional)")
                    sort_by = gr.Dropdown(
                        ["timestamp", "count"], 
                        label="Sort By", 
                        value="timestamp"
                    )
                    ascending = gr.Checkbox(label="Ascending Order")
                    get_freshness_btn = gr.Button("Retrieve Freshness Records")
                    freshness_records_output = gr.JSON(label="Freshness Records")

            get_freshness_btn.click(
                get_freshness_records, 
                inputs=[filter_freshness, sort_by, ascending], 
                outputs=freshness_records_output
            )
            
                

        with gr.Tab("Analytics & Search"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("## 📈 Brand Trend Analysis")
                    time_period = gr.Number(label="Time Period (Days)", value=30)
                    analyze_trends_btn = gr.Button("Analyze Brand Trends")
                    trends_output = gr.JSON(label="Brand Trends")

                with gr.Column():
                    gr.Markdown("## 🔬 Advanced Search")
                    collection_dropdown = gr.Dropdown(
                        ["brand", "ocr", "freshness"], 
                        label="Collection"
                    )
                    search_criteria = gr.Textbox(
                        label="Search Criteria (JSON)", 
                        placeholder='e.g., {"count": {"$gt": 100}}'
                    )
                    projection = gr.Textbox(
                        label="Projection (JSON)", 
                        placeholder='e.g., {"brand": 1, "count": 1}'
                    )
                    search_btn = gr.Button("Perform Search")
                    search_output = gr.JSON(label="Search Results")

            # Analytics Interactions
            analyze_trends_btn.click(
                analyze_brand_trends, 
                inputs=time_period, 
                outputs=trends_output
            )
            search_btn.click(
                perform_advanced_search, 
                inputs=[collection_dropdown, search_criteria, projection], 
                outputs=search_output
            )

        with gr.Tab("Export"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("## 📤 Export Collections")
                    export_collection_dropdown = gr.Dropdown(
                        ["brand", "ocr", "freshness"], 
                        label="Select Collection to Export"
                    )
                    export_btn = gr.Button("Export to JSON")
                    export_output = gr.Textbox(label="Export Result")

            # Export Interactions
            export_btn.click(
                export_collection, 
                inputs=export_collection_dropdown, 
                outputs=export_output
            )

    return demo


db_interface = create_gradio_interface()