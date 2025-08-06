
"""Main Chat-to-3D Gradio Application.

This is the entry point for the Chat-to-3D application that integrates:
- LLM Agent for scene planning
- Object gallery UI 
- 3D generation pipeline
"""

import gradio as gr
import os
import base64
from components.chat_interface import create_chat_interface, handle_scene_description
from components.image_gallery import create_image_gallery
from components.status_panel import create_status_panel
from services.agent_service import AgentService
from services.scene_service import SceneService
import config

# Load custom CSS and JS
try:
    with open(os.path.join("static", "css", "custom.css")) as f:
        custom_css = f.read()
    with open(os.path.join("static", "js", "custom.js")) as f:
        custom_js = f.read()
except FileNotFoundError:
    print("Custom CSS and JS not found")
    custom_css = ""
    custom_js = ""

# Load NVIDIA logo
try:
    with open("./static/images/nvidia_logo.png", "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode()
    
    nvidia_html = f"""<div class='nvidia-logo-container'> 
            <img src='data:image/png;base64,{encoded}' alt='NVIDIA Logo' class='nvidia-logo' width='24' height='24'> 
            <span class='nvidia-text'></span> 
            </div>"""
except FileNotFoundError:
    # Fallback if logo not found
    nvidia_html = """<div class='nvidia-logo-container'> 
            <span class='nvidia-text'>Chat-to-3D</span> 
            </div>"""


def create_app():
    """Create and configure the main Gradio application."""
    
    # Initialize services
    agent_service = AgentService()
    scene_service = SceneService()
    
    with gr.Blocks(
        title="Chat-to-3D", 
        # css_paths=["static/css/custom.css"]
    ) as app:
        
        # Global state
        scene_state = gr.State({"objects": [], "description": ""})
        status_state = gr.State("System ready")
        
        # Inject custom CSS and JS
        gr.HTML(f"<style>{custom_css}</style>")
        gr.HTML(f"<script>{custom_js}</script>")
        
        # Header with NVIDIA branding and status
        with gr.Row(elem_classes=["header-section"]):
            with gr.Column(scale=3):
                with gr.Row():
                    gr.HTML(nvidia_html)
            with gr.Column(scale=1):
                with gr.Row():
                    gr.HTML("""
                    <div class="status-section">
                        <span class="status-text">LLM Agent is online and running</span>
                        <button class="status-button" id="refresh-status">
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8"></path>
                                <path d="M21 3v5h-5"></path>
                                <path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16"></path>
                                <path d="M3 21v-5h5"></path>
                            </svg>
                        </button>
                    </div>
                    """)
                    toggle_btn = gr.Button("📊", elem_classes=["toggle-status-btn"], size="sm")
        
        # Main layout
        with gr.Row():
            # Left: Chat and Gallery
            with gr.Column(scale=4, elem_classes=["main-content"]):
                # Chat interface
                chat_components = create_chat_interface()
                
                # Object gallery
                gallery_components = create_image_gallery()
                
            # Right: Status panel
            with gr.Column(scale=1, elem_classes=["right-panel"], visible=False) as right_panel:
                status_components = create_status_panel()
        
        # Wire up the event handlers
        def process_scene_description(scene_description, gallery_data):
            """Process scene description and generate objects, then update gallery."""
            if not scene_description.strip():
                return "Please enter a scene description.", gallery_data
            
            message, new_gallery_data = handle_scene_description(scene_description, agent_service, gallery_data)
            
            return message, new_gallery_data
        
        # Connect send button to process scene description
        chat_components["send_btn"].click(
            fn=process_scene_description,
            inputs=[chat_components["input"], gallery_components["data"]],
            outputs=[chat_components["input"], gallery_components["data"]]
        ).then(
            fn=gallery_components["shift_card_ui"],
            inputs=[gallery_components["data"]],
            outputs=gallery_components["get_all_card_outputs"]()
        )
        
        # Connect Enter key for scene input
        chat_components["input"].submit(
            fn=process_scene_description,
            inputs=[chat_components["input"], gallery_components["data"]],
            outputs=[chat_components["input"], gallery_components["data"]]
        ).then(
            fn=gallery_components["shift_card_ui"],
            inputs=[gallery_components["data"]],
            outputs=gallery_components["get_all_card_outputs"]()
        )
        
        # Connect toggle button to show/hide right panel
        toggle_btn.click(
            fn=lambda: gr.update(visible=True),
            outputs=[right_panel]
        )
        
        # Connect close button to hide right panel
        status_components["close_btn"].click(
            fn=lambda: gr.update(visible=False),
            outputs=[right_panel]
        )
        
    return app


if __name__ == "__main__":
    app = create_app()
    app.launch(debug=True) 