"""Image gallery component for displaying generated objects."""

import gradio as gr


def create_image_gallery():
    """Create the object gallery interface."""
    
    with gr.Column(elem_classes=["gallery-section"]) as gallery_section:
        gr.Markdown("### Object Gallery", elem_classes=["gallery-header"])
        
        # Gallery grid - will be populated dynamically
        gallery_data = gr.State([])
        
        # Gallery container
        with gr.Column(elem_classes=["gallery-container"]) as gallery_container:
            gr.HTML("<div class='gallery-placeholder'>Objects will appear here as you create your scene...</div>")
        
        # Action buttons
        with gr.Row():
            generate_prompts_btn = gr.Button("Generate 3D Prompts", variant="primary")
            clear_scene_btn = gr.Button("Clear Scene", variant="secondary")
    
    return {
        "section": gallery_section,
        "container": gallery_container,
        "data": gallery_data,
        "generate_btn": generate_prompts_btn,
        "clear_btn": clear_scene_btn
    } 