import gradio as gr
import os

def open_image_settings(image_path, title):
    """Open settings modal for the clicked image."""
    overlay_html = "<div id='modal-overlay' style='display:block; position:fixed; top:0; left:0; width:100vw; height:100vh; background:rgba(0,0,0,0.4); z-index:999;'></div>"
    return f"### {title}", image_path, True, overlay_html

def close_modal():
    """Close the settings modal."""
    overlay_html = "<div id='modal-overlay' style='display:none; position:fixed; top:0; left:0; width:100vw; height:100vh; background:rgba(0,0,0,0.4); z-index:999;'></div>"
    return None, None, False, overlay_html

def save_settings(image_path, title):
    """Save the image settings."""
    return f"Settings saved for {title}", False

def create_modal():
    """Create the settings modal UI block and return its components."""
    with gr.Group(elem_id="settings-modal") as settings_modal:
        modal_image_title = gr.Markdown("### Image Preview")
        close_btn = gr.Button("X", elem_id="close-modal-btn", size="sm")
        with gr.Tabs():
            with gr.TabItem("2D"):
                with gr.Row():
                    with gr.Column():
                        modal_image = gr.Image(
                            label="Selected Image",
                            interactive=False,
                            show_fullscreen_button=False,
                            height=350,
                            sources=[]
                        )
            with gr.TabItem("3D"):
                glb_path = os.path.join("assets", "models", "Beach_Umbrella.glb")
                modal_3d = gr.Model3D(
                    value=glb_path,
                    label="3D Model Preview",
                    clear_color=[0.0, 0.0, 0.0, 0.0],
                    height=400
                )
    return settings_modal, modal_image_title, close_btn, modal_image, modal_3d

def create_edit_modal():
    """Create a modal for editing image description."""
    with gr.Group(elem_id="edit-modal") as edit_modal:
        edit_image_title = gr.Markdown("### Edit Image")
        edit_description = gr.Textbox(
            label="Description", 
            lines=5, 
            placeholder="Enter a detailed description for the image generation..."
        )
        with gr.Row():
            cancel_btn = gr.Button("Cancel", elem_id="cancel-edit-btn", variant="secondary")
            update_btn = gr.Button("Update & Generate", elem_id="update-edit-btn", variant="primary")
    return edit_modal, edit_image_title, edit_description, cancel_btn, update_btn 