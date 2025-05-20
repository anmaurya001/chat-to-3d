import gradio as gr
from gradio_litmodel3d import LitModel3D
import os

def create_glb_preview():
    """
    Creates a GLB preview component using gradio_litmodel3d.
    
    Returns:
        tuple: (model_output, download_button) - The 3D model viewer and download button components
    """
    with gr.Row():
        with gr.Column(scale=4):
            model_output = LitModel3D(
                label="3D Model Preview",
                exposure=10.0,
                height=400
            )
        # with gr.Column(scale=1):
        #     download_btn = gr.DownloadButton(
        #         label="Download GLB",
        #         interactive=False
        #     )
    
    return model_output

def update_preview(glb_path: str) -> tuple:
    """
    Updates the GLB preview with a new model.
    
    Args:
        glb_path (str): Path to the GLB file
        
    Returns:
        tuple: (glb_path, True) - The GLB path 
    """
    if not os.path.exists(glb_path):
        return None
    return glb_path

def clear_preview() -> tuple:
    """
    Clears the GLB preview.
    
    Returns:
        tuple: (None) - Clears the preview
    """
    return None