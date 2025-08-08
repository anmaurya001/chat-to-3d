"""Blender export component for exporting 3D objects to Blender."""

import gradio as gr

def create_blender_export_section():
    """Create the Blender export section interface."""
    
    with gr.Column(elem_classes=["blender-export-section"]) as export_section:
        gr.Markdown("### Export 3D objects for Blender", elem_classes=["export-header"])
        
        # Instructional link
        with gr.Row():
            gr.HTML(
                value="<a href='#' style='color: #0066cc; text-decoration: none; display: flex; align-items: center; gap: 4px;'>"
                      "Learn how to load them into Blender's library"
                      "<svg width='12' height='12' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2'>"
                      "<path d='M7 7h10v10'></path>"
                      "<path d='M7 17 17 7'></path>"
                      "</svg>"
                      "</a>",
                elem_classes=["export-link"]
            )
        
        # Main content area with placeholder
        with gr.Column(elem_classes=["export-content"]) as content_container:
            gr.HTML(
                value="<div style='border: 2px solid #e0e0e0; border-radius: 8px; background-color: #f8f8f8; padding: 40px; text-align: center;'>"
                      "<div style='display: flex; justify-content: center; align-items: center; margin-bottom: 20px;'>"
                      "<svg width='48' height='48' viewBox='0 0 24 24' fill='none' xmlns='http://www.w3.org/2000/svg'>"
                      "<circle cx='12' cy='12' r='10' stroke='#ccc' stroke-width='2' fill='none'/>"
                      "<path d='M12 16v-4' stroke='#ccc' stroke-width='2' stroke-linecap='round'/>"
                      "<path d='M12 8h.01' stroke='#ccc' stroke-width='2' stroke-linecap='round'/>"
                      "</svg>"
                      "</div>"
                      "<div style='color: #666; font-size: 16px; line-height: 1.5;'>"
                      "Convert objects from your object gallery to 3D to export them for Blender"
                      "</div>"
                      "</div>",
                elem_classes=["placeholder-text"]
            )
    
    return {
        "section": export_section,
        "content": content_container,
    } 