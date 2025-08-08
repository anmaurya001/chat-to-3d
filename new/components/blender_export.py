"""Blender export component for exporting 3D objects to Blender."""

import gradio as gr
import os
import base64
from PIL import Image
import io

def create_blender_export_section():
    """Create the Blender export section interface."""
    
    with gr.Column(elem_classes=["blender-export-section"]) as export_section:
        gr.Markdown("### Export your collection for Blender", elem_classes=["export-header"])
        
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
        
        # Main content area with dynamic content
        with gr.Column(elem_classes=["export-content"]) as content_container:
            # Placeholder for empty state (shown when no 3D assets)
            placeholder = gr.HTML(
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
                elem_classes=["placeholder-text"],
                visible=True
            )
            
            # Export content (shown when 3D assets are available)
            with gr.Column(visible=False, elem_classes=["export-content-active"]) as export_content_active:
                # Count display
                count_display = gr.HTML(
                    value="<div style='color: #666; font-size: 14px; margin-bottom: 16px;'>0 objects ready to export</div>",
                    elem_classes=["export-count"]
                )
                
                # Thumbnails container
                thumbnails_container = gr.HTML(
                    value="<div style='display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 16px;'></div>",
                    elem_classes=["thumbnails-container"]
                )
                
                # Export button
                export_btn = gr.Button(
                    "Export objects to file",
                    elem_classes=["export-btn"],
                    interactive=False
                )
    
    return {
        "section": export_section,
        "content": content_container,
        "count_display": count_display,
        "thumbnails_container": thumbnails_container,
        "export_btn": export_btn,
        "placeholder": placeholder,
        "export_content_active": export_content_active,
    }

def update_export_section(gallery_data):
    """Update the export section based on gallery data with 3D assets."""
    if not gallery_data:
        return (
            gr.update(value="<div style='color: #666; font-size: 14px; margin-bottom: 16px;'>0 objects ready to export</div>"),
            gr.update(value="<div style='display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 16px;'></div>"),
            gr.update(interactive=False),
            gr.update(visible=True),
            gr.update(visible=False)
        )
    
    # Filter for objects that have 3D models
    exportable_objects = []
    for obj in gallery_data:
        if obj.get("glb_path") and os.path.exists(obj["glb_path"]):
            exportable_objects.append(obj)
    
    count = len(exportable_objects)
    
    if count == 0:
        # No 3D assets to export - show placeholder, hide export content
        return (
            gr.update(value="<div style='color: #666; font-size: 14px; margin-bottom: 16px;'>0 objects ready to export</div>"),
            gr.update(value="<div style='display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 16px;'></div>"),
            gr.update(interactive=False),
            gr.update(visible=True),
            gr.update(visible=False)
        )
    
    # Generate thumbnails HTML
    thumbnails_html = "<div style='display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 16px;'>"
    
    for obj in exportable_objects:
        try:
            # Create thumbnail from original image
            if obj.get("path") and os.path.exists(obj["path"]):
                # Read and resize image for thumbnail
                with Image.open(obj["path"]) as img:
                    # Resize to thumbnail size (64x64)
                    img.thumbnail((64, 64), Image.Resampling.LANCZOS)
                    
                    # Convert to base64 for inline display
                    buffer = io.BytesIO()
                    img.save(buffer, format='PNG')
                    img_base64 = base64.b64encode(buffer.getvalue()).decode()
                    
                    # Create thumbnail HTML
                    thumbnails_html += f"""
                    <div style='display: inline-block; text-align: center;'>
                        <img src='data:image/png;base64,{img_base64}' 
                             alt='{obj["title"]}' 
                             style='width: 64px; height: 64px; object-fit: cover; border-radius: 8px; border: 2px solid #e0e0e0;'
                             title='{obj["title"]}'>
                    </div>
                    """
            else:
                # Fallback if image not found
                thumbnails_html += f"""
                <div style='display: inline-block; text-align: center;'>
                    <div style='width: 64px; height: 64px; background-color: #f0f0f0; border-radius: 8px; border: 2px solid #e0e0e0; display: flex; align-items: center; justify-content: center; color: #999; font-size: 12px;'>
                        {obj["title"][:8]}...
                    </div>
                </div>
                """
        except Exception as e:
            print(f"Error creating thumbnail for {obj['title']}: {e}")
            # Fallback thumbnail
            thumbnails_html += f"""
            <div style='display: inline-block; text-align: center;'>
                <div style='width: 64px; height: 64px; background-color: #f0f0f0; border-radius: 8px; border: 2px solid #e0e0e0; display: flex; align-items: center; justify-content: center; color: #999; font-size: 12px;'>
                    {obj["title"][:8]}...
                </div>
            </div>
            """
    
    thumbnails_html += "</div>"
    
    # Has 3D assets - hide placeholder, show export content
    return (
        gr.update(value=f"<div style='color: #666; font-size: 14px; margin-bottom: 16px;'>{count} objects ready to export</div>"),
        gr.update(value=thumbnails_html),
        gr.update(interactive=True),
        gr.update(visible=False),
        gr.update(visible=True)
    )

def export_3d_assets(gallery_data):
    """Export all 3D assets to a zip file."""
    if not gallery_data:
        return "No objects to export."
    
    # Filter for objects that have 3D models
    exportable_objects = []
    for obj in gallery_data:
        if obj.get("glb_path") and os.path.exists(obj["glb_path"]):
            exportable_objects.append(obj)
    
    if not exportable_objects:
        return "No 3D objects ready for export."
    
    try:
        import zipfile
        import tempfile
        
        # Create temporary zip file
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp_file:
            with zipfile.ZipFile(tmp_file.name, 'w') as zipf:
                for obj in exportable_objects:
                    glb_path = obj["glb_path"]
                    # Use object title as filename in zip
                    zip_filename = f"{obj['title'].replace(' ', '_')}.glb"
                    zipf.write(glb_path, zip_filename)
        
        return f"Successfully exported {len(exportable_objects)} objects to {tmp_file.name}"
    except Exception as e:
        return f"Error exporting objects: {str(e)}" 