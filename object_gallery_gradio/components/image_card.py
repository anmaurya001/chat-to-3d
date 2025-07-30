import gradio as gr

def create_image_card(image_path, title, output_widget, modal_image_title, modal_image, modal_visible, settings_modal, overlay):
    """Create a single image card with action buttons and modal trigger."""
    with gr.Column(elem_classes=["card-content"]):
        title_component = gr.Markdown(f"### {title if title else ''}", elem_classes=["card-title"])
        image_component = gr.Image(
            image_path if image_path else None,
            show_label=False,
            show_download_button=False,
            interactive=False,
            height=180,
            sources=[],
            show_fullscreen_button=False,
            elem_classes=["card-image"]
        )
        with gr.Row(elem_classes=["card-actions"]):
            gr.HTML("<div style='flex-grow: 1;'></div>")
            refresh_btn = gr.Button("🔄", size="sm", min_width=10, elem_classes=["action-btn"])
            edit_btn = gr.Button("✏️", size="sm", min_width=10, elem_classes=["action-btn"])
            delete_btn = gr.Button("🗑️", size="sm", min_width=10, elem_classes=["action-btn"])
            to_3d_btn = gr.Button("3D", size="sm", min_width=10, elem_classes=["action-btn"])
            gr.HTML("<div style='flex-grow: 1;'></div>")

    # Return the image component and buttons for event binding in app.py
    return {
        "title_component": title_component,
        "image_component": image_component,
        "refresh_btn": refresh_btn,
        "edit_btn": edit_btn,
        "delete_btn": delete_btn,
        "to_3d_btn": to_3d_btn
    } 