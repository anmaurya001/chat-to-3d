import gradio as gr

def handle_action(action_name, item_title):
    message = f"'{action_name}' clicked for item: '{item_title}'"
    print(message)
    return message

def open_image_settings(image_path, title):
    """Open settings modal for the clicked image"""
    # Show overlay and modal
    overlay_html = "<div id='modal-overlay' style='display:block; position:fixed; top:0; left:0; width:100vw; height:100vh; background:rgba(0,0,0,0.4); z-index:999;'></div>"
    return f"### {title}", image_path, True, overlay_html

def close_modal():
    """Close the settings modal"""
    # Hide overlay and modal
    overlay_html = "<div id='modal-overlay' style='display:none; position:fixed; top:0; left:0; width:100vw; height:100vh; background:rgba(0,0,0,0.4); z-index:999;'></div>"
    return None, None, False, overlay_html

def save_settings(image_path, title):
    """Save the image settings"""
    return f"Settings saved for {title}", False

def create_image_card(image_path, title, output_widget, modal_image_title, modal_image, modal_visible, settings_modal, overlay):
    # Create a card container
    with gr.Column():
        # Title at the top
        gr.Markdown(f"### {title}")
        
        # Image with fixed height
        image_component = gr.Image(
            image_path,
            show_label=False,
            show_download_button=False,
            interactive=False,
            height=250,
            sources=[],
            show_fullscreen_button=False
        )
        
        # Button row below image - all in one row for horizontal layout
        with gr.Row():
            gr.HTML("<div style='flex-grow: 1;'></div>")  # Left spacer
            refresh_btn = gr.Button("🔄", size="sm", min_width=10)
            edit_btn = gr.Button("✏️", size="sm", min_width=10)
            delete_btn = gr.Button("🗑️", size="sm", min_width=10)
            to_3d_btn = gr.Button("3D", size="sm", min_width=10)
            gr.HTML("<div style='flex-grow: 1;'></div>")  # Right spacer
    
    # Make the image clickable to open modal
    image_component.select(
        fn=lambda: open_image_settings(image_path, title),
        outputs=[modal_image_title, modal_image, modal_visible, overlay]
    ).then(
        fn=lambda: gr.update(visible=True),
        outputs=settings_modal
    )
            
    refresh_btn.click(lambda t=title: handle_action("Refresh", t), outputs=output_widget)
    edit_btn.click(lambda t=title: handle_action("Edit", t), outputs=output_widget)
    delete_btn.click(lambda t=title: handle_action("Delete", t), outputs=output_widget)
    to_3d_btn.click(lambda t=title: handle_action("To 3D", t), outputs=output_widget)


with gr.Blocks() as demo:
    gr.Markdown("# My Object Gallery")
    output_text = gr.Textbox(label="Last Action")
    
    # Modal state
    modal_visible = gr.State(False)
    
    # Overlay for modal
    overlay = gr.HTML("""
    <div id='modal-overlay' style='display:none; position:fixed; top:0; left:0; width:100vw; height:100vh; background:rgba(0,0,0,0.4); z-index:999;'></div>
    """)
    
    # Settings modal - make visibility reactive
    with gr.Group(elem_id="settings-modal") as settings_modal:
        #gr.Markdown("## Image Preview & Settings")
        modal_image_title = gr.Markdown("### Image Preview")
        close_btn = gr.Button("X", elem_id="close-modal-btn", size="sm")
        with gr.Tabs():
            with gr.TabItem("2D"):
                with gr.Row():
                    with gr.Column():
                        #modal_image_title = gr.Markdown("### Image Preview")
                        modal_image = gr.Image(
                            label="Selected Image",
                            interactive=False,                # Allow interaction (zoom in fullscreen)
                            show_fullscreen_button=False,      # Hide fullscreen (zoom) button
                            height=350,                      # Set a fixed height (adjust as needed)
                            sources=[]                     # Optional: set a fixed width
                        )
            
            with gr.TabItem("3D"):
                gr.Markdown("3D preview coming soon!")

        # Close button event
        close_btn.click(
            fn=close_modal,
            inputs=[],
            outputs=[modal_image_title, modal_image, modal_visible, overlay]
        ).then(
            fn=lambda: gr.update(visible=False),
            outputs=settings_modal
        )

    # Custom CSS for floating close button, modal overlay, and fullscreen image centering
    gr.HTML("""
    <style>
    #settings-modal {
        position: fixed !important;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        z-index: 1000;
        background: white;
        border-radius: 12px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.25);
        width: 600px;
        max-width: 95vw;
        box-sizing: border-box;
        padding: 24px 16px 16px 16px;
        max-height: 90vh;
        overflow-y: auto;
    }
    #close-modal-btn {
        position: absolute;
        top: 18px;
        right: 30px;
        z-index: 1001;
        background: transparent;
        border: none;
        font-size: 1.5em;
        cursor: pointer;
        box-shadow: none;
    }
    </style>
    """)
    
    # Initially hide the modal
    settings_modal.visible = False
    
    gallery_data = [
        {"title": "Beach Umbrella", "path": "images/umbrella.jpeg"},
        {"title": "Mountain View", "path": "images/umbrella.jpeg"},
        {"title": "Forest Trail", "path": "images/umbrella.jpeg"},
        {"title": "Laptop & Coffee", "path": "images/umbrella.jpeg"},
        {"title": "Cityscape", "path": "images/umbrella.jpeg"},
        {"title": "Ocean Waves", "path": "images/umbrella.jpeg"},
    ]
    
    # Use responsive grid layout
    with gr.Column():
        for i in range(0, len(gallery_data), 3):
            with gr.Row():
                row_items = gallery_data[i : i + 3]
                for item in row_items:
                    create_image_card(item["path"], item["title"], output_text, modal_image_title, modal_image, modal_visible, settings_modal, overlay)

demo.launch(debug=True)