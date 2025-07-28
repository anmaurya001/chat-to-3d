import gradio as gr
from components.image_card import create_image_card
from components.modal import create_modal, open_image_settings, close_modal
from utils import handle_action
import os

# Inject custom CSS/JS
with open(os.path.join("static", "custom.css")) as f:
    custom_css = f.read()
with open(os.path.join("static", "custom.js")) as f:
    custom_js = f.read()

with gr.Blocks() as demo:
    gr.Markdown("# My Object Gallery")
    gr.HTML(f"<style>{custom_css}</style>")
    gr.HTML(f"<script>{custom_js}</script>")

    output_text = gr.Textbox(label="Last Action")
    modal_visible = gr.State(False)
    overlay = gr.HTML("""
    <div id='modal-overlay' style='display:none; position:fixed; top:0; left:0; width:100vw; height:100vh; background:rgba(0,0,0,0.4); z-index:999;'></div>
    """)

    # Modal UI
    settings_modal, modal_image_title, close_btn, modal_image = create_modal()
    settings_modal.visible = False

    # Close button event
    close_btn.click(
        fn=close_modal,
        inputs=[],
        outputs=[modal_image_title, modal_image, modal_visible, overlay]
    ).then(
        fn=lambda: gr.update(visible=False),
        outputs=settings_modal
    )

    gallery_data = [
        {"title": "Beach Umbrella", "path": "images/umbrella.jpeg"},
        {"title": "Mountain View", "path": "images/umbrella.jpeg"},
        {"title": "Forest Trail", "path": "images/umbrella.jpeg"},
        {"title": "Laptop & Coffee", "path": "images/umbrella.jpeg"},
        {"title": "Cityscape", "path": "images/umbrella.jpeg"},
        {"title": "Ocean Waves", "path": "images/umbrella.jpeg"},
    ]

    # Responsive grid layout
    with gr.Column():
        for i in range(0, len(gallery_data), 3):
            with gr.Row():
                row_items = gallery_data[i : i + 3]
                for item in row_items:
                    card = create_image_card(
                        item["path"], item["title"], output_text,
                        modal_image_title, modal_image, modal_visible, settings_modal, overlay
                    )
                    # Image click opens modal
                    card["image_component"].select(
                        fn=lambda p=item["path"], t=item["title"]: open_image_settings(p, t),
                        outputs=[modal_image_title, modal_image, modal_visible, overlay]
                    ).then(
                        fn=lambda: gr.update(visible=True),
                        outputs=settings_modal
                    )
                    # Action buttons
                    card["refresh_btn"].click(lambda t=item["title"]: handle_action("Refresh", t), outputs=output_text)
                    card["edit_btn"].click(lambda t=item["title"]: handle_action("Edit", t), outputs=output_text)
                    card["delete_btn"].click(lambda t=item["title"]: handle_action("Delete", t), outputs=output_text)
                    card["to_3d_btn"].click(lambda t=item["title"]: handle_action("To 3D", t), outputs=output_text)

demo.launch(debug=True) 