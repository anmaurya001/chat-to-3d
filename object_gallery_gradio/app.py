import gradio as gr
from components.image_card import create_image_card
from components.modal import create_modal, open_image_settings, close_modal, create_edit_modal
from utils import handle_action
import os

# Inject custom CSS/JS
with open(os.path.join("static", "custom.css")) as f:
    custom_css = f.read()
with open(os.path.join("static", "custom.js")) as f:
    custom_js = f.read()

MAX_CARDS = 20

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
    settings_modal, modal_image_title, close_btn, modal_image, modal_3d = create_modal()
    settings_modal.visible = False

    # Edit Modal UI
    edit_modal, edit_image_title, edit_description, cancel_edit_btn, update_edit_btn = create_edit_modal()
    edit_modal.visible = False
    edit_current_index = gr.State(None)

    # Initial gallery data
    initial_gallery_data = [
        {"title": "Beach Umbrella", "path": "images/umbrella.png", "description": "A colorful beach umbrella on a white background."},
        {"title": "Mountain View", "path": "images/umbrella.png", "description": "A scenic mountain view with clear skies."},
        {"title": "Forest Trail", "path": "images/umbrella.png", "description": "A peaceful trail through a dense forest."},
        {"title": "Laptop & Coffee", "path": "images/umbrella.png", "description": "A laptop and coffee cup on a desk."},
        {"title": "Cityscape", "path": "images/umbrella.png", "description": "A vibrant cityscape at sunset."},
        {"title": "Ocean Waves", "path": "images/umbrella.png", "description": "Rolling ocean waves under a blue sky."},
    ]
    gallery_data = gr.State(initial_gallery_data)

    card_components = []
    card_containers = []

    # Create cards in a responsive grid using Gradio's native layout
    cards_per_row = 4  # Adjust this number to control cards per row
    
    for row_start in range(0, MAX_CARDS, cards_per_row):
        with gr.Row() as row:
            for col_idx in range(cards_per_row):
                card_idx = row_start + col_idx
                if card_idx < MAX_CARDS:
                    # Use scale=1 to fill available space, but with min_width to maintain consistency
                    with gr.Column(scale=1, min_width=180, visible=(card_idx < len(initial_gallery_data)), elem_classes=["gallery-card"]) as card_container:
                        if card_idx < len(initial_gallery_data):
                            item = initial_gallery_data[card_idx]
                            card = create_image_card(
                                item["path"], item["title"], output_text,
                                modal_image_title, modal_image, modal_visible, settings_modal, overlay
                            )
                        else:
                            card = create_image_card("", "", output_text,
                                modal_image_title, modal_image, modal_visible, settings_modal, overlay)

                        # Image click opens modal
                        if card_idx < len(initial_gallery_data):
                            card["image_component"].select(
                                fn=lambda p=item["path"], t=item["title"]: open_image_settings(p, t),
                                outputs=[modal_image_title, modal_image, modal_visible, overlay]
                            ).then(
                                fn=lambda: gr.update(visible=True),
                                outputs=settings_modal
                            )
                        
                        card_components.append(card)
                        card_containers.append(card_container)

    # --- Utility: get all card outputs ---
    def get_all_card_outputs():
        outputs = []
        for card in card_components:
            outputs.extend([card["title_component"], card["image_component"]])
        outputs.extend(card_containers)
        return outputs

    # --- Logic ---
    def delete_card_by_title(title, gallery_data):
        updated = [item for item in gallery_data if item["title"] != title]
        return updated
    
    def delete_card_by_index(index, gallery_data):
        print(f"Deleting card at index: {index} item: {gallery_data[index]}")
        updated = [item for i, item in enumerate(gallery_data) if i != index]
        return updated

    def create_delete_function(card_idx):
        def delete_specific_card(gallery_data):
            return delete_card_by_index(card_idx, gallery_data)
        return delete_specific_card

    def add_card(gallery_data):
        gallery_data = gallery_data.copy()
        if len(gallery_data) < MAX_CARDS:
            new_idx = len(gallery_data)
            gallery_data.append({"title": f"New Card {new_idx+1}", "path": "images/umbrella.png", "description": f"Added card {new_idx+1}"})
        return gallery_data

    def shift_card_ui(gallery_data):
        updates = []
        for idx in range(MAX_CARDS):
            if idx < len(gallery_data):
                updates.append(gr.update(value=f"### {gallery_data[idx]['title']}"))
                updates.append(gr.update(value=gallery_data[idx]["path"]))
            else:
                updates.append(gr.update(value=""))
                updates.append(gr.update(value=None))
        for idx in range(MAX_CARDS):
            updates.append(gr.update(visible=(idx < len(gallery_data))))
        return updates

    # --- Edit modal logic ---
    def open_edit_modal(idx, gallery_data):
        if idx < len(gallery_data):
            item = gallery_data[idx]
            return (
                gr.update(visible=True),  # Show modal
                idx,                     # Set current index
                f"### {item['title']}",  # Modal title (optional)
                item["description"]      # Populate description
            )
        else:
            return (gr.update(visible=True), idx, "### Edit", "")

    # Now hook up event bindings
    for idx, card in enumerate(card_components):
        if idx < len(initial_gallery_data):
            title = initial_gallery_data[idx]["title"]

            card["delete_btn"].click(
                fn=create_delete_function(idx),
                inputs=[gallery_data],
                outputs=[gallery_data]
            ).then(
                fn=shift_card_ui,
                inputs=[gallery_data],
                outputs=get_all_card_outputs()
            )
            card["refresh_btn"].click(lambda t=title: handle_action("Refresh", t), outputs=output_text)
            # Edit button opens the edit modal and populates fields
            card["edit_btn"].click(
                fn=open_edit_modal,
                inputs=[gr.State(idx), gallery_data],
                outputs=[edit_modal, edit_current_index, edit_image_title, edit_description]
            )
            card["to_3d_btn"].click(lambda t=title: handle_action("To 3D", t), outputs=output_text)

    add_btn = gr.Button("Add Card")
    add_btn.click(
        fn=add_card,
        inputs=[gallery_data],
        outputs=[gallery_data]
    ).then(
        fn=shift_card_ui,
        inputs=[gallery_data],
        outputs=get_all_card_outputs()
    )

    close_btn.click(
        fn=close_modal,
        inputs=[],
        outputs=[modal_image_title, modal_image, modal_visible, overlay]
    ).then(
        fn=lambda: gr.update(visible=False),
        outputs=settings_modal
    )

    cancel_edit_btn.click(
        fn=lambda: (gr.update(visible=False), None, "", ""),
        outputs=[edit_modal, edit_current_index, edit_image_title, edit_description]
    )


demo.launch(debug=True)
