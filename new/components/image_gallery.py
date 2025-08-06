"""Image gallery component for displaying generated objects."""

import gradio as gr
from components.image_card import create_image_card

MAX_CARDS = 20

def create_image_gallery():
    """Create the object gallery interface."""
    
    with gr.Column(elem_classes=["gallery-section"]) as gallery_section:
        gr.Markdown("### Object Gallery", elem_classes=["gallery-header"])
        
        # Initial gallery data
        initial_gallery_data = []
        gallery_data = gr.State(initial_gallery_data)
        
        # Create cards in a responsive grid using Gradio's native layout
        cards_per_row = 4  # Adjust this number to control cards per row
        card_components = []
        card_containers = []
        
        for row_start in range(0, MAX_CARDS, cards_per_row):
            with gr.Row() as row:
                for col_idx in range(cards_per_row):
                    card_idx = row_start + col_idx
                    if card_idx < MAX_CARDS:
                        # Use scale=1 to fill available space, but with min_width to maintain consistency
                        with gr.Column(scale=1, min_width=180, visible=False, elem_classes=["gallery-card"]) as card_container:
                            # Create placeholder modal components for card creation
                            # These will be replaced by actual modal components from main app
                            card = create_image_card("", "", None, None, None, None, None, None)
                            
                            card_components.append(card)
                            card_containers.append(card_container)
        
        # Utility: get all card outputs
        def get_all_card_outputs():
            outputs = []
            for card in card_components:
                outputs.extend([card["title_component"], card["image_component"]])
            outputs.extend(card_containers)
            return outputs
        
        # Logic functions
        def delete_card_by_index(index, gallery_data):
            print(f"Deleting card at index: {index} title: {gallery_data[index]['title']}")
            updated = [item for i, item in enumerate(gallery_data) if i != index]
            return updated

        def create_delete_function(card_idx):
            def delete_specific_card(gallery_data):
                return delete_card_by_index(card_idx, gallery_data)
            return delete_specific_card

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
        
        # Hook up event bindings for each card
        for idx, card in enumerate(card_components):
            card["delete_btn"].click(
                fn=create_delete_function(idx),
                inputs=[gallery_data],
                outputs=[gallery_data]
            ).then(
                fn=shift_card_ui,
                inputs=[gallery_data],
                outputs=get_all_card_outputs()
            )
    
    return {
        "section": gallery_section,
        "data": gallery_data,
        "card_components": card_components,
        "card_containers": card_containers,
        "get_all_card_outputs": get_all_card_outputs,
        "shift_card_ui": shift_card_ui
    } 