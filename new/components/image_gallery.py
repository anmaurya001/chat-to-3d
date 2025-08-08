"""Image gallery component for displaying generated objects."""

import gradio as gr
from components.image_card import create_image_card
import config

MAX_CARDS = config.MAX_CARDS
CARDS_PER_ROW = config.CARDS_PER_ROW

def create_image_gallery():
    """Create the object gallery interface."""
    
    with gr.Column(elem_classes=["gallery-section"]) as gallery_section:
        gr.Markdown("### Object Gallery", elem_classes=["gallery-header"])
        
        # Initial gallery data
        initial_gallery_data = []
        gallery_data = gr.State(initial_gallery_data)
        
        # Create placeholder for empty gallery state
        with gr.Column(visible=True, elem_classes=["gallery-placeholder"]) as placeholder_container:
            with gr.Column(elem_classes=["placeholder-content"]):
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
                          "First describe the scene you want above, and then we'll generate some ideas to populate the scene"
                          "</div>"
                          "</div>",
                    elem_classes=["placeholder-text"]
                )
        
        # Create cards in a responsive grid using Gradio's native layout
        card_components = []
        card_containers = []
        
        for row_start in range(0, MAX_CARDS, CARDS_PER_ROW):
            with gr.Row() as row:
                for col_idx in range(CARDS_PER_ROW):
                    card_idx = row_start + col_idx
                    if card_idx < MAX_CARDS:
                        # Use scale=1 to fill available space, but with min_width to maintain consistency
                        with gr.Column(scale=1, min_width=180, visible=False, elem_classes=["gallery-card"]) as card_container:
                            # Create placeholder modal components for card creation
                            # These will be replaced by actual modal components from main app
                            card = create_image_card("", "", None, None, None, None, None, None)
                            
                            card_components.append(card)
                            card_containers.append(card_container)
        
        # Add "Convert all to 3D" button at the bottom
        with gr.Row(elem_classes=["convert-all-section"]):
            convert_all_btn = gr.Button(
                "Convert all to 3D", 
                #size="lg", 
                #variant="primary",
                elem_classes=["convert-all-btn"],
                visible=False,  # Initially hidden, will be shown when gallery has items
                interactive=False  # Initially disabled, will be enabled when there are unconverted items
            )
        
        # Utility: get all card outputs
        def get_all_card_outputs():
            outputs = []
            for card in card_components:
                outputs.extend([card["title_component"], card["image_component"], card["refresh_btn"], card["edit_btn"], card["delete_btn"], card["to_3d_btn"]])
            outputs.extend(card_containers)
            outputs.append(placeholder_container)  # Add placeholder to outputs
            outputs.append(convert_all_btn)  # Add convert all button to outputs
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
            """Update the UI to reflect the current gallery data state."""
            print(f"🔄 Updating gallery UI with {len(gallery_data)} objects")
            updates = []
            for idx in range(MAX_CARDS):
                if idx < len(gallery_data):
                    obj = gallery_data[idx]
                    print(f"  Card {idx}: {obj['title']} -> {obj['path']}")
                    
                    updates.append(gr.update(value=f"### {obj['title']}"))
                    updates.append(gr.update(value=obj["path"]))
                    
                    # Check if 3D generation is in progress or batch processing to disable other buttons
                    is_3d_generating = obj.get("3d_generating", False)
                    is_batch_processing = obj.get("batch_processing", False)
                    is_processing = is_3d_generating or is_batch_processing
                    
                    # Update refresh button state
                    refresh_interactive = not is_processing
                    refresh_classes = ["action-btn"]
                    if is_processing:
                        refresh_classes.append("disabled-btn")
                    updates.append(gr.update(interactive=refresh_interactive, elem_classes=refresh_classes))
                    
                    # Update edit button state
                    edit_interactive = not is_processing
                    edit_classes = ["action-btn"]
                    if is_processing:
                        edit_classes.append("disabled-btn")
                    updates.append(gr.update(interactive=edit_interactive, elem_classes=edit_classes))
                    
                    # Update delete button state
                    delete_interactive = not is_processing
                    delete_classes = ["action-btn"]
                    if is_processing:
                        delete_classes.append("disabled-btn")
                    updates.append(gr.update(interactive=delete_interactive, elem_classes=delete_classes))
                    
                    # Update 3D button text and state based on 3D model status
                    if "glb_path" in obj and obj["glb_path"]:
                        # 3D model exists - show completed state
                        button_text = "✓ 3D"
                        button_interactive = False
                        button_classes = ["action-btn", "three-d-completed"]
                    elif "3d_generating" in obj and obj["3d_generating"]:
                        # 3D generation in progress
                        button_text = "⏳ 3D"
                        button_interactive = False
                        button_classes = ["action-btn", "three-d-generating"]
                    elif is_batch_processing:
                        # Batch processing - disable button
                        button_text = "⏳ 3D"
                        button_interactive = False
                        button_classes = ["action-btn", "three-d-generating"]
                    else:
                        # No 3D model - ready to generate
                        button_text = "→ 3D"
                        button_interactive = True
                        button_classes = ["action-btn"]
                    
                    updates.append(gr.update(value=button_text, interactive=button_interactive, elem_classes=button_classes))
                else:
                    updates.append(gr.update(value=""))
                    updates.append(gr.update(value=None))
                    updates.append(gr.update(interactive=True, elem_classes=["action-btn"]))  # refresh
                    updates.append(gr.update(interactive=True, elem_classes=["action-btn"]))  # edit
                    updates.append(gr.update(interactive=True, elem_classes=["action-btn"]))  # delete
                    updates.append(gr.update(value="→ 3D", interactive=True, elem_classes=["action-btn"]))  # 3D
            
            # Show/hide card containers based on data
            for idx in range(MAX_CARDS):
                updates.append(gr.update(visible=(idx < len(gallery_data))))
            
            # Show/hide placeholder based on whether gallery has items
            updates.append(gr.update(visible=(len(gallery_data) == 0)))
            
            # Show/enable convert all button based on whether there are items and unconverted items
            # and whether batch processing is in progress
            has_items = len(gallery_data) > 0
            
            has_unconverted_items = any(
                idx < len(gallery_data) and 
                not gallery_data[idx].get("glb_path") and 
                not gallery_data[idx].get("3d_generating", False)
                for idx in range(len(gallery_data))
            )
            
            # Check if any item is in batch processing mode
            is_batch_processing = any(
                idx < len(gallery_data) and 
                gallery_data[idx].get("batch_processing", False)
                for idx in range(len(gallery_data))
            )
            
            # Show button if there are items, enable if there are unconverted items and no batch processing
            show_convert_all = has_items
            enable_convert_all = has_unconverted_items and not is_batch_processing
            updates.append(gr.update(visible=show_convert_all, interactive=enable_convert_all))
            
            print(f"✅ Gallery UI updated with {len(gallery_data)} visible cards")
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
        "placeholder": placeholder_container,
        "convert_all_btn": convert_all_btn,
        "get_all_card_outputs": get_all_card_outputs,
        "shift_card_ui": shift_card_ui,
    } 