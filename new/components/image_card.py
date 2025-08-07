import gradio as gr
import random
import os
from services.image_generation_service import ImageGenerationService
from services.model_3d_service import Model3DService
import datetime

def create_image_card(image_path, title, output_widget, modal_image_title, modal_image, modal_visible, settings_modal, overlay):
    """Create a single image card with action buttons and modal trigger."""
    with gr.Column(elem_classes=["card-content", "clickable-card"]):
        # Create a container for the image and title that covers most of the card
        with gr.Column(elem_classes=["card-main-area"]) as main_area:
            # Transparent button that covers the entire main area for clickability
            card_click_btn = gr.Button("", elem_classes=["card-click-btn"], visible=True)
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
            to_3d_btn = gr.Button("→ 3D", size="sm", min_width=10, elem_classes=["action-btn"], elem_id="to-3d-btn")
            gr.HTML("<div style='flex-grow: 1;'></div>")

    # Return the image component and buttons for event binding in app.py
    return {
        "title_component": title_component,
        "image_component": image_component,
        "main_area": main_area,
        "card_click_btn": card_click_btn,
        "refresh_btn": refresh_btn,
        "edit_btn": edit_btn,
        "delete_btn": delete_btn,
        "to_3d_btn": to_3d_btn
    }

def create_refresh_handler(image_generation_service):
    """Create a refresh handler that generates a new image with a random seed."""
    def refresh_image(card_idx, gallery_data):
        """Refresh the image for a specific card with a new random seed."""
        try:
            if card_idx >= len(gallery_data):
                print(f"Card index {card_idx} out of range")
                return gallery_data
            
            # Get the current object data
            obj = gallery_data[card_idx]
            object_name = obj["title"]
            prompt = obj["description"]
            
            # Validate that we have the required data
            if not object_name or not prompt:
                print(f"❌ Missing required data for card {card_idx}: title='{object_name}', prompt='{prompt}'")
                return gallery_data
            
            # Generate a new random seed
            new_seed = random.randint(1, 999999)
            
            # Set output directory for generated images
            output_dir = "static/images/generated"
            
            print(f"🔄 Refreshing image for '{object_name}' with seed {new_seed}")
            print(f"   Prompt: {prompt}")
            
            # Generate new image using SANA service
            success, message, new_image_path = image_generation_service.generate_image_from_prompt(
                object_name=object_name,
                prompt=prompt,
                output_dir=output_dir,
                seed=new_seed
            )
            
            if success and new_image_path:
                # Update the gallery data with the new image path
                updated_data = gallery_data.copy()
                updated_data[card_idx]["path"] = new_image_path
                updated_data[card_idx]["seed"] = new_seed
                
                print(f"✅ Successfully refreshed image: {new_image_path}")
                return updated_data
            else:
                print(f"❌ Failed to refresh image: {message}")
                return gallery_data
                
        except Exception as e:
            print(f"❌ Error refreshing image: {str(e)}")
            return gallery_data
    
    return refresh_image 

def create_3d_generation_handler(model_3d_service):
    """Create a 3D generation handler that converts images to 3D models."""
    def generate_3d_model(card_idx, gallery_data):
        """Generate a 3D model for a specific card."""
        try:
            if card_idx >= len(gallery_data):
                print(f"❌ Card index {card_idx} out of range")
                return gallery_data
            
            # Get the current object data
            obj = gallery_data[card_idx]
            object_name = obj["title"]
            image_path = obj["path"]
            
            # Validate that we have the required data
            if not object_name or not image_path:
                print(f"❌ Missing required data for card {card_idx}: title='{object_name}', path='{image_path}'")
                return gallery_data
            
            # Check if 3D model already exists
            if "glb_path" in obj and obj["glb_path"]:
                print(f"✅ 3D model already exists for '{object_name}': {obj['glb_path']}")
                return gallery_data
            
            # Check if generation is already in progress (should be true from immediate update)
            if "3d_generating" in obj and obj["3d_generating"]:
                print(f"⏳ 3D generation in progress for '{object_name}' - continuing...")
            else:
                print(f"⚠️ 3D generation not marked as in progress, but continuing...")
            
            # Set output directory for generated 3D models
            output_dir = "assets/models"
              
            # Generate 3D model using Model3DService
            success, message, glb_path = model_3d_service.generate_3d_model(
                image_path=image_path,
                output_dir=output_dir
            )
            
            # Update the gallery data with the result
            updated_data = gallery_data.copy()
            
            if success and glb_path:
                # Update the gallery data with the 3D model path
                updated_data[card_idx]["glb_path"] = glb_path
                updated_data[card_idx]["3d_generated"] = True
                updated_data[card_idx]["3d_timestamp"] = datetime.datetime.now().isoformat()
                updated_data[card_idx]["3d_generating"] = False  # Mark as complete                
                print(f"✅ Successfully generated 3D model: {glb_path}")
                return updated_data
            else:
                # Mark generation as failed
                updated_data[card_idx]["3d_generating"] = False
                print(f"❌ Failed to generate 3D model: {message}")
                print(f"🔍 DEBUG: Generation failed, reset 3d_generating=False for card {card_idx}")
                return updated_data
                
        except Exception as e:
            print(f"❌ Error generating 3D model: {str(e)}")
            print(f"🔍 DEBUG: Exception occurred, resetting 3d_generating=False for card {card_idx}")
            # Mark generation as failed
            updated_data = gallery_data.copy()
            updated_data[card_idx]["3d_generating"] = False
            return updated_data
    
    return generate_3d_model 