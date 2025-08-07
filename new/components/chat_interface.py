"""Chat interface component for LLM agent interaction."""

import gradio as gr


def create_chat_interface():
    """Create the chat interface for scene planning."""
    
    with gr.Column(elem_classes=["scene-section"]) as chat_section:
        gr.Markdown("### What scene do you want to create?")
        
        with gr.Row():
            with gr.Column(scale=8):
                scene_input = gr.Textbox(
                    placeholder="Describe your scene and any details you want to include...",
                    label="",
                    lines=1,
                    container=False,
                    elem_classes=["scene-input"]
                )
            with gr.Column(scale=1, min_width=50):
                send_btn = gr.Button("▶", elem_classes=["send-button"], size="sm")
    
    return {
        "section": chat_section,
        "input": scene_input,
        "send_btn": send_btn
    }


def handle_scene_description(scene_description, agent_service, gallery_data, image_generation_service=None):
    """Handle scene description and generate objects with 2D prompts, then populate gallery with generated images."""
    if not scene_description.strip():
        return "Please enter a scene description.", gallery_data
    
    try:
        # Use the existing generate_objects_and_prompts function which does both
        success, prompts, message = agent_service.generate_objects_and_prompts(scene_description)
        
        if success and prompts:
            # Print to console
            print(f"\n=== Scene Description: '{scene_description}' ===")
            print(f"Generated {len(prompts)} objects with 2D prompts:")
            print("=" * 60)
            
            # Update gallery data with new objects
            new_gallery_data = []
            for i, (obj_name, prompt) in enumerate(prompts.items(), 1):
                print(f"{i:2d}. {obj_name}")
                print(f"    2D Prompt: {prompt}")
                print()
                
                # Start with placeholder image
                gallery_item = {
                    "title": obj_name,
                    "path": None,
                    "description": prompt
                }
                new_gallery_data.append(gallery_item)
            
            print(f"Total objects: {len(prompts)}")
            print("=" * 60)
            
            # Automatically generate images if image generation service is available
            if image_generation_service:
                print("🎨 Generating images for all objects...")
                try:
                    success, message, generated_images = image_generation_service.generate_images_for_objects(new_gallery_data)
                    
                    if success and generated_images:
                        # Update gallery data with generated image paths
                        for obj in new_gallery_data:
                            object_name = obj["title"]
                            if object_name in generated_images:
                                obj["path"] = generated_images[object_name]
                                print(f"✅ Generated image for {object_name}: {generated_images[object_name]}")
                            else:
                                print(f"⚠️ No image generated for {object_name}")
                    else:
                        print(f"❌ Image generation failed: {message}")
                        
                except Exception as e:
                    print(f"❌ Error during image generation: {str(e)}")
            
            # Create LLM-style response with count, subject reiteration, and suggested actions
            # response = f"{len(prompts)} objects generated for \"{scene_description}\". "
            # response += "Objects have been added to the gallery with 2D prompts. "
            # if image_generation_service:
            #     response += "Images have been automatically generated for all objects. "
            response = f"Review the {len(prompts)} objects and delete any that are not needed. "
            response += "Next step: Generate 3D assets for selected objects."
            
            return response, new_gallery_data
        else:
            print(f"Error: {message}")
            return f"Error: {message}", gallery_data
        
    except Exception as e:
        print(f"Error generating objects and prompts: {str(e)}")
        return f"Error generating objects and prompts: {str(e)}", gallery_data 