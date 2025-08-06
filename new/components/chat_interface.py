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


def handle_scene_description(scene_description, agent_service):
    """Handle scene description and generate objects with 3D prompts."""
    if not scene_description.strip():
        return "Please enter a scene description."
    
    try:
        # Use the existing generate_3d_prompts function which does both
        success, prompts, message = agent_service.generate_objects_and_prompts(scene_description)
        
        if success and prompts:
            # Print to console
            print(f"\n=== Scene Description: '{scene_description}' ===")
            print(f"Generated {len(prompts)} objects with 2D prompts:")
            print("=" * 60)
            
            for i, (obj_name, prompt) in enumerate(prompts.items(), 1):
                print(f"{i:2d}. {obj_name}")
                print(f"    3D Prompt: {prompt}")
                print()
            
            print(f"Total objects: {len(prompts)}")
            print("=" * 60)
            
            return "Objects and 2D prompts generated! Check console for details."
        else:
            print(f"Error: {message}")
            return f"Error: {message}"
        
    except Exception as e:
        print(f"Error generating objects and prompts: {str(e)}")
        return f"Error generating objects and prompts: {str(e)}" 