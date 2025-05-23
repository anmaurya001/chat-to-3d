import gradio as gr
from agent import ScenePlanningAgent
from generator import AssetGenerator
from utils import delete_prompts_file
from glb_preview import create_glb_preview, update_preview, clear_preview
from config import (
    DEFAULT_SEED,
    DEFAULT_SPARSE_STEPS,
    DEFAULT_SLAT_STEPS,
    OUTPUT_DIR,
    PROMPTS_FILE,
    INITIAL_MESSAGE,
    DEFAULT_TRELLIS_MODEL,
)
import json
import os
import logging
import shutil

logger = logging.getLogger(__name__)


class SceneGeneratorInterface:
    def __init__(self):
        self.agent = ScenePlanningAgent()
        self.generator = AssetGenerator(default_model=DEFAULT_TRELLIS_MODEL)
        self.INITIAL_MESSAGE = INITIAL_MESSAGE
        # Delete existing prompts file if it exists
        delete_prompts_file()

    def parse_object_list(self, response):
        """Parse the LLM response to extract suggested objects from numbered lists.
        
        Args:
            response (str): The LLM's response text
            
        Returns:
            list: List of extracted objects with proper formatting
        """
        objects = []
        in_objects_section = False
        
        for line in response.split('\n'):
            line = line.strip()
            
            # Look for the start of the objects section
            if "Suggested objects:" in line:
                in_objects_section = True
                continue
                
            # Skip empty lines
            if not line:
                continue
                
            # If we're in the objects section and the line starts with a number
            if in_objects_section and line[0].isdigit():
                # Extract the object name after the number and dot
                obj = line.split('.', 1)[-1].strip()
                if obj:
                    # Format the object name
                    # Replace underscores with spaces and capitalize first letter of each word
                    obj = obj.replace('_', ' ').title()
                    objects.append(obj)
            # If we hit a line that doesn't start with a number, we're done with the objects section
            elif in_objects_section and not line[0].isdigit():
                break
                
        return objects

    def create_interface(self):
        """Create the Gradio interface for the 3D Scene Generator."""
        with gr.Blocks() as demo:
            gr.Markdown("# 3D Scene Generator")

            with gr.Tabs():
                with gr.TabItem("Chat & Generate Prompts"):
                    with gr.Row():
                        # Left Column - Chat Interface (scale=3 for wider)
                        with gr.Column(scale=3):
                            # Scene settings at the top
                            gr.Markdown("### Scene Settings")
                            chatbot = gr.Chatbot(
                                height=400, value=[(None, self.INITIAL_MESSAGE)]
                            )
                            msg = gr.Textbox(
                                label="Your message",
                                placeholder="Describe your scene...",
                            )
                            with gr.Row():
                                submit_btn = gr.Button("Send")
                                clear_btn = gr.Button("Clear Chat")

                            gr.Markdown("""
                            ### Ready to Generate 3D Prompts?
                            When you're satisfied with your scene, click the 'Generate 3D Prompts' button below to create detailed 3D prompts for each object.
                            """)
                            
                            generate_prompts_btn = gr.Button("Generate 3D Prompts")

                            prompts_output = gr.HTML(
                                value="<div style='background-color: #f8f9fa; padding: 15px; border-radius: 8px; border: 1px solid #dee2e6;'>"
                                      "<p style='margin: 0; color: #6c757d;'>No prompts generated yet</p>"
                                      "</div>"
                            )

                            generation_prompt_display = gr.Textbox(
                                label="Generation Prompt Used",
                                lines=5,
                                interactive=False,
                            )

                        # Right Column - Object Management Panel (scale=1 for narrower)
                        with gr.Column(scale=1):
                            gr.Markdown("### Object Management")
                            
                            # Suggested Objects Section
                            with gr.Group():
                                gr.Markdown("#### Suggested Objects")
                                suggested_objects_checkboxes = gr.CheckboxGroup(
                                    choices=[],
                                    label="Check objects to add to accepted list",
                                    interactive=True
                                )
                            
                            # Accepted Objects Section
                            with gr.Group():
                                gr.Markdown("#### Accepted Objects")
                                accepted_objects_display = gr.HTML(
                                    value="<div style='background-color: #f8f9fa; padding: 15px; border-radius: 8px; border: 1px solid #dee2e6;'>"
                                          "<p style='margin: 0; color: #6c757d;'>No objects accepted yet</p>"
                                          "</div>"
                                )
                            
                            # State components to track objects
                            suggested_objects_state = gr.State([])  # List of suggested objects
                            accepted_objects_state = gr.State([])   # List of accepted objects

                    def clear_chat():
                        """Clear the chat and reset object lists."""
                        # Clear agent's memory
                        self.agent.clear_memory()
                        # Reset all states
                        return [
                            [(None, self.INITIAL_MESSAGE)],  # chatbot
                            "",                              # msg
                            gr.update(choices=[], value=[]), # suggested_objects_checkboxes (clear choices and uncheck all)
                            [],                              # suggested_objects_state
                            [],                              # accepted_objects_state
                            gr.update(                      # accepted_objects_display
                                value="<div style='background-color: #f8f9fa; padding: 15px; border-radius: 8px; border: 1px solid #dee2e6;'>"
                                      "<p style='margin: 0; color: #6c757d;'>No objects accepted yet</p>"
                                      "</div>"
                            ),
                            gr.update(                      # prompts_output
                                value="<div style='background-color: #f8f9fa; padding: 15px; border-radius: 8px; border: 1px solid #dee2e6;'>"
                                      "<p style='margin: 0; color: #6c757d;'>No prompts generated yet</p>"
                                      "</div>"
                            ),
                            ""                              # generation_prompt_display
                        ]

                    def respond(message, chat_history, suggested, accepted):
                        """Handle chat response and update suggested objects."""
                        # Pass current accepted objects to the agent
                        response = self.agent.chat(message, current_objects=accepted)
                        # First element is user message, second is assistant response
                        chat_history.append((message, response))
                        
                        # Parse and update object list
                        objects = self.parse_object_list(response)
                        
                        # Keep existing accepted objects in the suggested list
                        # but don't show them as checked
                        all_objects = [obj for obj in accepted if obj not in objects] + objects
                        
                        # Format accepted objects display
                        accepted_html = "<div style='background-color: #f8f9fa; padding: 15px; border-radius: 8px; border: 1px solid #dee2e6;'>"
                        if accepted:
                            accepted_html += "<ul style='margin: 0; padding-left: 20px;'>"
                            for obj in accepted:
                                accepted_html += f"<li style='margin: 5px 0; color: #212529;'>{obj}</li>"
                            accepted_html += "</ul>"
                        else:
                            accepted_html += "<p style='margin: 0; color: #6c757d;'>No objects accepted yet</p>"
                        accepted_html += "</div>"
                        
                        return chat_history, "", gr.update(choices=all_objects), accepted, gr.update(value=accepted_html)

                    def update_accepted_objects(selected_objects):
                        """Update accepted objects based on checkbox selection."""
                        # Format accepted objects display
                        accepted_html = "<div style='background-color: #f8f9fa; padding: 15px; border-radius: 8px; border: 1px solid #dee2e6;'>"
                        if selected_objects:
                            accepted_html += "<ul style='margin: 0; padding-left: 20px;'>"
                            for obj in selected_objects:
                                accepted_html += f"<li style='margin: 5px 0; color: #212529;'>{obj}</li>"
                            accepted_html += "</ul>"
                        else:
                            accepted_html += "<p style='margin: 0; color: #6c757d;'>No objects accepted yet</p>"
                        accepted_html += "</div>"
                        
                        return selected_objects, gr.update(value=accepted_html)

                    def on_generate_prompts(chat_history, accepted_objects):
                        """Generate 3D prompts for the accepted objects."""
                        if not accepted_objects:
                            return gr.update(
                                value="<div style='background-color: #f8f9fa; padding: 15px; border-radius: 8px; border: 1px solid #dee2e6;'>"
                                      "<p style='margin: 0; color: #dc3545;'>Please select at least one object from the suggested list before generating prompts.</p>"
                                      "</div>"
                            ), "", gr.update(choices=[], value=None, interactive=False), "Please select objects first"
                        
                        try:
                            # Get the last user message as scene description
                            scene_description = chat_history[-1][0] if chat_history else ""
                            
                            # Generate prompts for accepted objects
                            success, prompts, generation_prompt = self.agent.generate_3d_prompts(
                                scene_name="current_scene",  # We can generate a better name if needed
                                initial_description=accepted_objects
                            )
                            
                            if not success:
                                return gr.update(
                                    value="<div style='background-color: #f8f9fa; padding: 15px; border-radius: 8px; border: 1px solid #dee2e6;'>"
                                          "<p style='margin: 0; color: #dc3545;'>Error generating prompts. Please try again.</p>"
                                          "</div>"
                                ), "", gr.update(choices=[], value=None, interactive=False), "Error generating prompts"
                            
                            # Format prompts for display
                            prompts_html = "<div style='background-color: #f8f9fa; padding: 15px; border-radius: 8px; border: 1px solid #dee2e6;'>"
                            for obj, prompt in prompts.items():
                                prompts_html += f"""
                                    <div style='margin-bottom: 15px;'>
                                        <h4 style='margin: 0 0 5px 0; color: #212529;'>{obj}</h4>
                                        <p style='margin: 0; color: #495057;'>{prompt}</p>
                                    </div>
                                """
                            prompts_html += "</div>"
                            
                            # Update object list immediately after generating prompts
                            objects = list(prompts.keys())
                            return (
                                gr.update(value=prompts_html),
                                generation_prompt,
                                gr.update(choices=objects, value=objects[0] if objects else None, interactive=True),
                                "Ready to generate variants"
                            )
                        except Exception as e:
                            return gr.update(
                                value=f"<div style='background-color: #f8f9fa; padding: 15px; border-radius: 8px; border: 1px solid #dee2e6;'>"
                                      f"<p style='margin: 0; color: #dc3545;'>Error generating prompts: {str(e)}</p>"
                                      f"</div>"
                            ), "", gr.update(choices=[], value=None, interactive=False), f"Error: {str(e)}"

                with gr.TabItem("Generate & Select Variants"):
                    #with gr.Column(elem_classes="variant-tab"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("### Object & Prompt Management")
                            with gr.Column():
                                object_dropdown = gr.Dropdown(
                                    label="Select Object",
                                    choices=[],
                                    interactive=True,
                                )
                                
                                prompt_display = gr.Textbox(
                                    label="Generation Prompt",
                                    lines=4,
                                    interactive=True,
                                    value="",
                                )
                             
                                num_variants = gr.Slider(
                                        label="Number of Variants",
                                        minimum=1,
                                        maximum=5,
                                        value=3,
                                        step=1,
                                        interactive=True,                                       
                                )

                                with gr.Accordion("Advanced Settings", open=False):
                                    seed = gr.Number(
                                        label="Random Seed",
                                        value=DEFAULT_SEED,
                                        interactive=True,
                                    )
                                
                                generate_btn = gr.Button(
                                    "Generate Variants", 
                                )
                                
                                generate_all_btn = gr.Button(
                                    "Generate Variants for All Objects",   
                                )
                                
                                generation_status = gr.Textbox(
                                    label="Status",
                                    lines=2,
                                    interactive=False
                                )
                                              
                        with gr.Column(scale=3):
                            gr.Markdown("### Image Variant Gallery")   
                            variant_gallery = gr.Gallery(
                                label="Generated Variants",
                                show_label=False,
                                elem_id="variant_gallery",
                                columns=3,
                                rows=1,
                                height="auto",
                                allow_preview=True,
                                show_download_button=False,
                                object_fit="contain",
                                value=[],
                            )

                            with gr.Row():
                                select_variant_btn = gr.Button(
                                    "Select This Variant",
                                    interactive=False,
                            )
                                

                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("### Selected Variants Overview")
                            selected_variants_gallery = gr.Gallery(
                                label="Selected Variants",
                                show_label=False,
                                columns=5,
                                rows=1,
                                height="auto",
                                allow_preview=True,
                                show_download_button=False,
                                object_fit="contain",
                                value=[],
                            )
                            with gr.Row():
                                generate_3d_btn = gr.Button(
                                    "Generate 3D Model for Selected Variant",
                                    interactive=False,
                                )
                                clear_selection_btn = gr.Button(
                                    "Clear Selection",
                                    interactive=False,
                                    variant="secondary"
                                )
                            generate_all_3d_btn = gr.Button(
                            "Generate 3D Models for All Selected Variants",
                            interactive=False,
                            )

                        with gr.Column(scale=3):
                            gr.Markdown("### 3d Model Preview")

                            with gr.Row():
                                # 3D Preview
                                with gr.Column(elem_classes="preview-container", scale=3):
                                    gr.Markdown("#### Main 3D Model Preview") # Renamed for clarity
                                    model_output = create_glb_preview()
                                    model_status = gr.Textbox(
                                        label="Generation Status",
                                        lines=2,
                                        interactive=False,
                                        elem_classes="control-element"
                                    )
                                
                                # Step 1.2: Add UI for all generated models
                                with gr.Column(elem_classes="preview-container", scale=1): # Keep same styling for now
                                    gr.Markdown("#### All Generated 3D Models (Session)")
                                    all_generated_models_display = gr.Dataset(
                                        components=["text"], # Just display name for now
                                        headers=["Generated Model"],
                                        label="Session Models",
                                        samples=[], 
                                        samples_per_page=5,
                                    )

                                    gr.Markdown("#### Save Generated Models")
                                    with gr.Row():
                                        save_folder = gr.Textbox(
                                            label="Save scene as",
                                            placeholder="Enter scene name...",
                                            interactive=True
                                        )
                                        save_btn = gr.Button("Save Assets")

                    # State components
                    current_object_state = gr.State(None)
                    current_variants_state = gr.State([])
                    selected_variants_state = gr.State({})
                    current_variant_image = gr.State(None)
                    selected_variant_index = gr.State(None)
                    all_variants_state = gr.State({})
                    all_session_models_state = gr.State([]) # Step 1.1: New state for all generated models
                    selected_overview_variant = gr.State(None) # New state for tracking selected overview variant

                    def update_object_list():
                        """Update the object dropdown with objects from the prompts file."""
                        try:
                            if not os.path.exists(PROMPTS_FILE):
                                logger.error(f"Prompts file not found at: {PROMPTS_FILE}")
                                return gr.update(
                                    choices=[],
                                    value=None,
                                    interactive=False
                                ), "Please generate prompts first in the Chat & Generate Prompts tab"
                            
                            logger.info(f"Reading prompts file from: {PROMPTS_FILE}")
                            with open(PROMPTS_FILE, 'r') as f:
                                data = json.load(f)
                                logger.info(f"Prompts file contents: {json.dumps(data, indent=2)}")
                                objects = [obj['name'] for scene in data['scenes'] for obj in scene['objects']]
                                logger.info(f"Found objects: {objects}")
                            
                            if not objects:
                                logger.warning("No objects found in prompts file")
                                return gr.update(
                                    choices=[],
                                    value=None,
                                    interactive=False
                                ), "No objects found in prompts file. Please generate prompts first."
                            
                            return gr.update(
                                choices=objects,
                                value=objects[0] if objects else None,  # Select first object by default
                                interactive=True
                            ), "Ready to generate variants"
                        except Exception as e:
                            logger.error(f"Error reading prompts file: {str(e)}")
                            return gr.update(
                                choices=[],
                                value=None,
                                interactive=False
                            ), f"Error reading prompts file: {str(e)}"

                    def create_object_gallery(object_name, variants):
                        """Create a gallery component for a specific object's variants."""
                        with gr.Group():
                            gr.Markdown(f"### {object_name}")
                            gallery = gr.Gallery(
                                label=f"{object_name} Variants",
                                show_label=False,
                                columns=3,
                                rows=1,
                                height="auto",
                                allow_preview=True,
                                show_download_button=False,
                                object_fit="contain",
                                value=[(v['image_path'], f"Variant {i+1} (Seed: {v['seed']})") for i, v in enumerate(variants)]
                            )
                            return gallery
                        
                    def build_image_variant_name(object_name, variants):
                        """Build a formatted image variant name."""
                        gallery_items = [
                                (v['image_path'], f"{object_name} (Seed: {v['seed']})")
                                for i, v in enumerate(variants)
                            ]
                        return gallery_items
                        
                    def generate_variants_for_object(object_name, seed, num_variants, prompt_display, all_variants_state):
                        """Generate variants for a single object."""
                        try:
                            if not object_name:
                                return "Please select an object first", None, [], all_variants_state
                            
                            if not os.path.exists(PROMPTS_FILE):
                                return "Please generate prompts first in the Chat & Generate Prompts tab", None, [], all_variants_state
                            
                            with open(PROMPTS_FILE, 'r') as f:
                                data = json.load(f)
                            
                            object_prompt = None
                            for scene in data['scenes']:
                                for obj in scene['objects']:
                                    if obj['name'] == object_name:
                                        object_prompt = prompt_display if prompt_display else obj['prompt']
                                        break
                                if object_prompt:
                                    break
                            
                            if not object_prompt:
                                return f"Object '{object_name}' not found in prompts file", None, [], all_variants_state
                            
                            object_dir = os.path.join(OUTPUT_DIR, object_name)
                            os.makedirs(object_dir, exist_ok=True)
                            
                            success, message, variants = self.generator.generate_variants(
                                object_name=object_name,
                                prompt=object_prompt,
                                output_dir=object_dir,
                                num_variants=num_variants,
                                seed=seed
                            )
                            
                            if not success:
                                return f"Error: {message}", None, [], all_variants_state
                            
                            if all_variants_state is None:
                                all_variants_state = {}
                            all_variants_state[object_name] = variants
                            
                            # gallery_items = [
                            #     (v['image_path'], f"{object_name} (Seed: {v['seed']})")
                            #     for i, v in enumerate(variants)
                            # ]
                            
                            gallery_items = build_image_variant_name(object_name, variants)
                            return (
                                f"Successfully generated {len(variants)} variants",
                                object_name,
                                gallery_items,
                                all_variants_state
                            )
                        except Exception as e:
                            return f"Error during generation: {str(e)}", None, [], all_variants_state

                    def generate_variants_for_all_objects(seed, num_variants, prompt_display):
                        """Generate variants for all objects in the prompts file."""
                        try:
                            if not os.path.exists(PROMPTS_FILE):
                                return "Please generate prompts first in the Chat & Generate Prompts tab", None, [], {}
                            
                            # Read prompts file
                            with open(PROMPTS_FILE, 'r') as f:
                                data = json.load(f)
                            
                            all_variants = {}
                            
                            # Get the current selected object
                            current_object = object_dropdown.value
                            
                            for scene in data['scenes']:
                                for obj in scene['objects']:
                                    object_name = obj['name']
                                    # Use edited prompt only for the currently selected object
                                    object_prompt = prompt_display if object_name == current_object else obj['prompt']
                                    
                                    # Create output directory for this object
                                    object_dir = os.path.join(OUTPUT_DIR, object_name)
                                    os.makedirs(object_dir, exist_ok=True)
                                    
                                    # Generate variants
                                    success, message, variants = self.generator.generate_variants(
                                        object_name=object_name,
                                        prompt=object_prompt,
                                        output_dir=object_dir,
                                        num_variants=num_variants,
                                        seed=seed 
                                    )
                                    
                                    if success:
                                        all_variants[object_name] = variants
                                    
                            
                            if not all_variants:
                                return "Failed to generate variants for any objects", None, [], {}
                            
                            # Get list of object names for the dropdown
                            object_names = list(all_variants.keys())
                            selected_object = object_names[0] if object_names else None
                            
                            # Create a list of variants for the selected object only
                            gallery_items = []
                            if selected_object and selected_object in all_variants:
                                logger.info(f"Selected object: {selected_object}")
                                gallery_items = [
                                    (v['image_path'], f"{selected_object} (Seed: {v['seed']})")
                                    for i, v in enumerate(all_variants[selected_object])
                                ]
                            
                            return (
                                f"Successfully generated variants for {len(all_variants)} objects",
                                selected_object,
                                gallery_items,
                                all_variants  # Return all variants for state
                            )
                        except Exception as e:
                            return f"Error during batch generation: {str(e)}", None, [], {}

                    def filter_variants_by_object(selected_object, all_variants):
                        """Filter variants to show only those for the selected object."""
                        logger.info(f"Filtering variants for object: {selected_object}")
                        if not selected_object or not all_variants:
                            return []
                        
                        # Get the variants for the selected object
                        if selected_object in all_variants:
                            # variants = [
                            #     (v['image_path'], f"{selected_object} - Variant {i+1} (Seed: {v['seed']})")
                            #     for i, v in enumerate(all_variants[selected_object])
                            # ]
                            variants = build_image_variant_name(selected_object, all_variants[selected_object])
                            logger.info(f"Found {len(variants)} variants for {selected_object}")
                            return variants
                        logger.warning(f"No variants found for {selected_object}")
                        return []

                    def update_prompt_display(selected_object):
                        """Update the prompt display when an object is selected."""
                        try:
                            if not selected_object or not os.path.exists(PROMPTS_FILE):
                                return ""
                            
                            with open(PROMPTS_FILE, 'r') as f:
                                data = json.load(f)
                            
                            for scene in data['scenes']:
                                for obj in scene['objects']:
                                    if obj['name'] == selected_object:
                                        return obj['prompt']
                            
                            return ""
                        except Exception as e:
                            logger.error(f"Error updating prompt display: {e}")
                            return ""

                    def clear_model_preview_on_object_change():
                        """Clears the 3D model preview and associated status when the object dropdown changes."""
                        logger.info("Object dropdown changed, clearing 3D model preview.")
                        cleared_model_output = clear_preview() # We expect the model path to be None
                        # The download button value should be None when there's no file
                        logger.info(f"Cleared model output: {cleared_model_output}, Download button value: None")
                        return cleared_model_output, "" # Cleared model, Empty status

                    def on_variant_select(object_name, all_variants, evt: gr.SelectData):
                        """Handle variant selection in gallery."""
                        logger.info(f"Variant selected: {evt.index}")
                        logger.info(f"Current object: {object_name}")
                        logger.info(f"All variants keys: {list(all_variants.keys()) if all_variants else 'None'}")
                        
                        if not object_name or not all_variants or object_name not in all_variants:
                            logger.warning(f"No variants available for object: {object_name}")
                            return gr.Button(interactive=False), None, None, gr.Button(interactive=False)
                        
                        variants = all_variants[object_name]
                        if evt.index is not None and evt.index < len(variants):
                            selected_variant = variants[evt.index]
                            logger.info(f"Selected variant: {selected_variant['image_path']}")
                            logger.info(f"Selected index: {evt.index}")
                            return (
                                gr.Button(interactive=True),  # select_variant_btn
                                selected_variant['image_path'],  # current_variant_image
                                evt.index,  # selected_variant_index
                                gr.Button(interactive=False)  # generate_3d_btn
                            )
                        return gr.Button(interactive=False), None, None, gr.Button(interactive=False)

                    def select_variant(object_name, variants, selected_variants, selected_idx, all_variants_state):
                        """Handle variant selection."""
                        logger.info(f"Selecting variant with index: {selected_idx}")
                        if not object_name or not all_variants_state or object_name not in all_variants_state:
                            return gr.update(value="No variants available to select"), []
                        
                        if selected_idx is None:
                            return gr.update(value="Please select a variant from the gallery first"), []
                        
                        if selected_idx >= len(all_variants_state[object_name]):
                            return gr.update(value="Invalid variant selection"), []
                        
                        # Get the selected variant from all_variants_state and add its original index
                        selected_variant_details = dict(all_variants_state[object_name][selected_idx]) # Make a copy
                        selected_variant_details['variant_idx'] = selected_idx # Store the original index
                        
                        # Update selected variants state
                        if selected_variants is None:
                            selected_variants = {}
                        selected_variants[object_name] = selected_variant_details # Store the augmented dict
                        
                        # Create gallery items for selected variants display (shows images)
                        gallery_items = [
                            (v['image_path'], f"{obj_name} (Seed: {v['seed']})")
                            for obj_name, v in selected_variants.items()
                        ]

                        # gallery_items = build_image_variant_name(object_name, selected_variants)
                        
                        return gr.update(value=f"Selected variant for {object_name}"), gallery_items

                    def on_overview_variant_select(evt: gr.SelectData, selected_variants):
                        """Handle selection of a variant from the overview gallery."""
                        if not selected_variants:
                            return None, gr.Button(interactive=False), gr.Button(interactive=False)
                        
                        # Get the selected variant's image path from the event
                        # In Gradio's Gallery component, the selected value is a dict with 'image' and 'caption' keys
                        selected_data = evt.value
                        logger.info(f"on_overview_variant_select: Selected overview variant: {selected_data}")
                        
                        if not selected_data or not isinstance(selected_data, dict) or 'image' not in selected_data:
                            logger.warning(f"Invalid selection data: {selected_data}")
                            return None, gr.Button(interactive=False), gr.Button(interactive=False)
                            
                        selected_image_path = selected_data['image']['path']
                        selected_filename = os.path.basename(selected_image_path)
                        logger.info(f"Selected overview variant image path: {selected_image_path}")
                        logger.info(f"Selected filename: {selected_filename}")
                        
                        # Find the object name and variant details for this image
                        for obj_name, variant in selected_variants.items():
                            variant_filename = os.path.basename(variant['image_path'])
                            logger.info(f"Comparing with variant filename: {variant_filename}")
                            if variant_filename == selected_filename:
                                logger.info(f"Found matching variant for object: {obj_name}")
                                return variant, gr.Button(interactive=True), gr.Button(interactive=True)
                        
                        logger.warning(f"No matching variant found for image: {selected_filename}")
                        return None, gr.Button(interactive=False), gr.Button(interactive=False)

                    def clear_selected_variant(selected_variants, selected_overview_variant):
                        """Clear the selected variant and reset button states."""
                        logger.info("Clearing selected variant")
                        
                        if not selected_overview_variant:
                            return None, gr.Button(interactive=False), gr.Button(interactive=False), "No variant selected to clear", selected_variants, []
                        
                        # Get the image path of the selected variant
                        selected_image_path = selected_overview_variant.get('image_path', '')
                        selected_filename = os.path.basename(selected_image_path)
                        
                        # Create a copy of selected variants to modify
                        updated_selected_variants = dict(selected_variants) if selected_variants else {}
                        
                        # Find and remove the selected variant
                        for obj_name, variant in list(updated_selected_variants.items()):
                            variant_filename = os.path.basename(variant['image_path'])
                            if variant_filename == selected_filename:
                                logger.info(f"Removing variant for object: {obj_name}")
                                del updated_selected_variants[obj_name]
                                break
                        
                        # Create gallery items for the updated selected variants
                        gallery_items = [
                            (v['image_path'], f"{obj_name} (Seed: {v['seed']})")
                             for obj_name, v in updated_selected_variants.items()
                         ]
                       
                        
                        return (
                            None,  # Clear selected overview variant
                            gr.Button(interactive=False),  # Disable generate button
                            gr.Button(interactive=False),  # Disable clear selection button
                            f"Cleared selection for {selected_filename}",  # Status message
                            updated_selected_variants,  # Updated selected variants state
                            gallery_items  # Updated gallery items
                        )

                    def generate_3d_model(object_name, all_variants_state, selected_idx, current_all_session_models, selected_overview_variant):
                        """Generate a 3D model for the selected object using the selected variant."""
                        logger.info(f"Starting 3D model generation for object: {object_name}")
                        logger.info(f"Selected overview variant: {selected_overview_variant}")
                        
                        # Initialize current_all_session_models if it's None
                        if current_all_session_models is None:
                            current_all_session_models = []

                        # Prepare placeholder for dataset display in case of early return
                        dataset_display_data = [[item['display_name']] for item in current_all_session_models]

                        # If we have a selected overview variant, use that instead
                        if selected_overview_variant:
                            logger.info(f"Using selected overview variant: {selected_overview_variant.get('image_path', 'No path')}")
                            variant = selected_overview_variant
                            # Get the object name from the variant's image path
                            image_path = variant.get('image_path', '')
                            object_name = os.path.basename(os.path.dirname(image_path))
                            logger.info(f"Extracted object name from path: {object_name}")
                            
                            if not object_name:
                                logger.error("Could not determine object name from variant path")
                                return "Error: Could not determine object name from variant", None, current_all_session_models, dataset_display_data, gr.Button(interactive=True)
                        else:
                            logger.info(f"No overview variant selected, using main gallery selection (index: {selected_idx})")
                            if not object_name or not all_variants_state or object_name not in all_variants_state:
                                logger.warning("No object selected or no variants available")
                                return "Please select an object and generate variants first", None, current_all_session_models, dataset_display_data, gr.Button(interactive=True)
                            
                            if selected_idx is None:
                                logger.warning("No variant selected")
                                return "Please select a variant first", None, current_all_session_models, dataset_display_data, gr.Button(interactive=True)
                            
                            variants = all_variants_state[object_name]
                            if not variants or selected_idx >= len(variants):
                                logger.error(f"Invalid variant selection: index {selected_idx} out of range for {len(variants)} variants")
                                return "Invalid variant selection", None, current_all_session_models, dataset_display_data, gr.Button(interactive=True)
                            
                            variant = variants[selected_idx]
                            logger.info(f"Using main gallery variant: {variant.get('image_path', 'No path')}")

                        try:
                            output_dir = os.path.join(OUTPUT_DIR, object_name, "3d_assets")
                            os.makedirs(output_dir, exist_ok=True)
                            
                            logger.info(f"Calling TRELLIS model for 3D generation for {object_name} using variant: {variant.get('image_path', 'No path')}")
                            success, message, outputs = self.generator.generate_3d_from_image(
                                object_name=object_name,
                                image_path=variant['image_path'],
                                output_dir=output_dir,
                                image_seed=variant.get('seed', '')
                            )
                            
                            if not success:
                                logger.error(f"3D generation failed for {object_name}: {message}")
                                return message, None, current_all_session_models, dataset_display_data, gr.Button(interactive=True)
                            
                            logger.info(f"3D model for {object_name} generated successfully. GLB path: {outputs['glb_path']}")
                            glb_path_for_preview = update_preview(outputs['glb_path'])
                            
                            # Create display name and new entry for the session models state
                            variant_seed = variant.get('seed', '') # Get seed if available
                            display_name = f"{object_name} - (Seed: {variant_seed}) - (Ts: {outputs['timestamp']})"
                            new_model_entry = {'display_name': display_name, 'glb_path': outputs['glb_path']}

                            # Update session models state: Upsert based on glb_path
                            updated_all_session_models = list(current_all_session_models) # Make a copy
                            found_existing = False
                            for i, existing_model in enumerate(updated_all_session_models):
                                if existing_model['glb_path'] == new_model_entry['glb_path']:
                                    updated_all_session_models[i] = new_model_entry # Update existing entry
                                    logger.info(f"Updated model in session: {display_name}")
                                    found_existing = True
                                    break
                            if not found_existing:
                                updated_all_session_models.append(new_model_entry)
                                logger.info(f"Added new model to session: {display_name}")

                            # Prepare data for the gr.Dataset component
                            final_dataset_display_data = [[item['display_name']] for item in updated_all_session_models]
                            
                            return (
                                f"Successfully generated 3D: {display_name}", 
                                glb_path_for_preview, 
                                updated_all_session_models, # The updated state itself
                                gr.update(samples=final_dataset_display_data), # Updated for Dataset
                                gr.Button(interactive=True)  # Re-enable generate button
                            )
                            
                        except Exception as e:
                            logger.error(f"Error in generate_3d_model for {object_name}: {str(e)}", exc_info=True)
                            error_msg = f"Error generating 3D model: {str(e)}"
                            cleared_glb_path = clear_preview()
                            # Return current (unmodified by this error) session models and its display data
                            current_dataset_display_data = [[item['display_name']] for item in current_all_session_models]
                            return error_msg, cleared_glb_path, current_all_session_models, gr.update(samples=current_dataset_display_data), gr.Button(interactive=True)

                    def generate_3d_for_all_selected(selected_variants, current_all_session_models):
                        """Generate 3D models for all selected variants."""
                        if not selected_variants:
                            # Ensure all_session_models_display is updated with its current state
                            dataset_display_data = [[item['display_name']] for item in current_all_session_models] if current_all_session_models else []
                            return "No variants selected", None, current_all_session_models, dataset_display_data
                        
                        if current_all_session_models is None:
                            current_all_session_models = [] # Initialize if None

                        updated_all_session_models = list(current_all_session_models) # Make a copy to modify throughout the loop
                        results_log = []
                        last_successful_glb_path_for_preview = None

                        try:
                            for object_name, variant_details in selected_variants.items():
                                logger.info(f"Batch generating 3D for {object_name} from variant: {variant_details.get('image_path')}")
                                output_dir = os.path.join(OUTPUT_DIR, object_name, "3d_assets")
                                os.makedirs(output_dir, exist_ok=True)

                                success, message, outputs = self.generator.generate_3d_from_image(
                                    object_name=object_name,
                                    image_path=variant_details['image_path'],
                                    output_dir=output_dir,
                                    image_seed=variant_details.get('seed', '')
                                )
                                
                                if success:
                                    results_log.append(f"Successfully generated 3D for {object_name}")
                                    last_successful_glb_path_for_preview = update_preview(outputs['glb_path'])
                                    
                                    variant_seed = variant_details.get('seed', '') # Get seed if available
                                    display_name = f"{object_name} - (Seed: {variant_seed}) - (Ts: {outputs['timestamp']})"                               
                                    new_model_entry = {'display_name': display_name, 'glb_path': outputs['glb_path']}

                                    # Upsert logic for this new model entry
                                    found_existing = False
                                    for i, existing_model in enumerate(updated_all_session_models):
                                        if existing_model['glb_path'] == new_model_entry['glb_path']:
                                            updated_all_session_models[i] = new_model_entry
                                            logger.info(f"Updated model in session (batch): {display_name}")
                                            found_existing = True
                                            break
                                    if not found_existing:
                                        updated_all_session_models.append(new_model_entry)
                                        logger.info(f"Added new model to session (batch): {display_name}")
                                else:
                                    results_log.append(f"Failed for {object_name}: {message}")
                                    logger.error(f"Batch 3D generation failed for {object_name}: {message}")
                            
                            final_status_message = "\n".join(results_log) if results_log else "No models processed."
                            final_dataset_display_data = [[item['display_name']] for item in updated_all_session_models]

                            return (
                                final_status_message,
                                last_successful_glb_path_for_preview, 
                                updated_all_session_models,
                                gr.update(samples=final_dataset_display_data)
                            )
                            
                        except Exception as e:
                            logger.error(f"Error during batch 3D generation: {str(e)}", exc_info=True)
                            error_msg = f"Error during batch 3D generation: {str(e)}"
                            current_dataset_display_data = [[item['display_name']] for item in updated_all_session_models] # Use potentially partially updated list
                            return error_msg, None, updated_all_session_models, gr.update(samples=current_dataset_display_data)

                    def on_session_model_select(evt: gr.SelectData, current_all_session_models):
                        """Handles selection of a model from the all_generated_models_display Dataset."""
                        if evt.index is None or not current_all_session_models or evt.index >= len(current_all_session_models):
                            logger.warning("Invalid selection from session models display or no models available.")
                            # Optionally clear preview or return current values if no valid selection
                            cleared_model_output = clear_preview()
                            return cleared_model_output, "Invalid selection or no model data."

                        selected_model_data = current_all_session_models[evt.index]
                        glb_path = selected_model_data.get('glb_path')
                        display_name = selected_model_data.get('display_name', 'Selected Model')

                        if not glb_path or not os.path.exists(glb_path):
                            logger.error(f"GLB path not found or invalid for selected session model: {glb_path}")
                            cleared_model_output = clear_preview()
                            return cleared_model_output, f"Error: GLB file not found for {display_name}."
                        
                        logger.info(f"Loading model from session display: {display_name} (Path: {glb_path})")
                        model_output_val = update_preview(glb_path)
                        return model_output_val,  f"Displaying: {display_name}"

                    def save_session_models(save_path, current_all_session_models):
                        """Save all session models to the specified folder."""
                        if not current_all_session_models:
                            return "No models to save"
                        
                        if not save_path:
                            return "Please provide a scene name"
                        
                        try:
                            # Create the save directory using the scene name
                            save_dir = os.path.join(OUTPUT_DIR, save_path)
                            os.makedirs(save_dir, exist_ok=True)
                            
                            saved_models = []
                            for model in current_all_session_models:
                                glb_path = model.get('glb_path')
                                if glb_path and os.path.exists(glb_path):
                                    # Get the filename from the original path
                                    filename = os.path.basename(glb_path)
                                    # Create new path in save directory
                                    new_path = os.path.join(save_dir, filename)
                                    # Copy the file
                                    shutil.copy2(glb_path, new_path)
                                    saved_models.append(filename)
                            
                            if saved_models:
                                return f"Successfully saved {len(saved_models)} models to {save_dir}"
                            else:
                                return "No models were saved"
                                
                        except Exception as e:
                            logger.error(f"Error saving models: {str(e)}")
                            return f"Error saving models: {str(e)}"

                    # Add handler for save functionality
                    save_btn.click(
                        fn=save_session_models,
                        inputs=[save_folder, all_session_models_state],
                        outputs=[model_status]
                    )

                    
                    # Add handler for object dropdown to filter variants
                    object_dropdown.change(
                        fn=filter_variants_by_object,
                        inputs=[object_dropdown, all_variants_state],
                        outputs=[variant_gallery]
                    ).then(
                        fn=lambda x: x,  # Pass through the selected object
                        inputs=[object_dropdown],
                        outputs=[current_object_state]
                    ).then(
                        fn=update_prompt_display,
                        inputs=[object_dropdown],
                        outputs=[prompt_display]
                    )


                    # Set up event handlers for variant generation
                    generate_btn.click(
                        fn=generate_variants_for_object,
                        inputs=[
                            object_dropdown,
                            seed,
                            num_variants,
                            prompt_display,
                            all_variants_state
                        ],
                        outputs=[
                            generation_status,
                            current_object_state,
                            variant_gallery,
                            all_variants_state
                        ]
                    )

                    # Add handler for generate all variants button
                    generate_all_btn.click(
                        fn=generate_variants_for_all_objects,
                        inputs=[seed, num_variants, prompt_display],
                        outputs=[
                            generation_status,
                            object_dropdown,
                            variant_gallery,
                            all_variants_state
                        ]
                    )

                    # Enable select variant button when a variant is selected
                    variant_gallery.select(
                        fn=on_variant_select,
                        inputs=[current_object_state, all_variants_state],
                        outputs=[select_variant_btn, current_variant_image, selected_variant_index, gr.Button(interactive=False)]  # Always disable generate_3d_btn
                    )

                    # Handle variant selection
                    select_variant_btn.click(
                        fn=select_variant,
                        inputs=[
                            current_object_state,
                            current_variants_state,
                            selected_variants_state,
                            selected_variant_index,
                            all_variants_state
                        ],
                        outputs=[generation_status, selected_variants_gallery]
                    ).then(
                        fn=lambda: gr.Button(interactive=True),
                        outputs=[generate_all_3d_btn]
                    )

                    # Add handler for overview gallery selection
                    selected_variants_gallery.select(
                        fn=on_overview_variant_select,
                        inputs=[selected_variants_state],
                        outputs=[selected_overview_variant, generate_3d_btn, clear_selection_btn]  # Added clear_selection_btn
                    )

                    # Add handler for clear selection button
                    clear_selection_btn.click(
                        fn=clear_selected_variant,
                        inputs=[selected_variants_state, selected_overview_variant],
                        outputs=[
                            selected_overview_variant,
                            generate_3d_btn,
                            clear_selection_btn,
                            model_status,
                            selected_variants_state,
                            selected_variants_gallery
                        ]
                    )

                    # Update generate_3d_btn click handler to include selected_overview_variant
                    generate_3d_btn.click(
                        fn=lambda: gr.Button(interactive=False),  # Disable button during generation
                        outputs=[generate_3d_btn]
                    ).then(
                        generate_3d_model,
                        inputs=[
                            current_object_state, 
                            all_variants_state, 
                            selected_variant_index, 
                            all_session_models_state,
                            selected_overview_variant
                        ],
                        outputs=[model_status, model_output, all_session_models_state, all_generated_models_display, generate_3d_btn]
                    )

                    # Add confirmation dialog components
                    with gr.Row(visible=False) as confirm_dialog:
                        gr.Markdown("⚠️ This process may take several minutes depending on the number and complexity of objects. Do you want to continue?")
                        with gr.Row():
                            confirm_btn = gr.Button("Continue")
                            cancel_btn = gr.Button("Cancel")

                    # Add handler for generate all 3D models button
                    generate_all_3d_btn.click(
                        fn=lambda: gr.Row(visible=True),
                        outputs=[confirm_dialog]
                    )

                    # Handle confirmation
                    confirm_btn.click(
                        fn=lambda: (gr.Row(visible=False), gr.Button(interactive=False)),
                        outputs=[confirm_dialog, generate_all_3d_btn]
                    ).then(
                        generate_3d_for_all_selected,
                        inputs=[selected_variants_state, all_session_models_state],
                        outputs=[model_status, model_output, all_session_models_state, all_generated_models_display]
                    ).then(
                        fn=lambda: gr.Button(interactive=True),
                        outputs=[generate_all_3d_btn]
                    )

                    # Handle cancellation
                    cancel_btn.click(
                        fn=lambda: (gr.Row(visible=False), "Generation cancelled by user"),
                        outputs=[confirm_dialog, model_status]
                    )

                    
                    # Add handler for session model selection
                    all_generated_models_display.select(
                        fn=on_session_model_select,
                        inputs=[all_session_models_state], # evt: gr.SelectData is implicitly the first arg to fn
                        outputs=[model_output, model_status]
                    )

                    

                # Update object list when the tab is selected
                demo.load(fn=update_object_list, outputs=[object_dropdown, generation_status])

                # Set up event handlers for chat and prompt generation
                clear_btn.click(
                    fn=clear_chat,
                    inputs=[],
                    outputs=[
                        chatbot,
                        msg,
                        suggested_objects_checkboxes,
                        suggested_objects_state,
                        accepted_objects_state,
                        accepted_objects_display,
                        prompts_output,
                        generation_prompt_display
                    ]
                )
                
                submit_btn.click(
                    respond, 
                    [msg, chatbot, suggested_objects_state, accepted_objects_state], 
                    [chatbot, msg, suggested_objects_checkboxes, accepted_objects_state, accepted_objects_display]
                )
                
                msg.submit(
                    respond, 
                    [msg, chatbot, suggested_objects_state, accepted_objects_state], 
                    [chatbot, msg, suggested_objects_checkboxes, accepted_objects_state, accepted_objects_display]
                )

                # Handle checkbox changes
                suggested_objects_checkboxes.change(
                    update_accepted_objects,
                    [suggested_objects_checkboxes],
                    [accepted_objects_state, accepted_objects_display]
                )

                # Add event handler for generate prompts button
                generate_prompts_btn.click(
                    on_generate_prompts,
                    [chatbot, accepted_objects_state],
                    [prompts_output, generation_prompt_display, object_dropdown, generation_status]
                )

        return demo

    def launch(self, **kwargs):
        """Launch the Gradio interface."""
        demo = self.create_interface()
        demo.launch(**kwargs)



