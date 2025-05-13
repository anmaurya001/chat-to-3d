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
            gr.Markdown("# 3D Scene Generator POC")

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
                    with gr.Row():
                        # Left Column - Generation Settings
                        with gr.Column(scale=1):
                            gr.Markdown("### Generation Settings")
                            
                            # Generation parameters
                            with gr.Group():
                                seed = gr.Number(
                                    label="Random Seed",
                                    value=DEFAULT_SEED,
                                    interactive=True
                                )
                                num_variants = gr.Number(
                                    label="Number of Variants",
                                    value=3,
                                    minimum=1,
                                    maximum=5,
                                    interactive=True
                                )
                            
                            # Add prompt display and editing
                            prompt_display = gr.Textbox(
                                label="Generation Prompt",
                                lines=3,
                                interactive=True,
                                value=""
                            )
                            
                            # Generation controls
                            generate_btn = gr.Button("Generate Variants")
                            generate_all_btn = gr.Button("Generate Variants for All Objects")
                            generation_status = gr.Textbox(
                                label="Generation Status",
                                lines=2,
                                interactive=False
                            )

                        # Right Column - Variant Selection
                        with gr.Column(scale=2):
                            gr.Markdown("### Variant Selection")
                            
                            # Object selection
                            object_dropdown = gr.Dropdown(
                                label="Select Object",
                                choices=[],
                                interactive=True
                            )
                            
                            # Variant gallery
                            with gr.Column():
                                gr.Markdown("### Object Variants")
                                variant_gallery = gr.Gallery(
                                    label="Generated Variants",
                                    show_label=True,
                                    elem_id="variant_gallery",
                                    columns=3,
                                    rows=1,
                                    height="auto",
                                    allow_preview=True,
                                    show_download_button=False,
                                    object_fit="contain",
                                    value=[]  # Initialize with empty list
                                )
                            
                            # Selection controls
                            select_variant_btn = gr.Button("Select This Variant")
                            
                            # Selected Variants Display
                            with gr.Column():
                                gr.Markdown("### Selected Variants")
                                selected_variants_display = gr.HTML(
                                    value="<div style='background-color: #f8f9fa; padding: 15px; border-radius: 8px; border: 1px solid #dee2e6;'>"
                                          "<p style='margin: 0; color: #6c757d;'>No variants selected yet</p>"
                                          "</div>"
                                )
                                selected_variants_gallery = gr.Gallery(
                                    label="Selected Variants",
                                    show_label=False,
                                    columns=3,
                                    rows=1,
                                    height="auto",
                                    allow_preview=True,
                                    show_download_button=False,
                                    object_fit="contain",
                                    value=[]
                                )
                            
                            # 3D Generation Controls
                            with gr.Row():
                                generate_3d_btn = gr.Button("Generate 3D Model", interactive=False)
                                generate_all_3d_btn = gr.Button("Generate 3D Models for All Selected Variants", interactive=False)
                            
                            # 3D Preview
                            gr.Markdown("### 3D Model Preview")
                            model_status = gr.Textbox(
                                label="Generation Status",
                                lines=2,
                                interactive=False
                            )

                    # State components
                    current_object_state = gr.State(None)
                    current_variants_state = gr.State([])
                    selected_variants_state = gr.State({})
                    current_variant_image = gr.State(None)
                    selected_variant_index = gr.State(None)  # Add state for selected index
                    all_variants_state = gr.State({})  # Add state to store all variants

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

                    def generate_variants_for_all_objects(seed, num_variants, prompt_display):
                        """Generate variants for all objects in the prompts file."""
                        try:
                            if not os.path.exists(PROMPTS_FILE):
                                return "Please generate prompts first in the Chat & Generate Prompts tab", None, [], {}
                            
                            # Read prompts file
                            with open(PROMPTS_FILE, 'r') as f:
                                data = json.load(f)
                            
                            all_variants = {}
                            total_objects = len([obj for scene in data['scenes'] for obj in scene['objects']])
                            processed_objects = 0
                            
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
                                        prompt=object_prompt,
                                        output_dir=object_dir,
                                        num_variants=num_variants,
                                        seed=seed + processed_objects  # Use different seed for each object
                                    )
                                    
                                    if success:
                                        all_variants[object_name] = variants
                                    
                                    processed_objects += 1
                            
                            if not all_variants:
                                return "Failed to generate variants for any objects", None, [], {}
                            
                            # Get list of object names for the dropdown
                            object_names = list(all_variants.keys())
                            selected_object = object_names[0] if object_names else None
                            
                            # Create a list of variants for the selected object only
                            selected_variants = []
                            if selected_object and selected_object in all_variants:
                                selected_variants = [
                                    (v['image_path'], f"{selected_object} - Variant {i+1} (Seed: {v['seed']})")
                                    for i, v in enumerate(all_variants[selected_object])
                                ]
                            
                            return (
                                f"Successfully generated variants for {len(all_variants)} objects",
                                selected_object,
                                selected_variants,
                                all_variants  # Return all variants for state
                            )
                        except Exception as e:
                            return f"Error during batch generation: {str(e)}", None, [], {}

                    def generate_variants_for_object(object_name, seed, num_variants, prompt_display, all_variants_state):
                        """Generate variants for a single object."""
                        try:
                            if not object_name:
                                return "Please select an object first", None, [], all_variants_state
                            
                            if not os.path.exists(PROMPTS_FILE):
                                return "Please generate prompts first in the Chat & Generate Prompts tab", None, [], all_variants_state
                            
                            # Read prompts file
                            with open(PROMPTS_FILE, 'r') as f:
                                data = json.load(f)
                            
                            # Find the object's prompt
                            object_prompt = None
                            for scene in data['scenes']:
                                for obj in scene['objects']:
                                    if obj['name'] == object_name:
                                        # Use edited prompt if available
                                        object_prompt = prompt_display if prompt_display else obj['prompt']
                                        break
                                if object_prompt:
                                    break
                            
                            if not object_prompt:
                                return f"Object '{object_name}' not found in prompts file", None, [], all_variants_state
                            
                            # Create output directory for this object
                            object_dir = os.path.join(OUTPUT_DIR, object_name)
                            os.makedirs(object_dir, exist_ok=True)
                            
                            # Generate variants
                            success, message, variants = self.generator.generate_variants(
                                prompt=object_prompt,
                                output_dir=object_dir,
                                num_variants=num_variants,
                                seed=seed
                            )
                            
                            if not success:
                                return f"Error: {message}", None, [], all_variants_state
                            
                            # Update the variants state with new variants for this object
                            if all_variants_state is None:
                                all_variants_state = {}
                            all_variants_state[object_name] = variants
                            
                            # Create gallery items
                            gallery_items = [
                                (v['image_path'], f"{object_name} - Variant {i+1} (Seed: {v['seed']})")
                                for i, v in enumerate(variants)
                            ]
                            
                            return (
                                f"Successfully generated {len(variants)} variants",
                                object_name,
                                gallery_items,
                                all_variants_state
                            )
                        except Exception as e:
                            return f"Error during generation: {str(e)}", None, [], all_variants_state

                    def on_variant_select(object_name, all_variants, evt: gr.SelectData):
                        """Handle variant selection in gallery."""
                        logger.info(f"Variant selected: {evt.index}")
                        logger.info(f"Current object: {object_name}")
                        logger.info(f"All variants keys: {list(all_variants.keys()) if all_variants else 'None'}")
                        
                        if not object_name or not all_variants or object_name not in all_variants:
                            logger.warning(f"No variants available for object: {object_name}")
                            return gr.Button(interactive=False), None, None
                        
                        variants = all_variants[object_name]
                        if evt.index is not None and evt.index < len(variants):
                            selected_variant = variants[evt.index]
                            logger.info(f"Selected variant: {selected_variant['image_path']}")
                            logger.info(f"Selected index: {evt.index}")
                            return (
                                gr.Button(interactive=True),  # generate_3d_btn
                                selected_variant['image_path'],  # current_variant_image
                                evt.index  # selected_variant_index
                            )
                        return gr.Button(interactive=False), None, None

                    def select_variant(object_name, variants, selected_variants, selected_idx, all_variants_state):
                        """Handle variant selection."""
                        logger.info(f"Selecting variant with index: {selected_idx}")
                        if not object_name or not all_variants_state or object_name not in all_variants_state:
                            return gr.update(value="No variants available to select"), []
                        
                        if selected_idx is None:
                            return gr.update(value="Please select a variant from the gallery first"), []
                        
                        if selected_idx >= len(all_variants_state[object_name]):
                            return gr.update(value="Invalid variant selection"), []
                        
                        # Get the selected variant from all_variants_state
                        selected_variant = all_variants_state[object_name][selected_idx]
                        
                        # Update selected variants
                        if selected_variants is None:
                            selected_variants = {}
                        selected_variants[object_name] = selected_variant
                        
                        # Format the selected variants display
                        html = "<div style='background-color: #f8f9fa; padding: 15px; border-radius: 8px; border: 1px solid #dee2e6;'>"
                        if selected_variants:
                            html += "<ul style='margin: 0; padding-left: 20px;'>"
                            for obj_name, variant in selected_variants.items():
                                html += f"""
                                    <li style='margin: 10px 0; color: #212529;'>
                                        <div style='display: flex; align-items: center; gap: 10px;'>
                                            <strong>{obj_name}</strong>
                                            <br>
                                            <small style='color: #6c757d;'>Seed: {variant['seed']}</small>
                                        </div>
                                    </li>
                                """
                            html += "</ul>"
                        else:
                            html += "<p style='margin: 0; color: #6c757d;'>No variants selected yet</p>"
                        html += "</div>"
                        
                        # Create gallery items for selected variants
                        gallery_items = [
                            (variant['image_path'], f"{obj_name} (Seed: {variant['seed']})")
                            for obj_name, variant in selected_variants.items()
                        ]
                        
                        return gr.update(value=html), gallery_items

                    def generate_3d_model(variant_image, object_name):
                        """Generate a 3D model from the selected variant."""
                        logger.info(f"Starting 3D model generation for object: {object_name}")
                        if not variant_image or not object_name:
                            logger.warning("Missing variant image or object name")
                            return "Please select a variant first", None, None
                        
                        try:
                            # Create output directory for this object
                            output_dir = os.path.join(OUTPUT_DIR, object_name, "3d_assets")
                            logger.info(f"Creating output directory: {output_dir}")
                            os.makedirs(output_dir, exist_ok=True)
                            
                            # Generate 3D model
                            logger.info("Calling TRELLIS model for 3D generation")
                            success, message, outputs = self.generator.generate_3d_from_image(
                                image_path=variant_image,
                                output_dir=output_dir
                            )
                            
                            if not success:
                                logger.error(f"3D generation failed: {message}")
                                return message, None, None
                            
                            logger.info("3D model generation completed successfully")
                            logger.info(f"Output GLB path: {outputs['glb_path']}")
                            
                            # Update the GLB preview
                            glb_path, _ = update_preview(outputs['glb_path'])
                            return "Successfully generated 3D model", glb_path, outputs['glb_path']
                            
                        except Exception as e:
                            logger.error(f"Error generating 3D model: {str(e)}", exc_info=True)
                            error_msg = f"Error generating 3D model: {str(e)}"
                            glb_path, _ = clear_preview()
                            return error_msg, glb_path, None

                    # Add GLB preview components
                    model_output, download_btn = create_glb_preview()

                    # Connect the generate_3d_model function to the preview
                    generate_3d_btn.click(
                        generate_3d_model,
                        inputs=[current_variant_image, current_object_state],
                        outputs=[model_status, model_output, download_btn]
                    )

                    # Clear preview when clearing chat
                    clear_btn.click(
                        lambda: clear_preview(),
                        outputs=[model_output, download_btn]
                    )

                    def filter_variants_by_object(selected_object, all_variants):
                        """Filter variants to show only those for the selected object."""
                        logger.info(f"Filtering variants for object: {selected_object}")
                        if not selected_object or not all_variants:
                            return []
                        
                        # Get the variants for the selected object
                        if selected_object in all_variants:
                            variants = [
                                (v['image_path'], f"{selected_object} - Variant {i+1} (Seed: {v['seed']})")
                                for i, v in enumerate(all_variants[selected_object])
                            ]
                            logger.info(f"Found {len(variants)} variants for {selected_object}")
                            return variants
                        logger.warning(f"No variants found for {selected_object}")
                        return []

                    def generate_3d_for_all_selected(selected_variants):
                        """Generate 3D models for all selected variants."""
                        if not selected_variants:
                            return "No variants selected", None, None
                        
                        try:
                            results = []
                            for object_name, variant in selected_variants.items():
                                # Create output directory for this object
                                output_dir = os.path.join(OUTPUT_DIR, object_name, "3d_assets")
                                os.makedirs(output_dir, exist_ok=True)
                                
                                # Generate 3D model
                                success, message, outputs = self.generator.generate_3d_from_image(
                                    image_path=variant['image_path'],
                                    output_dir=output_dir
                                )
                                
                                if success:
                                    results.append(f"Successfully generated 3D model for {object_name}")
                                else:
                                    results.append(f"Failed to generate 3D model for {object_name}: {message}")
                            
                            # Update the GLB preview with the last generated model
                            if success:
                                glb_path, _ = update_preview(outputs['glb_path'])
                                return "\n".join(results), glb_path, outputs['glb_path']
                            else:
                                return "\n".join(results), None, None
                            
                        except Exception as e:
                            return f"Error during batch 3D generation: {str(e)}", None, None

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

                # Add handler for object dropdown to filter variants
                object_dropdown.change(
                    fn=filter_variants_by_object,
                    inputs=[object_dropdown, all_variants_state],
                    outputs=[variant_gallery]
                ).then(
                    fn=lambda x: x,  # Pass through the selected object
                    inputs=[object_dropdown],
                    outputs=[current_object_state]
                )

                # Enable 3D generation button when a variant is selected
                variant_gallery.select(
                    fn=on_variant_select,
                    inputs=[current_object_state, all_variants_state],
                    outputs=[generate_3d_btn, current_variant_image, selected_variant_index]
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
                    outputs=[selected_variants_display, selected_variants_gallery]
                ).then(
                    fn=lambda: gr.Button(interactive=True),
                    outputs=[generate_all_3d_btn]
                )

                # Add handler for generate all 3D models button
                generate_all_3d_btn.click(
                    fn=generate_3d_for_all_selected,
                    inputs=[selected_variants_state],
                    outputs=[model_status, model_output, download_btn]
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

                # Add handler for object dropdown to update prompt display
                object_dropdown.change(
                    fn=update_prompt_display,
                    inputs=[object_dropdown],
                    outputs=[prompt_display]
                )

        return demo

    def launch(self, **kwargs):
        """Launch the Gradio interface."""
        demo = self.create_interface()
        demo.launch(**kwargs)
