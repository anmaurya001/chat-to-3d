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
                            
                            # Generation controls
                            generate_btn = gr.Button("Generate Variants")
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
                            generate_3d_btn = gr.Button("Generate 3D Model", interactive=False)
                            selected_variants = gr.HTML(
                                label="Selected Variants",
                                value="<div style='background-color: #f8f9fa; padding: 15px; border-radius: 8px; border: 1px solid #dee2e6;'>"
                                      "<p style='margin: 0; color: #6c757d;'>No variants selected yet</p>"
                                      "</div>"
                            )
                            
                            # 3D Preview
                            gr.Markdown("### 3D Model Preview")
                            # model_preview = gr.Video(
                            #     label="3D Model Preview",
                            #     autoplay=True,
                            #     loop=True,
                            #     height=300
                            # )
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

                    def generate_variants_for_object(
                        object_name,
                        seed,
                        num_variants
                    ):
                        """Generate variants for the selected object."""
                        try:
                            if not object_name:
                                logger.warning("No object selected")
                                return (
                                    "Please select an object first",
                                    None,
                                    []
                                )
                            
                            logger.info(f"Generating variants for object: {object_name}")
                            
                            if not os.path.exists(PROMPTS_FILE):
                                logger.error(f"Prompts file not found at: {PROMPTS_FILE}")
                                return (
                                    "Please generate prompts first in the Chat & Generate Prompts tab",
                                    None,
                                    []
                                )
                            
                            # Read prompts file
                            logger.info(f"Reading prompts file for object: {object_name}")
                            with open(PROMPTS_FILE, 'r') as f:
                                data = json.load(f)
                                logger.info(f"Prompts file contents: {json.dumps(data, indent=2)}")
                            
                            # Find the object's prompt
                            object_prompt = None
                            for scene in data['scenes']:
                                for obj in scene['objects']:
                                    logger.info(f"Checking object: {obj['name']} against {object_name}")
                                    if obj['name'] == object_name:
                                        object_prompt = obj['prompt']
                                        logger.info(f"Found prompt for {object_name}: {object_prompt}")
                                        break
                                if object_prompt:
                                    break
                            
                            if not object_prompt:
                                logger.error(f"Object '{object_name}' not found in prompts file")
                                return (
                                    f"Object '{object_name}' not found in prompts file",
                                    None,
                                    []
                                )
                            
                            # Create output directory for this object
                            object_dir = os.path.join(OUTPUT_DIR, object_name)
                            os.makedirs(object_dir, exist_ok=True)
                            logger.info(f"Created output directory: {object_dir}")
                            
                            # Generate variants using SANA
                            logger.info(f"Generating variants for {object_name} with prompt: {object_prompt}")
                            success, message, variants = self.generator.generate_variants(
                                prompt=object_prompt,
                                output_dir=object_dir,
                                num_variants=num_variants,
                                seed=seed
                            )
                            
                            if not success:
                                logger.error(f"Failed to generate variants: {message}")
                                return (
                                    f"Error: {message}",
                                    None,
                                    []
                                )
                            
                            logger.info(f"Successfully generated {len(variants)} variants")
                            # Update gallery with generated variants
                            gallery_items = [
                                (v['image_path'], f"Variant {i+1} (Seed: {v['seed']})")
                                for i, v in enumerate(variants)
                            ]
                            
                            return (
                                f"Successfully generated {len(variants)} variants",
                                object_name,
                                variants
                            )
                        except Exception as e:
                            logger.error(f"Error during generation: {str(e)}")
                            return (
                                f"Error during generation: {str(e)}",
                                None,
                                []
                            )

                    def on_variant_select(variants, evt: gr.SelectData):
                        """Handle variant selection in gallery."""
                        logger.info(f"Variant selected: {evt.index}")
                        if not variants:
                            return gr.Button(interactive=False), None, None
                        
                        if evt.index is not None and evt.index < len(variants):
                            return gr.Button(interactive=True), variants[evt.index]['image_path'], evt.index
                        return gr.Button(interactive=False), None, None

                    def select_variant(object_name, variants, selected_variants, selected_idx):
                        """Handle variant selection."""
                        logger.info(f"Selecting variant with index: {selected_idx}")
                        if not object_name or not variants:
                            return gr.update(value="No variants available to select")
                        
                        if selected_idx is None or selected_idx >= len(variants):
                            return gr.update(value="Please select a variant from the gallery")
                        
                        selected_variant = variants[selected_idx]
                        
                        # Update selected variants
                        selected_variants[object_name] = selected_variant
                        
                        # Format the selected variants display
                        html = "<div style='background-color: #f8f9fa; padding: 15px; border-radius: 8px; border: 1px solid #dee2e6;'>"
                        if selected_variants:
                            html += "<ul style='margin: 0; padding-left: 20px;'>"
                            for obj_name, variant in selected_variants.items():
                                html += f"""
                                    <li style='margin: 5px 0; color: #212529;'>
                                        <strong>{obj_name}</strong>
                                        <br>
                                        <small style='color: #6c757d;'>Seed: {variant['seed']}</small>
                                    </li>
                                """
                            html += "</ul>"
                        else:
                            html += "<p style='margin: 0; color: #6c757d;'>No variants selected yet</p>"
                        html += "</div>"
                        
                        return gr.update(value=html)

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

                # Set up event handlers for variant generation
                generate_btn.click(
                    fn=generate_variants_for_object,
                    inputs=[
                        object_dropdown,
                        seed,
                        num_variants
                    ],
                    outputs=[
                        generation_status,
                        current_object_state,
                        current_variants_state
                    ]
                ).then(
                    fn=lambda variants: [(v['image_path'], f"Variant {i+1} (Seed: {v['seed']})") for i, v in enumerate(variants)] if variants else [],
                    inputs=[current_variants_state],
                    outputs=[variant_gallery]
                )

                # Handle variant selection
                select_variant_btn.click(
                    fn=select_variant,
                    inputs=[
                        current_object_state,
                        current_variants_state,
                        selected_variants_state,
                        selected_variant_index
                    ],
                    outputs=[selected_variants]
                )

                # Enable 3D generation button when a variant is selected
                variant_gallery.select(
                    fn=on_variant_select,
                    inputs=[current_variants_state],
                    outputs=[generate_3d_btn, current_variant_image, selected_variant_index]
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
