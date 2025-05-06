import gradio as gr
from agent import ScenePlanningAgent
from generator import AssetGenerator
from utils import delete_prompts_file
from config import (
    DEFAULT_SEED,
    DEFAULT_SPARSE_STEPS,
    DEFAULT_SLAT_STEPS,
    OUTPUT_DIR,
    PROMPTS_FILE,
    INITIAL_MESSAGE,
    DEFAULT_TRELLIS_MODEL,
)


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

                    # Set up event handlers
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

                    def on_generate_prompts(chat_history, accepted_objects):
                        """Generate 3D prompts for the accepted objects."""
                        if not accepted_objects:
                            return gr.update(
                                value="<div style='background-color: #f8f9fa; padding: 15px; border-radius: 8px; border: 1px solid #dee2e6;'>"
                                      "<p style='margin: 0; color: #dc3545;'>Please select at least one object from the suggested list before generating prompts.</p>"
                                      "</div>"
                            ), ""
                        
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
                                ), ""
                            
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
                            
                            return gr.update(value=prompts_html), generation_prompt
                        except Exception as e:
                            return gr.update(
                                value=f"<div style='background-color: #f8f9fa; padding: 15px; border-radius: 8px; border: 1px solid #dee2e6;'>"
                                      f"<p style='margin: 0; color: #dc3545;'>Error generating prompts: {str(e)}</p>"
                                      f"</div>"
                            ), ""

                    # Add event handler for generate prompts button
                    generate_prompts_btn.click(
                        on_generate_prompts,
                        [chatbot, accepted_objects_state],
                        [prompts_output, generation_prompt_display]
                    )

                with gr.TabItem("Generate 3D Assets"):
                    # Info section at the top
                    with gr.Group():
                        gr.Markdown("### File Locations")
                        with gr.Row():
                            prompts_file = gr.Textbox(
                                label="Prompts JSON File",
                                value=str(PROMPTS_FILE),
                                interactive=False,
                                show_copy_button=True
                            )
                            output_dir = gr.Textbox(
                                label="Assets Directory",
                                value=str(OUTPUT_DIR),
                                interactive=False,
                                show_copy_button=True
                            )

                    with gr.Row():
                        # Model selection - default to base model
                        model_choice = gr.Radio(
                            choices=["TRELLIS-text-large", "TRELLIS-text-base"],
                            value=DEFAULT_TRELLIS_MODEL,  # Use default from config
                            label="TRELLIS Model",
                            interactive=True
                        )
                        
                        seed = gr.Number(
                            label="Random Seed", value=DEFAULT_SEED, interactive=True
                        )

                    with gr.Row():
                        sparse_steps = gr.Number(
                            label="Sparse Structure Steps",
                            value=DEFAULT_SPARSE_STEPS,
                            interactive=True,
                        )
                        slat_steps = gr.Number(
                            label="Slat Sampler Steps",
                            value=DEFAULT_SLAT_STEPS,
                            interactive=True,
                        )

                    delete_existing = gr.Checkbox(
                        label="Delete existing output directory", value=False
                    )

                    generate_btn = gr.Button("Generate Assets")
                    output = gr.Textbox(label="Generation Progress", lines=10)

                    # Debug section for model switching
                    # with gr.Group():
                    #     gr.Markdown("### Debug Model Switching")
                    #     debug_model_choice = gr.Radio(
                    #         choices=["TRELLIS-text-large", "TRELLIS-text-base"],
                    #         value=DEFAULT_TRELLIS_MODEL,
                    #         label="Switch Model",
                    #         interactive=True
                    #     )
                    #     debug_switch_btn = gr.Button("Switch Model & Check VRAM")
                    #     debug_output = gr.Textbox(label="Debug Output", lines=5)

                    #     def debug_switch_model(model_name):
                    #         success, message = self.generator.debug_switch_model(model_name)
                    #         return message

                    #     debug_switch_btn.click(
                    #         debug_switch_model,
                    #         inputs=[debug_model_choice],
                    #         outputs=debug_output
                    #     )


                    def generate_assets_with_progress(prompts_file, output_dir, delete_existing, model_choice, seed, sparse_steps, slat_steps):
                        try:
                            # Disable button immediately
                            disable_btn = gr.update(interactive=False)

                            # Show intermediate state
                            yield disable_btn, "Starting generation..."

                            # Do the actual generation
                            result = self.generator.generate_all_assets(
                                prompts_file,
                                output_dir,
                                delete_existing,
                                seed,
                                sparse_steps,
                                slat_steps,
                                model_name=model_choice,
                                progress=gr.Progress()
                            )

                            # Re-enable button & return result
                            enable_btn = gr.update(interactive=True)
                            yield enable_btn, result

                        except Exception as e:
                            enable_btn = gr.update(interactive=True)
                            yield enable_btn, f"Error during generation: {str(e)}"

                    generate_btn.click(
                        fn=generate_assets_with_progress,
                        inputs=[prompts_file, output_dir, delete_existing, model_choice, seed, sparse_steps, slat_steps],
                        outputs=[generate_btn, output]
                    )


        return demo

    def launch(self, **kwargs):
        """Launch the Gradio interface."""
        demo = self.create_interface()
        demo.launch(**kwargs)
