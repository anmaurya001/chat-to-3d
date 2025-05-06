import os
import json
import logging
import shutil
import socket
import threading
import os
import numpy as np
import torch
import gc
from pathlib import Path
from trellis.pipelines import TrellisTextTo3DPipeline
from trellis.utils import postprocessing_utils, render_utils
import imageio
from config import (
    SPCONV_ALGO,
    DEFAULT_SEED,
    DEFAULT_SPARSE_STEPS,
    DEFAULT_SLAT_STEPS,
    DEFAULT_CFG_STRENGTH,
    OUTPUT_DIR,
    LOG_LEVEL,
    LOG_FORMAT,
    TRELLIS_MODEL_NAME_MAP,
    DEFAULT_TRELLIS_MODEL,
)

# Set up logging
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


class AssetGenerator:
    def __init__(self, default_model=DEFAULT_TRELLIS_MODEL):
        os.environ["SPCONV_ALGO"] = SPCONV_ALGO
        self.pipeline = None
        self.current_model = None
        self.termination_thread = None
        self.start_termination_server()
        
        # Load the default model during initialization
        self.load_model(default_model)
    
    def start_termination_server(self):
        """Start the termination server in a separate thread"""
        def handle_termination():
            server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind(('localhost', 12345))
            server.listen(1)
            logger.info("Termination server started on port 12345")

            while True:
                try:
                    conn, addr = server.accept()
                    with conn:
                        data = conn.recv(1024)
                        if data == b'terminate':
                            logger.info("Termination signal received")
                            # Send PID to client
                            pid = os.getpid()
                            logger.info(f"Sending PID {pid} to client")
                            conn.send(f"terminating:{pid}".encode())
                            # Let client handle the termination
                        else:
                            conn.send(b'error: invalid command')
                except Exception as e:
                    logger.error(f"Error handling connection: {e}")

        self.termination_thread = threading.Thread(target=handle_termination)
        self.termination_thread.start()

    def cleanup(self):
        """Clean up the current model"""
        if self.pipeline is not None:
            try:
                # Move to CPU first
                if hasattr(self.pipeline, 'cuda'):
                    self.pipeline.cpu()

                # Clear internal tensors if any
                if hasattr(self.pipeline, '__dict__'):
                    for k, v in list(vars(self.pipeline).items()):
                        if torch.is_tensor(v):
                            setattr(self.pipeline, k, None)
                            del v
                # Delete the model reference
                del self.pipeline
                self.pipeline = None
                self.current_model = None

                # Force garbage collection
                gc.collect()

                # Empty unused memory from GPU cache
                torch.cuda.empty_cache()
                logger.info("Successfully cleaned up pipeline")
            except Exception as e:
                logger.error(f"Error during cleanup: {e}")

    def load_model(self, model_name):
        """Load a specific TRELLIS model"""
        try:
            # Clean up existing model if any
            if self.pipeline is not None:
                if self.current_model != model_name:
                    logger.info(f"Switching from {self.current_model} to {model_name}")
                    self.cleanup()
                else:
                    logger.info(f"Model {model_name} is already loaded")
                    return True

            # Load new model
            logger.info(f"Loading model: {model_name}")
            self.pipeline = TrellisTextTo3DPipeline.from_pretrained(TRELLIS_MODEL_NAME_MAP[model_name])
            self.pipeline.cuda()
            self.current_model = model_name
            logger.info(f"Successfully loaded model: {model_name}")
            return True
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            return False

    def __del__(self):
        """Cleanup when the object is destroyed"""
        self.cleanup()
        if self.termination_thread:
            self.termination_thread.join(timeout=1.0)

    def generate_preview_image(self, mesh, output_path):
        """Generate a preview image for the mesh and save it"""
        try:
            # Generate a high-quality preview
            mesh_img = render_utils.render_snapshot(
                mesh
            )['normal']
            
            if isinstance(mesh_img, list) and len(mesh_img) > 0:
                # Get the first image
                preview_img = mesh_img[0]
                
                # Convert to uint8 if needed
                if preview_img.dtype != np.uint8:
                    preview_img = (preview_img * 255).astype(np.uint8)
                
                # Make sure image is in RGB format
                if len(preview_img.shape) == 3 and preview_img.shape[2] > 3:
                    preview_img = preview_img[:, :, :3]
                
                # Save the preview image
                imageio.imwrite(output_path, preview_img)
                logger.info(f"Generated preview: {output_path}")
                return True
            else:
                logger.warning(f"Failed to generate preview: No image data")
                return False
        except Exception as e:
            logger.error(f"Error generating preview image: {e}")
            return False

    def generate_assets(
        self,
        scene_name,
        object_name,
        prompt,
        output_dir,
        model_name="TRELLIS-text-large",  # Default to large model
        seed=DEFAULT_SEED,
        sparse_steps=DEFAULT_SPARSE_STEPS,
        slat_steps=DEFAULT_SLAT_STEPS,
    ):
        """Generate 3D assets for a single object."""
        try:
            # Ensure the correct model is loaded
            if not self.load_model(model_name):
                return False, f"Failed to load model {model_name}", None

            outputs = self.pipeline.run(
                prompt,
                seed=seed,
                sparse_structure_sampler_params={
                    "steps": sparse_steps,
                    "cfg_strength": DEFAULT_CFG_STRENGTH,
                },
                slat_sampler_params={
                    "steps": slat_steps,
                    "cfg_strength": DEFAULT_CFG_STRENGTH,
                },
            )

            # Handle versioning for the base filename
            base_filename = object_name
            version = 0
            while True:
                if version == 0:
                    # First try without version number
                    glb_path = os.path.join(output_dir, f"{base_filename}.glb")
                    if not os.path.exists(glb_path):
                        break
                    version = 1
                else:
                    # Try with version number
                    glb_path = os.path.join(output_dir, f"{base_filename}_v{version}.glb")
                    if not os.path.exists(glb_path):
                        break
                    version += 1

            # Generate the GLB file
            glb = postprocessing_utils.to_glb(
                outputs["gaussian"][0],
                outputs["mesh"][0],
                simplify=0.95,
                texture_size=1024,
            )
            glb.export(glb_path)
            
            return True, f"Successfully generated assets for {object_name}", glb_path
        except Exception as e:
            return False, f"Error generating assets for {object_name}: {str(e)}", None

    def process_scene(
        self,
        scene_data,
        output_dir,
        progress,
        seed=DEFAULT_SEED,
        sparse_steps=DEFAULT_SPARSE_STEPS,
        slat_steps=DEFAULT_SLAT_STEPS,
        model_name="TRELLIS-text-large",
    ):
        """Process all objects in a scene."""
        scene_name = scene_data["name"]
        results = [f"Processing scene: {scene_name}"]
        generated_assets = []

        total_objects = len(scene_data["objects"])
        for i, obj in enumerate(scene_data["objects"], 1):
            progress((i - 1) / total_objects, desc=f"Generating {obj['name']}...")
            success, message, glb_path = self.generate_assets(
                scene_name,
                obj["name"],
                obj["prompt"],
                output_dir,
                model_name,
                seed,
                sparse_steps,
                slat_steps,
            )
            results.append(message)
            if success and glb_path:
                generated_assets.append(glb_path)
            progress(i / total_objects, desc=f"Completed {obj['name']}")

        if generated_assets:
            results.append("\nGenerated assets in this scene:")
            for asset_path in generated_assets:
                results.append(f"- {asset_path}")

        return results

    def generate_all_assets(
        self,
        prompts_file,
        output_dir,
        delete_existing=False,
        seed=DEFAULT_SEED,
        sparse_steps=DEFAULT_SPARSE_STEPS,
        slat_steps=DEFAULT_SLAT_STEPS,
        model_name="TRELLIS-text-large",
        progress=None,
    ):
        """Generate assets for all scenes in the prompts file."""
        try:
            if delete_existing and os.path.exists(output_dir):
                progress(0, desc="Deleting existing output directory...")
                shutil.rmtree(output_dir)
                progress(0.1, desc="Output directory deleted")

            os.makedirs(output_dir, exist_ok=True)

            with open(prompts_file, "r") as f:
                data = json.load(f)

            all_results = []
            total_scenes = len(data["scenes"])

            for i, scene in enumerate(data["scenes"], 1):
                progress(i / total_scenes, desc=f"Processing scene {i}/{total_scenes}")
                results = self.process_scene(
                    scene, output_dir, progress, seed, sparse_steps, slat_steps, model_name
                )
                all_results.extend(results)
                progress(i / total_scenes, desc=f"Completed scene {i}/{total_scenes}")

            # all_results.append("\n=== Summary of Generated Assets ===")
            # for root, _, files in os.walk(output_dir):
            #     for file in files:
            #         if file.endswith(".glb"):
            #             all_results.append(f"- {os.path.join(root, file)}")

            return "\n".join(all_results)
        except Exception as e:
            return f"Error processing prompts file: {str(e)}"

    #FOR DEBUGGING ONLY
    def debug_switch_model(self, model_name):
        """Debug function to switch models and check VRAM usage"""
        try:
            import torch
            
            # Get initial VRAM usage
            if torch.cuda.is_available():
                initial_vram = torch.cuda.memory_allocated()
                logger.info(f"Initial VRAM usage: {initial_vram / 1024**3:.2f} GB")
            
            # Clean up current model
            self.cleanup()
            
            # Get VRAM after cleanup
            if torch.cuda.is_available():
                after_cleanup_vram = torch.cuda.memory_allocated()
                logger.info(f"VRAM after cleanup: {after_cleanup_vram / 1024**3:.2f} GB")
            
            # Load new model
            success = self.load_model(model_name)
            
            # Get VRAM after loading
            if torch.cuda.is_available():
                after_load_vram = torch.cuda.memory_allocated()
                logger.info(f"VRAM after loading {model_name}: {after_load_vram / 1024**3:.2f} GB")
            
            return success, f"Model switched to {model_name}. VRAM usage logged."
        except Exception as e:
            return False, f"Error during model switch: {str(e)}"

