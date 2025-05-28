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
from typing import Tuple
from trellis.pipelines import TrellisTextTo3DPipeline, TrellisImageTo3DPipeline
from trellis.utils import postprocessing_utils, render_utils
from easydict import EasyDict as edict
from diffusers import SanaSprintPipeline
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
from trellis.representations import Gaussian, MeshExtractResult
from PIL import Image
import datetime

# Set up logging
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


class AssetGenerator:
    def __init__(self, default_model=DEFAULT_TRELLIS_MODEL):
        os.environ["SPCONV_ALGO"] = SPCONV_ALGO
        self.pipeline = None
        self.sana_pipeline = None
        self.trellis_model = None
        self.current_model = None
        self.termination_thread = None
        self.start_termination_server()
        
        # Load the default model during initialization
        #self.load_model(default_model)
        #self.load_sana_model()
        self.initialize_trellis()
    
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


    def cleanup_sana_pipeline(self):
        """Clean up the current model"""
        if self.sana_pipeline is not None:
            try:
                # Move to CPU first
                if hasattr(self.sana_pipeline, 'cuda'):
                    self.sana_pipeline.cpu()

                # Clear internal tensors if any
                if hasattr(self.sana_pipeline, '__dict__'):
                    for k, v in list(vars(self.sana_pipeline).items()):
                        if torch.is_tensor(v):
                            setattr(self.sana_pipeline, k, None)
                            del v
                # Delete the model reference
                del self.sana_pipeline
                self.sana_pipeline = None

                # Force garbage collection
                gc.collect()

                # Empty unused memory from GPU cache
                torch.cuda.empty_cache()
                logger.info("Successfully cleaned up SANA pipeline")
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

    def load_sana_model(self):
        """Load the SANA model for image generation."""
        try:
            start_time = datetime.datetime.now()
            logger.info("Loading SANA model...")
            self.sana_pipeline = SanaSprintPipeline.from_pretrained(
                "Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers",
                torch_dtype=torch.bfloat16
            )
            self.sana_pipeline.to("cuda:0")
            end_time = datetime.datetime.now()
            load_time = (end_time - start_time).total_seconds()
            logger.info(f"Successfully loaded SANA model in {load_time:.2f} seconds")
            return True
        except Exception as e:
            logger.error(f"Error loading SANA model: {e}")
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

    def generate_variants(
        self,
        object_name,
        prompt,
        output_dir,
        num_variants=3,
        seed=DEFAULT_SEED,
        num_inference_steps=2
    ):
        """Generate multiple image variants using SANA model."""
        try:
            if self.sana_pipeline is None:
                logger.info("SANA model not loaded, loading now...")
                if not self.load_sana_model():
                    return False, "Failed to load SANA model", []

            # Format object name: lowercase and replace spaces with underscores
            formatted_object_name = object_name.lower().replace(" ", "_")

            variants = []
            for i in range(num_variants):
                # Use different seeds for each variant
                variant_seed = seed + i
                
                # Generate image
                image = self.sana_pipeline(
                    prompt=prompt,
                    num_inference_steps=num_inference_steps,
                    generator=torch.Generator("cuda").manual_seed(variant_seed)
                ).images[0]
                
                # Generate timestamp
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # Create filename using convention: objectname_seed_timestamp
                image_path = os.path.join(output_dir, f"{formatted_object_name}_{variant_seed}_{timestamp}.png")
                
                # Save the image
                image.save(image_path)
                
                variants.append({
                    "image_path": image_path,
                    "seed": variant_seed,
                    "timestamp": timestamp,
                    "object_name": object_name,  # Keep original object name in the variant data
                    "formatted_object_name": formatted_object_name
                })

            return True, f"Successfully generated {num_variants} image variants", variants
        except Exception as e:
            return False, f"Error generating image variants: {str(e)}", []

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

    def initialize_trellis(self):
        """Initialize the TRELLIS image-to-3D model."""
        try:
            start_time = datetime.datetime.now()
            logger.info("Initializing TRELLIS model...")
            self.trellis_model = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
            self.trellis_model.cuda()
            end_time = datetime.datetime.now()
            load_time = (end_time - start_time).total_seconds()
            logger.info(f"Successfully initialized TRELLIS model in {load_time:.2f} seconds")
        except Exception as e:
            logger.error(f"Failed to initialize TRELLIS model: {e}")
            raise

    def pack_state(self, gs: Gaussian, mesh: MeshExtractResult) -> dict:
        return {
            'gaussian': {
                **gs.init_params,
                '_xyz': gs._xyz.cpu().numpy(),
                '_features_dc': gs._features_dc.cpu().numpy(),
                '_scaling': gs._scaling.cpu().numpy(),
                '_rotation': gs._rotation.cpu().numpy(),
                '_opacity': gs._opacity.cpu().numpy(),
            },
            'mesh': {
                'vertices': mesh.vertices.cpu().numpy(),
                'faces': mesh.faces.cpu().numpy(),
            },
        }
    
    
    def unpack_state(self, state: dict) -> Tuple[Gaussian, edict, str]:
        gs = Gaussian(
            aabb=state['gaussian']['aabb'],
            sh_degree=state['gaussian']['sh_degree'],
            mininum_kernel_size=state['gaussian']['mininum_kernel_size'],
            scaling_bias=state['gaussian']['scaling_bias'],
            opacity_bias=state['gaussian']['opacity_bias'],
            scaling_activation=state['gaussian']['scaling_activation'],
        )
        gs._xyz = torch.tensor(state['gaussian']['_xyz'], device='cuda')
        gs._features_dc = torch.tensor(state['gaussian']['_features_dc'], device='cuda')
        gs._scaling = torch.tensor(state['gaussian']['_scaling'], device='cuda')
        gs._rotation = torch.tensor(state['gaussian']['_rotation'], device='cuda')
        gs._opacity = torch.tensor(state['gaussian']['_opacity'], device='cuda')
        
        mesh = edict(
            vertices=torch.tensor(state['mesh']['vertices'], device='cuda'),
            faces=torch.tensor(state['mesh']['faces'], device='cuda'),
        )
        
        return gs, mesh

    def extract_glb(self, gaussian, mesh, output_dir, model_name, mesh_simplify=0.95, texture_size=1024):
        """
        Extract a GLB file from the 3D model.

        Args:
            gaussian: The gaussian representation
            mesh: The mesh representation
            output_dir (str): Directory to save the GLB file
            mesh_simplify (float): The mesh simplification factor
            texture_size (int): The texture resolution

        Returns:
            str: The path to the extracted GLB file
        """
        try:
            logger.info("Extracting GLB file")
            glb = postprocessing_utils.to_glb(
                gaussian, 
                mesh, 
                simplify=mesh_simplify, 
                texture_size=texture_size, 
                verbose=False
            )
            glb_path = os.path.join(output_dir, f'{model_name}.glb')
            glb.export(glb_path)
            torch.cuda.empty_cache()
            return glb_path
        except Exception as e:
            logger.error(f"Error extracting GLB: {str(e)}", exc_info=True)
            raise

    def generate_3d_from_image(self, object_name, image_path, output_dir,image_seed, seed=0):
        """Generate a 3D model from an image using TRELLIS."""
        try:
            #Cleanup the SANA pipeline      
            if self.sana_pipeline is not None:
                logger.info("Cleaning up SANA pipeline")
                self.cleanup_sana_pipeline()
            else:
                logger.info("SANA pipeline not loaded, skipping cleanup")
            formatted_object_name = object_name.lower().replace(" ", "_")
            logger.info(f"Starting 3D generation from image: {image_path} for object: {formatted_object_name}")
            
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Output directory: {output_dir}")
            
            # Load and preprocess image
            logger.info("Loading input image")
            image = Image.open(image_path)
            logger.info(f"Original image size: {image.size}")
            
            # Resize to 512x512
            target_size = (512, 512)
            image = image.resize(target_size, Image.Resampling.LANCZOS)
            logger.info(f"Reshaped image size: {image.size}")
            
            # Apply TRELLIS preprocessing
            logger.info("Applying TRELLIS preprocessing")
            image = self.trellis_model.preprocess_image(image)
            logger.info("Preprocessing completed")
            
            # Generate 3D model
            logger.info("Initializing TRELLIS model run")
            logger.info("Running sparse structure sampling...")
            outputs = self.trellis_model.run(
                image,
                seed=seed,
                formats=["gaussian", "mesh"],
                preprocess_image=False,
                sparse_structure_sampler_params={
                    "steps": 12,
                    "cfg_strength": 7.5,
                },
                slat_sampler_params={
                    "steps": 12,
                    "cfg_strength": 3.0,
                },
            )
            logger.info("TRELLIS model run completed")
            
            # Clear CUDA cache before GLB extraction
            logger.info("Clearing CUDA cache before GLB extraction")
            torch.cuda.empty_cache()
            
            # Pack and unpack state for GLB extraction
            state = self.pack_state(outputs['gaussian'][0], outputs['mesh'][0])
            gs, mesh = self.unpack_state(state)

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"{formatted_object_name}_{image_seed}_{timestamp}"
            
            # Extract GLB file
            glb_path = self.extract_glb(
                gs,
                mesh,
                output_dir,
                model_name
            )
            
            logger.info("3D generation completed successfully")
            return True, "Successfully generated 3D model", {
                'glb_path': glb_path,
                'model_name': model_name,
                'image_seed': image_seed,
                'timestamp': timestamp
            }
            
        except Exception as e:
            logger.error(f"Error generating 3D model: {str(e)}", exc_info=True)
            # Clear CUDA cache in case of error
            torch.cuda.empty_cache()
            return False, f"Error generating 3D model: {str(e)}", None

