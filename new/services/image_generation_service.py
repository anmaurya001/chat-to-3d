"""Image generation service using SANA model."""

import os
import logging
import datetime
import torch
from diffusers import SanaSprintPipeline
from PIL import Image

logger = logging.getLogger(__name__)

class ImageGenerationService:
    def __init__(self):
        self.sana_pipeline = None
        self.is_loaded = False
        
    def load_sana_model(self):
        """Load the SANA model for image generation."""
        try:
            if self.is_loaded and self.sana_pipeline is not None:
                logger.info("SANA model already loaded")
                return True
                
            logger.info("Loading SANA model...")
            self.sana_pipeline = SanaSprintPipeline.from_pretrained(
                "Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers",
                torch_dtype=torch.bfloat16
            )
            self.sana_pipeline.to("cuda:0")
            self.is_loaded = True
            logger.info("Successfully loaded SANA model")
            return True
        except Exception as e:
            logger.error(f"Error loading SANA model: {e}")
            return False
    
    def generate_image_from_prompt(self, object_name, prompt, output_dir, seed=42):
        """Generate a single image from a prompt using SANA model."""
        try:
            if not self.load_sana_model():
                return False, "Failed to load SANA model", None
            
            # Format object name: lowercase and replace spaces with underscores
            formatted_object_name = object_name.lower().replace(" ", "_")
            
            # Generate image - SCM requires num_inference_steps=2
            image = self.sana_pipeline(
                prompt=prompt,
                num_inference_steps=2,  # SCM requirement
                generator=torch.Generator("cuda").manual_seed(seed)
            ).images[0]
            
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create filename using convention: objectname_seed_timestamp
            image_path = os.path.join(output_dir, f"{formatted_object_name}_{seed}_{timestamp}.png")
            
            # Save the image
            image.save(image_path)
            
            logger.info(f"Generated image: {image_path}")
            
            return True, f"Successfully generated image for {object_name}", image_path
            
        except Exception as e:
            logger.error(f"Error generating image: {e}")
            return False, f"Error generating image: {str(e)}", None
    
    def generate_images_for_objects(self, objects_data, output_dir="static/images/generated"):
        """Generate images for all objects in the gallery data."""
        try:
            if not self.load_sana_model():
                return False, "Failed to load SANA model", {}
            
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            generated_images = {}
            for obj in objects_data:
                object_name = obj["title"]
                prompt = obj["description"]
                
                logger.info(f"Generating image for: {object_name}")
                success, message, image_path = self.generate_image_from_prompt(
                    object_name, prompt, output_dir
                )
                
                if success and image_path:
                    generated_images[object_name] = image_path
                    logger.info(f"Generated image for {object_name}: {image_path}")
                else:
                    logger.error(f"Failed to generate image for {object_name}: {message}")
            
            return True, f"Generated {len(generated_images)} images", generated_images
            
        except Exception as e:
            logger.error(f"Error generating images for objects: {e}")
            return False, f"Error generating images: {str(e)}", {} 