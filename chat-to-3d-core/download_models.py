import sys
from pathlib import Path
import gc

# --- Calculate important paths ---
# Path to the directory containing run.py (chat-to-3d-core)
current_dir = Path(__file__).parent
print(f"Current script directory: {current_dir}")

# Path to the project root (F:\poc\chat-to-3d)
# This is two levels up from the script's location
project_root = current_dir.parent
print(f"Project root: {project_root}")

# Path to the Trellis submodule directory (F:\poc\chat-to-3d\trellis)
trellis_submodule_path = project_root / "trellis"
trellis_submodule_path_str = str(trellis_submodule_path.resolve()) # Get the absolute path string
print(f"Trellis submodule path: {trellis_submodule_path_str}")

# Add the project root to sys.path.
# This allows importing 'chat_to_3d_core' as a top-level package.
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
    print(f"Added project root to sys.path: {project_root}")


# Add the trellis submodule directory to sys.path.
# This directory contains the actual 'trellis' Python package folder (trellis/trellis)
if trellis_submodule_path_str not in sys.path:
    sys.path.insert(0, trellis_submodule_path_str)
    print(f"Added trellis submodule path to sys.path: {trellis_submodule_path_str}")


import torch
from trellis.pipelines import TrellisImageTo3DPipeline
from diffusers import SanaSprintPipeline
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_models():
    try:
        logger.info("Starting model downloads...")
        
        # Download Trellis model
        logger.info("Downloading Trellis model...")
        trellis_model = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
        del trellis_model
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("Trellis model downloaded successfully!")
        
        # Download Sana Sprint model
        logger.info("Downloading Sana Sprint model...")
        sana_model = SanaSprintPipeline.from_pretrained(
            "Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers",
            torch_dtype=torch.bfloat16
        )
        del sana_model
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("Sana Sprint model downloaded successfully!")
        
        logger.info("All models downloaded successfully!")
        return True
    except Exception as e:
        logger.error(f"Error downloading models: {e}")
        return False

if __name__ == "__main__":
    download_models() 