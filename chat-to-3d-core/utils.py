import os
import json
import logging
from datetime import datetime
from pathlib import Path
from config import PROMPTS_FILE, LOG_LEVEL, LOG_FORMAT

# Set up logging
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


def save_prompts_to_json(scene_name, object_prompts, initial_description=None):
    """Save prompts to a JSON file in a structured format."""
    try:
        # Create the data structure
        new_scene = {
            "name": scene_name,
            "description": initial_description,
            "objects": [
                {"name": name, "prompt": prompt}
                for name, prompt in object_prompts.items()
            ],
            "timestamp": datetime.now().isoformat(),
        }

        # Delete existing file if it exists
        if os.path.exists(PROMPTS_FILE):
            os.remove(PROMPTS_FILE)
            logger.info(f"Deleted existing {PROMPTS_FILE}")

        # Create new data structure with just this scene
        data = {"scenes": [new_scene]}

        # Save data
        with open(PROMPTS_FILE, "w") as f:
            json.dump(data, f, indent=4)

        logger.info(f"Successfully saved prompts to {PROMPTS_FILE}")
        return True
    except Exception as e:
        logger.error(f"Failed to save prompts: {e}")
        return False


def delete_prompts_file():
    """Delete the prompts.json file if it exists."""
    try:
        if os.path.exists(PROMPTS_FILE):
            os.remove(PROMPTS_FILE)
            logger.info(f"Deleted existing {PROMPTS_FILE}")
            return True
    except Exception as e:
        logger.error(f"Failed to delete {PROMPTS_FILE}: {e}")
    return False


def ensure_output_directory(output_dir):
    """Ensure the output directory exists."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"Failed to create output directory: {e}")
        return False
