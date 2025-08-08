"""Simple configuration for Chat-to-3D application.

This configuration file centralizes file paths and basic variables
without changing core functionality or UI elements.
"""

import os
from pathlib import Path

# Get the base directory of the project
BASE_DIR = Path(__file__).parent

# Use user's home directory
HOME_DIR = Path.home()
TRELLIS_DIR = HOME_DIR / ".trellis"  # Hidden directory
ASSETS_DIR = TRELLIS_DIR / "assets"
PROMPTS_DIR = TRELLIS_DIR / "prompts"
SCENE_DIR = TRELLIS_DIR / "scene"

# Create directories
ASSETS_DIR.mkdir(parents=True, exist_ok=True)
PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
SCENE_DIR.mkdir(parents=True, exist_ok=True)

# Define file paths
OUTPUT_DIR = ASSETS_DIR
PROMPTS_FILE = PROMPTS_DIR / "prompts.json"

# Static asset paths
STATIC_DIR = BASE_DIR / "static"
CSS_DIR = STATIC_DIR / "css"
JS_DIR = STATIC_DIR / "js"
IMAGES_DIR = STATIC_DIR / "images"
GENERATED_IMAGES_DIR = IMAGES_DIR / "generated"
ASSETS_APP_DIR = BASE_DIR / "assets"
MODELS_DIR = ASSETS_APP_DIR / "models"

# Create application directories
GENERATED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# File paths for static assets
CUSTOM_CSS_FILE = CSS_DIR / "custom.css"
CUSTOM_JS_FILE = JS_DIR / "custom.js"
NVIDIA_LOGO_FILE = IMAGES_DIR / "nvidia_logo.png"

NUM_OF_OBJECTS = 2

# Basic configuration settings
MAX_CARDS = 2  # Maximum number of cards in gallery
CARDS_PER_ROW = 2  # Number of cards per row in gallery
VRAM_THRESHOLD = 16  # VRAM threshold in GB for stopping the LLM Agent
DEFAULT_SEED = 42
DEFAULT_SPARSE_STEPS = 25
DEFAULT_SLAT_STEPS = 25
DEFAULT_CFG_STRENGTH = 7.5
MAX_PROMPT_LENGTH = 50
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# TRELLIS pipeline settings
TRELLIS_TEXT_LARGE_MODEL = "JeffreyXiang/TRELLIS-text-large"
TRELLIS_TEXT_BASE_MODEL = "JeffreyXiang/TRELLIS-text-base"
TRELLIS_IMAGE_LARGE_MODEL = "microsoft/TRELLIS-image-large"

# Model configuration
TRELLIS_MODEL_NAME_MAP = {
    "TRELLIS-text-large": TRELLIS_TEXT_LARGE_MODEL,
    "TRELLIS-text-base": TRELLIS_TEXT_BASE_MODEL
}
DEFAULT_TRELLIS_MODEL = "TRELLIS-text-large"

# Algorithm configuration
SPCONV_ALGO = "spconv2"

# INITIAL MESSAGE
INITIAL_MESSAGE = "Hello! I'm your helpful scene planning assistant. Please describe the scene you'd like to create."

# Agent settings
AGENT_MODEL = "meta/llama-3.1-8b-instruct"
AGENT_BASE_URL ="http://localhost:19002/v1"
TWO_D_PROMPT_LENGTH = 30

# Simple helper functions
def get_static_paths():
    """Get static asset paths."""
    return {
        "css": CUSTOM_CSS_FILE,
        "js": CUSTOM_JS_FILE,
        "nvidia_logo": NVIDIA_LOGO_FILE,
        "generated_images": GENERATED_IMAGES_DIR,
        "models": MODELS_DIR
    }

def get_output_paths():
    """Get output paths."""
    return {
        "generated_images": GENERATED_IMAGES_DIR,
        "models": MODELS_DIR,
        "assets": ASSETS_DIR,
        "prompts": PROMPTS_DIR,
        "scene": SCENE_DIR
    } 