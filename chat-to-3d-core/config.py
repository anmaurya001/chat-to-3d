import os
from pathlib import Path

# Get the base directory of the project
BASE_DIR = Path(__file__).parent.parent.parent

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

# Other configuration settings
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
AGENT_BASE_URL = os.environ.get("AGENT_BASE_URL", "http://localhost:8000/v1")
# AGENT_BASE_URL = "http://localhost:8000/v1"
