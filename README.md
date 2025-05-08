# chat-to-3d

An application that combines natural language processing with 3D asset generation using TRELLIS.

## Features

- Interactive chat interface for scene planning
- AI-assisted object suggestion and layout
- Automatic 3D asset generation from text prompts
- Blender auto-import functionality for generated assets
- VRAM management with process termination

## Installation Methods

There are two ways to install and run the application:

### Method 1: Docker Installation (Recommended)

1. Clone this repository:
```bash
git clone --recursive https://github.com/anmaurya001/chat-to-3d.git

# If you forgot --recursive during clone:
# git submodule update --init --recursive
```

2. Build the Docker image in wsl:
```bash
# get into wsl shell and navigate to the repo
Wsl -d NVIDIA-Workbench -u root   
cd chat-to-3d

# Build with default settings
podman build -t chat-to-3d-app .

# Or force a rebuild by providing a unique value for FORCE_REBUILD
podman build --build-arg FORCE_REBUILD=$(date +%s) -t chat-to-3d-app .
```


### Method 2: Manual Installation

#### Prerequisites

### Windows Installation (Required for TRELLIS) - [TRELLIS Windows Installation Guide](https://github.com/ericcraft-mh/TRELLIS-install-windows) for Windows setup instructions

Before installing this project, you need to set up the development environment on Windows:

1. Install Visual Studio Build Tools 2022:
```bash
vs_buildtools.exe --norestart --passive --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended
```

2. Install Python 3.11.9

#### Installation Steps

1. Clone this repository:
```bash
git clone --recursive https://github.com/anmaurya001/chat-to-3d.git

# If you forgot --recursive during clone:
# git submodule update --init --recursive
```

2. Create and activate a virtual environment in the  directory:
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
```

3. Install dependencies:
```bash
# Update pip and install build tools
python -m pip install --upgrade pip wheel
python -m pip install setuptools==75.8.2

# Install TRELLIS requirements from Windows installation guide
pip install -r requirements-torch.txt
pip install -r requirements-other.txt

# Install POC requirements
pip install -r requirements.txt
```

## Usage

### Method 1: Docker Installation

1. Start the LLM NIM
```bash
cd nim_llm
pip install -r requirements.txt
python run_llama.py
```

2. Start the application in wsl:
```bash
# get into wsl shell 
Wsl -d NVIDIA-Workbench -u root

# Run the following in wsl
WINDOWS_USER=$(cmd.exe /c echo %USERNAME% | tr -d '\r')
export LOCAL_TRELLIS_CACHE="/mnt/c/Users/${WINDOWS_USER}/.trellis/"
export LOCAL_MODEL_CACHE=~/.cache/3d-guided-bp
mkdir -p "$LOCAL_TRELLIS_CACHE"
mkdir -p "$LOCAL_MODEL_CACHE"
chmod -R a+w "$LOCAL_TRELLIS_CACHE"
chmod -R a+w "$LOCAL_MODEL_CACHE"
podman run -it --rm --device nvidia.com/gpu=all -p 7860:7860 -v "$LOCAL_MODEL_CACHE:/home/user/.cache" -v  "$LOCAL_TRELLIS_CACHE:/home/user/.trellis/" localhost/chat-to-3d-app:latest
```

3. Open your browser to http://localhost:7860

4. To terminate the application:
   - Press Ctrl+C in the terminal where podman is running

### Method 2: Manual Installation

1. Start the LLM NIM
```bash
cd nim_llm
pip install -r requirements.txt
python run_llama.py
```

2. Start the application:
```bash
# On Windows:
cd chat-to-3d-core
set ATTN_BACKEND=flash-attn
set SPCONV_ALGO=native
set XFORMERS_FORCE_DISABLE_TRITON=1
python run.py
```

3. Open your browser to the URL shown in the terminal (typically http://localhost:7860)

4. To terminate the application and free VRAM:
```bash
cd chat-to-3d-core
python terminator.py
```
This will:
- Gracefully terminate the Gradio application
- Free up GPU memory
- Allow you to proceed with other operations (e.g., Blender)

### Using the Interface

Once the application is running (either through Docker or manual installation), you can:

- Describe your scene
- Get AI suggestions for objects
- Generate 3D assets
- Auto-import into Blender

### Blender Integration
- Open Blender
- Click on Scripting and New.
- Paste the script from blender/auto_import.py.
- Click run.
- Note - If you have new assets, rerun it.

## Acknowledgments

- [TRELLIS](https://github.com/microsoft/TRELLIS) for the 3D generation capabilities
- [Griptape](https://github.com/griptape-ai/griptape) for the agent framework
- [Gradio](https://github.com/gradio-app/gradio) for the web interface
- [TRELLIS Windows Installation Guide](https://github.com/ericcraft-mh/TRELLIS-install-windows) for Windows setup instructions

