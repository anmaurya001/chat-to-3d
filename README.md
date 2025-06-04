# chat-to-3d

An application that combines natural language processing with 3D asset generation using TRELLIS.

## Features

- Interactive chat interface for scene planning
- AI-assisted object suggestion and layout
- Automatic 3D asset generation from text prompts
- Blender import functionality for generated assets
- VRAM management with model termination

## Installation 

#### Prerequisites

### Windows Installation 

Before installing this project, you need to set up the development environment on Windows:

1. Install Visual Studio Build Tools 2022:
   - Download from: [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
   ```bash
   vs_buildtools.exe --norestart --passive --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended
   ```

2. Install Python 3.11.9 if not already installed
   - Download from: [Python 3.11.9](https://www.python.org/downloads/release/python-3119/)
   - Make sure to check "Add Python to PATH" during installation

3. Download Git LFS:
   - Download from: [Git LFS](https://git-lfs.com/)


#### Installation Steps

1. Clone this repository:
```bash
git lfs install
git clone --recursive https://github.com/anmaurya001/chat-to-3d.git

# If you forgot --recursive during clone:
git submodule update --init --recursive
```

2. Run the installer script:
```bash
cd chat-to-3d
.\install.bat
```

The installation process will:
- Create a Python virtual environment
- Install all required dependencies
- Set up necessary configurations

After successful installation, you'll see:
```
Installation completed successfully
Press any key to continue . . .
```

## Usage

### Starting the Application

1. Start the LLM Agent NIM:
```bash
# Activate virtual environment
# On Windows:
.venv\Scripts\activate
cd nim_llm
python run_llama.py
```

2. In a new terminal window, start the main application:
```bash
# On Windows:
cd chat-to-3d
.venv\Scripts\activate
cd chat-to-3d-core
set ATTN_BACKEND=flash-attn
set SPCONV_ALGO=native
set XFORMERS_FORCE_DISABLE_TRITON=1
python run.py
```

3. Open your browser to the URL shown in the terminal (typically http://localhost:7860)

4. Optional, to log memory usage
```bash
.venv\Scripts\activate
cd mem_logging
python monitor_resources.py -i <frequency in secs>
```

### Managing the Application

To terminate the application and free VRAM:
```bash
cd chat-to-3d-core
python terminator.py
```
This will:
- Gracefully terminate the Gradio application
- Free up GPU memory
- Allow you to proceed with other operations (e.g., Blender)

### Using the Interface

Once the application is running, you can:

1. **Scene Planning**:
   - Describe your desired scene in natural language
   - Get AI suggestions for objects and layout
   - Refine your scene description

2. **Asset Generation**:
   - Generate 3D assets from text prompts
   - Preview generated assets
   - Make adjustments as needed

3. **Blender Integration**:
   - Import generated assets directly into Blender
   - Continue working with the assets in your 3D workflow
   - Can be used with  [3D Guided Gen AI BP](https://github.com/NVIDIA-AI-Blueprints/3d-guided-genai-rtx)

## Troubleshooting

Common issues and solutions:

1. **Installation Issues**:
   - Ensure all prerequisites are installed correctly
   - Check if Python is in your system PATH
   - Verify Visual Studio Build Tools installation

2. **Runtime Issues**:
   - Make sure both NIM and main application are running
   - Check GPU memory usage
   - Verify all environment variables are set correctly

## Acknowledgments

- [TRELLIS](https://github.com/microsoft/TRELLIS) for the 3D generation capabilities
- [Griptape](https://github.com/griptape-ai/griptape) for the agent framework
- [Gradio](https://github.com/gradio-app/gradio) for the web interface
- [TRELLIS Windows Installation Guide](https://github.com/ericcraft-mh/TRELLIS-install-windows) for Windows setup instructions 