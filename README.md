﻿# chat-to-3d

An application that combines natural language processing with 3D asset generation using TRELLIS.

## Features

- Interactive chat interface for scene planning
- AI-assisted object suggestion and layout
- Automatic 3D asset generation from text prompts
- Blender import functionality for generated assets
- VRAM management with model termination

## Installation 

### Prerequisites
The NIM Prerequisite Installer requires Microsoft User Account Control (UAC) to be enabled.  UAC is enabled by default for Windows, but if it has been disabled, it must be enabled to ensure successful installation of the NIM Prerequisite Installer.  More information on Microsoft UAC can found [HERE](https://support.microsoft.com/en-us/windows/user-account-control-settings-d5b2046b-dcb8-54eb-f732-059f321afe18)

Download the [MS Visual Studio Build Tools vs_buildTools.exe](https://aka.ms/vs/17/release/vs_BuildTools.exe)
Open a command prompt at the vs_BuildTools.exe file location and send the following command:
```
vs_buildtools.exe --norestart --passive --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended
```
Use winget to install Miniconda:
```
winget install miniconda3
```
Download the NVIDIA CUDA Toolkit 12.9
[NVIDIA CUDA Toolkit 12.9](https://developer.download.nvidia.com/compute/cuda/12.9.0/local_installers/cuda_12.9.0_576.02_windows.exe)
Run the installer and select a custom installation.
![Screenshot 2025-05-22 221843](https://github.com/user-attachments/assets/e2e7fe07-d530-4aca-9668-a8566d1d5864)
From the options select ONLY:  
CUDA  >> Development >> Compiler
CUDA >> Runtime >> Libraries
![Screenshot 2025-05-22 222023](https://github.com/user-attachments/assets/9ccd92cc-55a5-467d-b4f3-f1e821a07689)
Complete the installation

Download the [NIM Prerequisite Installer](https://assets.ngc.nvidia.com/products/api-catalog/rtx/NIMSetup.exe), and run the NIMSetup.exe file, and follow the instructions in the setup dialogs. This will install the necessary system components to work with NVIDIA NIMs on your system.

You will need to reboot your computer to complete the installation.

Git is required and should be installed using winget from a command prompt::
```
winget install --id Git.Git 
```
Install Git LFS (large file system) support
```
winget install --id=GitHub.GitLFS  -e
```

Install Microsoft Visual C++ 2015-2022 Redistributable Package  
[https://aka.ms/vs/17/release/vc\_redist.x64.exe](https://aka.ms/vs/17/release/vc_redist.x64.exe)
or
```
winget install Microsoft.VCRedist.2015+.x64
```
![Untitled-8](https://github.com/user-attachments/assets/29184836-3791-4c22-8a40-3254590faa0e)

This blueprint requires the installation of Blender. The blueprint has been tested with the Blender 4.27 LTS (Long Term Support) build.   
[https://www.blender.org/download/release/Blender4.2/blender-4.2.7-windows-x64.msi](https://www.blender.org/download/release/Blender4.2/blender-4.2.7-windows-x64.msi)

Blender 4.27 can also be installed using winget from a command prompt:
```
winget install --id 9NW1B444LDLW
```
### Initilize Conda
1. From the Windows Start button, open a Anaconda (Miniconda) Prompt
2. ![image](https://github.com/user-attachments/assets/a53c4eb6-fb96-4f72-8cee-3aa12e0a470e)
3. You should see the command prompt with prompt prefixed with the (base) conda environment name
4. Initialize Conda for your system
```
conda init
```
5. ![image](https://github.com/user-attachments/assets/5dbe45a3-c11b-4392-9474-42966066ddfe)
6. Close the command prompt

#### Installation Steps
1. Open a new command prompt
2. Clone this repository:
```bash
git lfs install
git clone --recursive https://github.com/anmaurya001/chat-to-3d.git

# If you forgot --recursive during clone:
git submodule update --init --recursive
```

3. Run the installer script:
```bash
cd chat-to-3d
.\install.bat
```

The installation process will:
- Create a Conda virtual environment
- Install all required dependencies
- Set up necessary configurations

After successful installation, you'll see:
```
Installation completed successfully
Press any key to continue . . .
```

## Usage - Manual Steps

### Starting the Application
#### Start the LLM Agent NIM:
1. Open a new command prompt and cd to the chat-to-3d installation directory
2. Activate conda virtual environment
```
conda activate trellis
```
#### Start the LLM 
3. Change to the nim_llm directory
```
cd nim_llm
python run_llama.py
```
#### Start Trellis service
4. In a new terminal window, start the main application:
```
conda activate trellis
cd chat-to-3d-core

set ATTN_BACKEND=flash-attn
set SPCONV_ALGO=native
set XFORMERS_FORCE_DISABLE_TRITON=1
python run.py
```
4. Open your browser to the URL shown in the terminal (typically http://localhost:7860)

5. Optional, to log memory usage open a command prompt in the chat-to-3d install directory
```
conda activate trellis
cd mem_logging
python monitor_resources.py -i <frequency in secs>
```

### Managing the Application

To terminate the application and free VRAM:
Open a command prompt at the chat-to-3d install directory
```
conda activate trellis
cd chat-to-3d-core
python terminator.py
```
This will:
- Gracefully terminate the Gradio application
- Free up GPU memory
- Allow you to proceed with other operations which may require VRAM (e.g., Blender)

## Usage - Blender Add On
The CHAT-TO-3D Blender add-on can automatically manage the CHAT-TO-3D services without the need to manually start or stop these services outside of Blender.
### Initial Setup Step
1. Open Blender
2. Open Edit >> Preferences >> Add-Ons
3.![image](https://github.com/user-attachments/assets/0dd045e1-225a-425f-9f96-9047f3ad476a)
4. Enable CHAT-To-3D and Asset Importer by checking the boxes next to the add on names.
5. Open the CHAT-TO-3D add on preferences and set the CHAT-TO-3D base folder to the chat-to-3d local repository directory.
6. ![image](https://github.com/user-attachments/assets/1f9b6bec-dd13-4a7b-9a02-ac9c84a56869)
### Normal Usage
7. In the 3D layout view look for the Add On tabs on the right edge of the viewport, press N if they are not visible
8. ![image](https://github.com/user-attachments/assets/2bfc6cb2-aa3a-4422-b1d1-a983ed21407d)
9. Note: It is recommended to open a system console viewer to monitor the services and any information or errors that may be output.
   a. Blender Menu >> Window >> Toggle System Console
10. Click the Start CHAT-TO-3D button to start the LLM agent, and the Trellis 3D services.
11. ![image](https://github.com/user-attachments/assets/b5391eed-0cca-45da-952a-04381804c0b5)
12. Once all services have successfully started, the service will indicate: READY and the OPEN CHAT-TO-3D UI button will become available
13. Click the OPEN CHAT-TO-3D UI button to launch the CHAT-TO-3D interface
    
### Using the Interface
![image](https://github.com/user-attachments/assets/8590e6fd-32cb-4974-bcc9-284ef7d1eaa2)
Once the application is running, you can:
1. **Scene Planning**:
   - Describe your desired scene in natural language
   - ![image](https://github.com/user-attachments/assets/38c408f6-219e-4fdc-93f4-5abaea80b391)

   - Get AI suggestions for objects and layout
   - ![image](https://github.com/user-attachments/assets/40f3c85e-2024-499b-bc1c-926d478f0555)
  
   - Select the desired objects
   - ![image](https://github.com/user-attachments/assets/7e389a1b-f9da-4d1f-bc45-e854e8b6cee3)

   - Refine your scene description
   - ![image](https://github.com/user-attachments/assets/5779fbb4-40ad-437a-8f3f-7f4191df1cdd)
  
   - Generate 3D Prompts for selected objects
   - ![image](https://github.com/user-attachments/assets/cadf5ba7-516f-4fee-9992-ac4e70c0f6d6)

2. **Asset Generation**:
   - Modify the generated prompt if desired
   - Generate object variant previews for a single object or for all objects
   -![image](https://github.com/user-attachments/assets/5dfb5ba2-c718-4594-8a9b-b4584f5599fb)

   - Select a desired variant and click Select this Variant button
   - ![image](https://github.com/user-attachments/assets/8f3a488f-f50d-4586-ab1c-76f6d7e54c49)
  
   - Select a Single Image and click *Generate 3D Model for Selected Variant* button
   - Or click *Generate 3D Models for all Selected Variants* button
   - ![image](https://github.com/user-attachments/assets/d028ec91-b279-4057-ade1-ea4e15cf9454)
  
   - Preview generated assets
   - ![image](https://github.com/user-attachments/assets/593bcb43-a4ef-4f8a-b2ba-94d3cbb7a57e)

   - Save generated asset to a scene folder
   - Create a unique folder output for all objects generated in this session
   - ![image](https://github.com/user-attachments/assets/dddae747-0f54-43bc-9fc8-620e35f44a1f)


4. **Blender Integration**:
   - Import generated assets directly into Blender
   - ![image](https://github.com/user-attachments/assets/e6942a0c-6e03-4cbb-9bd6-63507e7eddac)
   - Use the Asset Importer add-on and select the desired scene folder, and click Import assets
   - ![image](https://github.com/user-attachments/assets/b9119900-1780-42cd-a071-6f7d9397984e)
   - Assets are imported and the asset tag is applied, saving the scene to the %userprofile%\Documents\Blender\assets folder will add the imported objects to the Blender asset browser.
   - ![image](https://github.com/user-attachments/assets/c3d705c1-fc21-4137-a6e1-004c9d9b0b5a)
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
