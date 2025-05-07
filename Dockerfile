# Use an official NVIDIA CUDA development image as the base
FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

# Set environment variable for GPU architecture compatibility during compilation
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.7;8.9"

# Install core system packages needed for building and running (as root)
RUN apt update && apt-get install -yq --no-install-recommends \
    python3 python3-dev git curl ninja-build \
    libgl1 libglib2.0-0 libxrender1 libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Install pip using the recommended get-pip.py method (as root)
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py \
    && python3 get-pip.py \
    && rm get-pip.py

# Install essential Python packages including the pydantic fix and wheel (as root)
RUN pip install \
    pydantic==2.10.6 \
    wheel

# Create symlink from python3 to python so commands expecting 'python' work (as root)
RUN ln -s /usr/bin/python3 /usr/bin/python


# --- Clone Application Repository and Submodules ---
# Create the parent directory and clone the main repo with submodules (as root)
# This gets your chat-to-3d code AND the TRELLIS submodule inside /app/repo
ARG BUILD_TIME=0 # Default value

RUN mkdir -p /app/repo && \
    git clone --recursive https://github.com/anmaurya001/chat-to-3d.git /app/repo


# --- Install Dependencies (within the cloned repo structure) ---
# Set working directory to the TRELLIS submodule for installing its dependencies
# This is required because some pip installs (like extensions/vox2seq) are relative to the submodule root
WORKDIR /app/repo/trellis

# Install core PyTorch and torchvision for CUDA 12.1 (as root, from TRELLIS WORKDIR)
RUN pip install \
    torch==2.4.0 \
    torchvision==0.19 --index-url https://download.pytorch.org/whl/cu121

# Install basic TRELLIS dependencies... (as root, from TRELLIS WORKDIR)
RUN pip install \
    pillow imageio imageio-ffmpeg tqdm easydict \
    opencv-python-headless scipy ninja \
    onnxruntime trimesh xatlas \
    pyvista pymeshfix \
    igraph \
    transformers \
    plyfile \
    rembg

# Install utils3d from git (as root, from TRELLIS WORKDIR)
RUN pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8

# Install XFORMERS (as root, from TRELLIS WORKDIR)
RUN pip install xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu121

# Install FLASHATTN (as root, from TRELLIS WORKDIR)
RUN pip install flash-attn

# Install KAOLIN (as root, from TRELLIS WORKDIR)
RUN pip install kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu121.html

# Install SPCONV (as root, from TRELLIS WORKDIR)
RUN pip install spconv-cu120

# Install Open3D (as root, from TRELLIS WORKDIR)
RUN pip install open3d

# Install TRELLIS components from source (as root)
# These use /tmp for cloning and don't depend on current WORKDIR being /app/repo/trellis
RUN mkdir -p /tmp/extensions && \
    # NVDIFFRAST
    git clone https://github.com/NVlabs/nvdiffrast.git /tmp/extensions/nvdiffrast && \
    pip install /tmp/extensions/nvdiffrast && \
    # DIFFOCTREERAST
    git clone --recurse-submodules https://github.com/JeffreyXiang/diffoctreerast.git /tmp/extensions/diffoctreerast && \
    pip install /tmp/extensions/diffoctreerast && \
    # MIPGAUSSIAN
    git clone https://github.com/autonomousvision/mip-splatting.git /tmp/extensions/mip-splatting && \
    pip install /tmp/extensions/mip-splatting/submodules/diff-gaussian-rasterization/ && \
    rm -rf /tmp/extensions # Clean up temp source clones

# Install vox2seq from the cloned repository (relative to WORKDIR /app/repo/trellis)
RUN pip install extensions/vox2seq

# Install DEMO dependencies (as root, from TRELLIS WORKDIR)
RUN pip install gradio==4.44.1 gradio_litmodel3d==0.0.1

WORKDIR /app/repo
RUN pip install -r requirements.txt # Assuming it exists

# --- User Setup (Recommended for Production) ---
# Create a non-root user to run the application
RUN useradd -ms /bin/bash user

# Create the user's standard home directory and standard cache directory (as root)
RUN mkdir -p /home/user/.cache

# Change ownership of the cloned code repo and the user's home/cache directories to the non-root user
RUN chown -R user:user /app/repo /home/user

# Switch to the non-root user for subsequent commands and runtime
USER user

# Ensure standard binary directories are in the PATH for the non-root user
ENV PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$PATH"

# Set the final working directory to the application logic directory for runtime
WORKDIR /app/repo/chat-to-3d-core

# --- Entrypoint Setup ---
# Copy the entrypoint script into the container and make it executable
# Ensure ownership is correct for the 'user'

COPY --chown=user:user entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# --- Runtime Configuration ---
EXPOSE 7860

# Set environment variables for Gradio
ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV GRADIO_SERVER_PORT="7860"

WORKDIR /app/repo/trellis


# Set the entrypoint script
ENTRYPOINT ["/entrypoint.sh"]


# Set the default command (relative to the final WORKDIR)
CMD ["python", "run.py"]