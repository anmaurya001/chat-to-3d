# Use an official NVIDIA CUDA development image as the base
FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

# Set environment variable for GPU architecture compatibility during compilation
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.7;8.9"

# Define build argument for forcing rebuilds
ARG FORCE_REBUILD=0

# Install core system packages needed for building and running (as root)
RUN apt update && apt-get install -yq --no-install-recommends \
    git-lfs \
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

RUN mkdir -p /app/repo && \
    git lfs install && \
    git clone --recursive https://github.com/anmaurya001/chat-to-3d.git /app/repo


# --- Install Dependencies (within the cloned repo structure) ---
WORKDIR /app/repo
RUN pip install -r requirements-torch.txt
RUN pip install -r requirements-other.txt
RUN pip install -r requirements.txt # Assuming it exists

RUN pip install flash-attn

# --- User Setup (Recommended for Production) ---
# Create a non-root user to run the application
RUN useradd -ms /bin/bash user

# Create the user's standard home directory and standard cache directory (as root)
RUN mkdir -p /home/user/.cache

# --- Entrypoint Setup ---
# Copy the entrypoint script from the cloned repository (as root)
RUN cp /app/repo/entrypoint.sh /entrypoint.sh && \
    chown user:user /entrypoint.sh && \
    chmod +x /entrypoint.sh

# Change ownership of the cloned code repo and the user's home/cache directories to the non-root user
RUN chown -R user:user /app/repo /home/user

# Switch to the non-root user for subsequent commands and runtime
USER user

# Ensure standard binary directories are in the PATH for the non-root user
ENV PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$PATH"

# Set the final working directory to the application logic directory for runtime
WORKDIR /app/repo/chat-to-3d-core

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