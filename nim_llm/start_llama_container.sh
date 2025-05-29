#!/bin/bash

# Get the NGC API key from the environment variable
#echo "Using NGC API Key: $NGC_API_KEY"

# Ensure NGC_API_KEY is set before running podman
if [ -z "$NGC_API_KEY" ]; then
  echo "NGC API Key is missing!"
  exit 1
fi

# Login to nvcr.io
echo "$NGC_API_KEY" | podman login nvcr.io -u '$oauthtoken' --password-stdin

# Export env vars
export LOCAL_NIM_CACHE=~/.cache/nim

# Setup cache dir
mkdir -p "$LOCAL_NIM_CACHE"
chmod -R a+w "$LOCAL_NIM_CACHE"

# Run container
podman run --name CHAT_TO_3D -it --rm \
    --device nvidia.com/gpu=all \
    --shm-size=8GB \
    -e NGC_API_KEY=$NGC_API_KEY \
    -v "$LOCAL_NIM_CACHE:/opt/nim/.cache" \
    -e NIM_RELAX_MEM_CONSTRAINTS=1 \
    -u $(id -u) \
    -p 8000:8000 \
    nvcr.io/nim/meta/llama-3.1-8b-instruct:1.8.0-RTX
