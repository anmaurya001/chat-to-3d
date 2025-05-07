#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

echo "Starting chat-to-3d container entrypoint (running as user)..."


export HF_HOME="/home/user/.cache/huggingface" # Point to the standard HF subdir within the mounted volume
export TORCH_HOME="/home/user/.cache/torch"    # Point to the standard Torch subdir within the mounted volume

# This handles the case where the volume is newly created or empty on the host
mkdir -p "$HF_HOME" "$TORCH_HOME" || { echo "FATAL: Failed to create cache subdirectories within the mounted volume."; exit 1; }
# Use the absolute path to the python executable
/usr/bin/python -u -c "from trellis.pipelines import TrellisTextTo3DPipeline; TrellisTextTo3DPipeline.from_pretrained('JeffreyXiang/TRELLIS-text-large')"
echo "Model download/check complete."

export AGENT_BASE_URL="http://host.containers.internal:8000/v1"
echo "LLM Server URL set to: $AGENT_BASE_URL"

cd /app/repo/chat-to-3d-core
echo "Executing main application command: $@"
# Execute the original CMD (python run.py)
# Use exec to replace the entrypoint process with the CMD process
exec "$@"