#!/bin/bash
set -eo pipefail

echo "=== CHATTERBOX CONTAINER START: $(date -u '+%Y-%m-%dT%H:%M:%SZ') ==="
echo "--- GPU ---"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>&1 || echo "WARN: nvidia-smi failed"
echo "--- Env ---"
echo "MODEL_S3_URI=${MODEL_S3_URI:-<unset>}"
echo "MODEL_DIR=${MODEL_DIR:-/app/model}"
echo "GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.7}"
echo "MAX_MODEL_LEN=${MAX_MODEL_LEN:-1000}"
echo "==========================================================================="

if [ -n "${MODEL_S3_URI:-}" ]; then
    mkdir -p /app/model
    echo "INFO: Downloading model from ${MODEL_S3_URI}..."
    aws s3 sync "${MODEL_S3_URI}" /app/model --no-progress \
        --exclude ".manifest" --exclude ".cache/*" \
        --exclude "*mtl*" --exclude "*23lang*" \
        --exclude "Cangjie5_TC.json" --exclude "*.md"
    echo "INFO: Model download complete. Size: $(du -sh /app/model | cut -f1)"
else
    echo "WARN: MODEL_S3_URI not set, will attempt HuggingFace download at inference"
fi

mkdir -p /app/t3-model
if [ -f /app/model/t3_cfg.safetensors ]; then
    ln -sf /app/model/t3_cfg.safetensors /app/t3-model/model.safetensors
    echo "INFO: Symlinked t3-model/model.safetensors"
fi

exec python /app/streaming_proxy.py
