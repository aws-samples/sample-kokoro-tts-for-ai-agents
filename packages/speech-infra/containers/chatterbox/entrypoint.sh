#!/bin/bash
set -eo pipefail

echo "=== CHATTERBOX CONTAINER START: $(date -u '+%Y-%m-%dT%H:%M:%SZ') ==="
echo "--- GPU ---"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>&1 || echo "WARN: nvidia-smi failed"
echo "--- Env ---"
echo "MODEL_S3_URI=${MODEL_S3_URI:-<unset>}"
echo "DEFAULT_VOICE=${DEFAULT_VOICE:-ENG_US_F_KimW}"
echo "==========================================================================="

if [ -n "${MODEL_S3_URI:-}" ]; then
    mkdir -p /app/model
    echo "INFO: Downloading model from ${MODEL_S3_URI}..."
    aws s3 sync "${MODEL_S3_URI}" /app/model --no-progress \
        --exclude ".manifest" --exclude ".cache/*" \
        --exclude "*.md"
    echo "INFO: Model download complete. Size: $(du -sh /app/model | cut -f1)"
fi

exec python /app/streaming_proxy.py
