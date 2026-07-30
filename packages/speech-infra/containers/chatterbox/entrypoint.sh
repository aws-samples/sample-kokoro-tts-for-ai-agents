#!/bin/bash
set -eo pipefail

# Stage markers parsed by `tts-bench ttotal`, byte-identical across all four
# containers. Exported so streaming_proxy.py measures elapsed_s from the same
# origin - the shell half of startup (S3 model sync) is otherwise invisible to it.
export CONTAINER_START_EPOCH="$(date +%s.%N)"

stage() {
    local now stamp elapsed
    # One clock reading for both fields: two `date` calls can straddle a second
    # boundary and emit a timestamp that disagrees with its own elapsed_s.
    now="$(date +%s.%N)"
    stamp="$(date -u -d "@${now}" '+%Y-%m-%dT%H:%M:%S.%3NZ')"
    elapsed="$(awk -v a="$now" -v b="$CONTAINER_START_EPOCH" 'BEGIN{printf "%.3f", a-b}')"
    echo "=== STAGE $1 t=${stamp} elapsed_s=${elapsed} ==="
}

stage container_start

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

# Bytes on local disk, not yet in GPU memory - streaming_proxy.py emits
# `weights_ready` once they are. Emitted unconditionally so the stage sequence is
# the same shape whether or not MODEL_S3_URI is set; with it unset the model comes
# from the HF cache instead and this lands ~0s after container_start.
stage weights_fetched

exec python /app/streaming_proxy.py
