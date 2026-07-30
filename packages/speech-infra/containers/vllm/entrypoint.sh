#!/bin/bash
# Custom SageMaker entrypoint for bidirectional streaming TTS.
#
# Architecture:
#   vLLM (port 8000, internal) <-> streaming_proxy.py (port 8080, SageMaker-facing)
#
# We override the DLC's sagemaker_entrypoint.sh because it binds vLLM directly
# to port 8080. We need a WebSocket-capable proxy on 8080 instead, with vLLM
# serving as the internal inference backend.
#
# Env var conventions (same as DLC):
#   SM_VLLM_*         -> translated to vLLM CLI args (--model, --max-model-len, etc.)
#   SM_VLLM_PORT      -> vLLM internal port (default 8000, NOT 8080)
#   MODEL_S3_URI      -> S3 path to pre-cached model weights
#   SNAC_S3_URI       -> S3 path to SNAC audio codec weights
#   SNAC_MODEL_PATH   -> local path to SNAC codec (set after S3 download)
set -eo pipefail
exec > >(tee -a /dev/fd/1) 2>&1

# ─── Startup Stage Markers ────────────────────────────────────────────────────
# Parsed by `tts-bench ttotal` to attribute scaling lag to a stage. The format is
# byte-identical across all four containers so one parser reads them all. Exported
# so streaming_proxy.py can share the origin if it ever grows a lifespan hook.
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

# ─── Early Diagnostics ────────────────────────────────────────────────────────
echo "=== CONTAINER START: $(date -u '+%Y-%m-%dT%H:%M:%SZ') | host=$(hostname) ==="
echo "--- GPU ---"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>&1 || echo "WARN: nvidia-smi failed"
echo "--- Disk ---"
df -h /tmp
echo "--- Memory ---"
free -h
echo "--- Python ---"
python3 --version
echo "--- AWS CLI ---"
aws --version 2>&1 || echo "WARN: aws cli not found"
echo "--- Env ---"
echo "MODEL_S3_URI=${MODEL_S3_URI:-<unset>}"
echo "SNAC_S3_URI=${SNAC_S3_URI:-<unset>}"
echo "SM_VLLM_MODEL=${SM_VLLM_MODEL:-<unset>}"
echo "==========================================================================="

# ─── S3 Model Download ────────────────────────────────────────────────────────
# SageMaker execution role provides credentials via instance metadata.
# Models are pre-synced to S3 by the SpeechModelCache stack (CodeBuild).

if [ -n "${MODEL_S3_URI:-}" ]; then
    mkdir -p /tmp/model
    echo "INFO: Downloading model from ${MODEL_S3_URI}..."
    aws s3 sync "${MODEL_S3_URI}" /tmp/model --no-progress \
        --exclude ".manifest" --exclude ".cache/*" --exclude "optimizer*" \
        --exclude "pytorch_model_fsdp*" --exclude "rng_state*" \
        --exclude "trainer_state*" --exclude "training_args*" --exclude "scheduler*" \
        || { echo "ERROR: S3 model sync failed"; exit 1; }
    export SM_VLLM_MODEL="/tmp/model"
    echo "INFO: Model download complete. Size: $(du -sh /tmp/model | cut -f1)"
fi

# SNAC codec: required for decoding vLLM token output into PCM audio
if [ -n "${SNAC_S3_URI:-}" ]; then
    mkdir -p /tmp/snac
    echo "INFO: Downloading SNAC codec from ${SNAC_S3_URI}..."
    aws s3 sync "${SNAC_S3_URI}" /tmp/snac --no-progress \
        --exclude ".manifest" --exclude ".cache/*" \
        || { echo "ERROR: S3 SNAC sync failed"; exit 1; }
    export SNAC_MODEL_PATH="/tmp/snac"
    echo "INFO: SNAC download complete. Size: $(du -sh /tmp/snac | cut -f1)"
fi

# Distinct from `weights_ready`: the bytes are on local disk, not yet in GPU memory.
# Only the two S3-syncing containers emit this stage; the kokoro images bake their
# weights into the image, so for them it would always be zero.
stage weights_fetched

# ─── Build vLLM CLI Args ──────────────────────────────────────────────────────
# Follows the same SM_VLLM_* -> --arg-name pattern as sagemaker_entrypoint.sh.
# Key difference: we default to port 8000 (internal), not 8080.

VLLM_PORT="${SM_VLLM_PORT:-8000}"
ARGS=(--port "${VLLM_PORT}")

# Model auto-detection: same precedence as DLC entrypoint
if [ -z "${SM_VLLM_MODEL:-}" ]; then
    if [ -d "/opt/ml/model" ] && [ "$(ls -A /opt/ml/model 2>/dev/null)" ]; then
        echo "INFO: Auto-detected model at /opt/ml/model"
        ARGS+=(--model /opt/ml/model)
    elif [ -n "${HF_MODEL_ID:-}" ]; then
        echo "INFO: Using HF_MODEL_ID=${HF_MODEL_ID}"
        ARGS+=(--model "${HF_MODEL_ID}")
    fi
fi

# Generic SM_VLLM_* env var -> CLI arg conversion (e.g. SM_VLLM_MAX_MODEL_LEN -> --max-model-len)
while IFS='=' read -r key value; do
    arg_name=$(echo "${key#SM_VLLM_}" | tr '[:upper:]' '[:lower:]' | tr '_' '-')
    # Skip port — already handled above
    case "$arg_name" in port) continue ;; esac
    lower_value=$(echo "$value" | tr '[:upper:]' '[:lower:]')
    if [ "$lower_value" = "true" ]; then
        ARGS+=("--${arg_name}")
    elif [ "$lower_value" = "false" ]; then
        continue
    else
        ARGS+=("--${arg_name}" "$value")
    fi
done < <(env | grep "^SM_VLLM_")

echo "INFO: Starting vLLM with args: ${ARGS[*]}"
# Argument assembly is done; everything after this is process startup. The kokoro
# containers emit this on lifespan entry, which is the same boundary.
stage framework_init

# ─── Process Supervision ──────────────────────────────────────────────────────
# Three processes, all critical. If any dies the container exits and SageMaker
# replaces it. ADOT is fatal because without metrics we have no performance
# visibility and autoscaling cannot function.

cleanup() { kill "$VLLM_PID" "$PROXY_PID" "$ADOT_PID" 2>/dev/null || true; }
trap cleanup EXIT

# 1. vLLM inference server (internal, not exposed to SageMaker)
python3 -m vllm.entrypoints.openai.api_server "${ARGS[@]}" &
VLLM_PID=$!

# Block until vLLM passes health check (exit early if process dies)
until curl -sf "http://localhost:${VLLM_PORT}/health" >/dev/null 2>&1; do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        wait "$VLLM_PID"; rc=$?
        echo "FATAL: vLLM process died during startup (exit code $rc)"
        exit "$rc"
    fi
    sleep 2
done
echo "INFO: vLLM healthy on port ${VLLM_PORT}"
# vLLM has loaded the weights onto the GPU and is accepting requests. The proxy that
# fronts it is not up yet, so this is not readiness.
stage weights_ready

# 2. Streaming proxy (port 8080 — SageMaker routes all traffic here)
#    Handles: GET /ping, POST /invocations, WS /invocations-bidirectional-stream
python3 /opt/streaming_proxy.py &
PROXY_PID=$!

until curl -sf http://localhost:8080/ping >/dev/null 2>&1; do
    if ! kill -0 "$PROXY_PID" 2>/dev/null; then
        wait "$PROXY_PID"; rc=$?
        echo "FATAL: Streaming proxy died during startup (exit code $rc)"
        exit "$rc"
    fi
    sleep 1
done
echo "INFO: Streaming proxy healthy on port 8080"

# One discarded synthesis. /ping answering 200 only proves the proxy is up: the SNAC
# decoder is lazy-loaded on first use (streaming_proxy.py:_get_snac_decoder), so
# without this the first real caller after a scale-out pays for decoder init plus
# CUDA autotune. Driven over HTTP because there is no lifespan hook to hang it on.
# Non-fatal by design - a container that cannot warm up can still serve, and failing
# startup here would turn a latency problem into an outage.
if curl -sf -m 120 -X POST http://localhost:8080/invocations \
        -H 'Content-Type: application/json' \
        -d "{\"text\": \"${WARMUP_TEXT:-Warming up.}\", \"stream\": false}" \
        -o /dev/null 2>&1; then
    echo "INFO: Warm-up synthesis complete"
else
    echo "WARN: Warm-up synthesis failed; serving anyway"
fi
stage warmup_done

# 3. AWS Distro for OpenTelemetry (metrics export)
/opt/aws/aws-otel-collector/bin/aws-otel-collector \
    --config=/opt/aws/aws-otel-collector/etc/config.yaml &
ADOT_PID=$!

echo "INFO: All processes started"
stage ready

# wait -n: exit when ANY process dies (all three are critical for production)
wait -n "$VLLM_PID" "$PROXY_PID" "$ADOT_PID"
EXIT_CODE=$?
for name_pid in "vLLM:$VLLM_PID" "proxy:$PROXY_PID" "ADOT:$ADOT_PID"; do
    name="${name_pid%%:*}"; pid="${name_pid##*:}"
    if ! kill -0 "$pid" 2>/dev/null; then
        echo "FATAL: $name (PID $pid) exited with code $EXIT_CODE"
    fi
done
exit "$EXIT_CODE"
