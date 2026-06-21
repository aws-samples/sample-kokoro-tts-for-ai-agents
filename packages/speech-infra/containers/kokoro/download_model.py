"""Pre-download Kokoro-82M model weights at Docker build time."""

from huggingface_hub import snapshot_download

snapshot_download("hexgrad/Kokoro-82M", local_dir="/app/models/kokoro-82m")
print("Model download complete: hexgrad/Kokoro-82M -> /app/models/kokoro-82m")
