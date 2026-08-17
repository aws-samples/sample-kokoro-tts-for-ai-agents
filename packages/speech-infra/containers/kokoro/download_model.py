# Copyright (c) 2026 Amazon.com
# This file is licensed under the MIT License.
# See the LICENSE file in the project root for full license information.

"""Pre-download Kokoro-82M model weights at Docker build time."""

from huggingface_hub import snapshot_download

# Pinned to the repo's current main-branch commit (confirmed via the HF Hub
# API) so a container rebuild can't silently pick up a different checkpoint.
# Bump deliberately -- via a PR, not by dropping the pin -- if the model
# needs updating.
KOKORO_82M_REVISION = "f3ff3571791e39611d31c381e3a41a3af07b4987"

snapshot_download(
    "hexgrad/Kokoro-82M", revision=KOKORO_82M_REVISION, local_dir="/app/models/kokoro-82m"
)
print("Model download complete: hexgrad/Kokoro-82M -> /app/models/kokoro-82m")
