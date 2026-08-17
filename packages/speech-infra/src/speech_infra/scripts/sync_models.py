# Copyright (c) 2026 Amazon.com
# This file is licensed under the MIT License.
# See the LICENSE file in the project root for full license information.

"""Model sync script for CodeBuild.

Downloads models from HuggingFace Hub and uploads to S3 for fast container startup.
Only downloads models that aren't already cached (checks for .manifest sentinel).
"""

import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from huggingface_hub import snapshot_download

MODEL_IDS = os.environ["MODEL_IDS"].split(",")
S3_BUCKET = os.environ["S3_BUCKET"]
HF_TOKEN_SECRET = os.environ.get("HF_TOKEN_SECRET")

#: Pin known models to a specific commit, confirmed via the HF Hub API, so a
#: rebuild can't silently pick up a different checkpoint. Bump deliberately
#: -- via a PR, not by dropping the pin -- if a model needs updating. Models
#: not listed here (there are none configured today) sync unpinned, with a
#: warning logged, rather than failing the build.
KNOWN_REVISIONS: dict[str, str] = {
    "hexgrad/Kokoro-82M": "f3ff3571791e39611d31c381e3a41a3af07b4987",
}


def log(msg: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"{timestamp} | {msg}", flush=True)


def format_duration(seconds: float) -> str:
    mins, secs = divmod(int(seconds), 60)
    return f"{mins}m {secs}s"


def format_size(bytes_val: int) -> str:
    size_float = float(bytes_val)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_float < 1024:
            return f"{size_float:.1f}{unit}"
        size_float /= 1024
    return f"{size_float:.1f}PB"


def get_dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def get_hf_token() -> str | None:
    """Retrieve HuggingFace token from AWS Secrets Manager."""
    if not HF_TOKEN_SECRET:
        return None
    result = subprocess.run(
        ["aws", "secretsmanager", "get-secret-value", "--secret-id", HF_TOKEN_SECRET],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        log(
            f"WARN: HF_TOKEN_SECRET '{HF_TOKEN_SECRET}' not found in Secrets Manager "
            f"(exit code {result.returncode}). Proceeding without token (public models only)."
        )
        return None
    secret = json.loads(result.stdout)
    token = secret.get("SecretString", "")
    if not token:
        log(f"WARN: HF_TOKEN_SECRET '{HF_TOKEN_SECRET}' exists but SecretString is empty")
        return None
    log("HuggingFace token retrieved from Secrets Manager")
    return str(token)


def model_is_cached(model_id: str) -> bool:
    """Check if model already exists in S3 by looking for .manifest sentinel."""
    manifest_key = f"models/{model_id}/.manifest"
    result = subprocess.run(
        ["aws", "s3api", "head-object", "--bucket", S3_BUCKET, "--key", manifest_key],
        capture_output=True,
        check=False,
    )
    return result.returncode == 0


def sync_model(model_id: str, token: str | None) -> None:
    """Download model from HuggingFace and upload to S3."""
    local_dir = Path(f"/tmp/models/{model_id}")
    s3_prefix = f"s3://{S3_BUCKET}/models/{model_id}/"
    manifest_key = f"models/{model_id}/.manifest"

    revision = KNOWN_REVISIONS.get(model_id)
    if revision is None:
        log(f"WARN: no pinned revision for {model_id} -- syncing unpinned (main)")

    log(f"Downloading {model_id} from HuggingFace Hub...")
    start = time.time()
    snapshot_download(
        model_id,
        revision=revision,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        token=token,
        ignore_patterns=[
            "optimizer*",
            "pytorch_model_fsdp*",
            "rng_state*",
            "trainer_state*",
            "training_args*",
            "*.ot",
            "global_step*",
        ],
    )
    download_time = time.time() - start
    model_size = get_dir_size(local_dir)
    log(f"Download complete: {format_size(model_size)} in {format_duration(download_time)}")

    log(f"Uploading to {s3_prefix}...")
    start = time.time()
    subprocess.run(
        ["aws", "s3", "sync", str(local_dir), s3_prefix, "--no-progress"],
        check=True,
    )
    upload_time = time.time() - start
    log(f"Upload complete in {format_duration(upload_time)}")

    manifest_data = json.dumps(
        {"model_id": model_id, "synced_at": time.strftime("%Y-%m-%dT%H:%M:%SZ")}
    )
    manifest_file = Path(f"/tmp/manifest_{model_id.replace('/', '_')}.json")
    manifest_file.write_text(manifest_data)
    subprocess.run(
        [
            "aws",
            "s3api",
            "put-object",
            "--bucket",
            S3_BUCKET,
            "--key",
            manifest_key,
            "--content-type",
            "application/json",
            "--body",
            str(manifest_file),
        ],
        check=True,
        capture_output=True,
    )
    log(f"Manifest written for {model_id}")


def main() -> None:
    log("=" * 60)
    log("MODEL SYNC BUILD - STARTED")
    log("=" * 60)
    log(f"Models: {', '.join(MODEL_IDS)}")
    log(f"Bucket: {S3_BUCKET}")

    token = get_hf_token()
    build_start = time.time()
    skipped = 0

    to_sync: list[str] = []
    for model_id in MODEL_IDS:
        model_id = model_id.strip()
        if not model_id:
            continue
        if model_is_cached(model_id):
            log(f"SKIP {model_id}: already cached in S3")
            skipped += 1
        else:
            to_sync.append(model_id)

    synced = 0
    if to_sync:
        log(f"Syncing {len(to_sync)} models in parallel...")
        with ThreadPoolExecutor(max_workers=len(to_sync)) as executor:
            futures = {executor.submit(sync_model, mid, token): mid for mid in to_sync}
            for future in as_completed(futures):
                model_id = futures[future]
                future.result()
                synced += 1
                log(f"DONE {model_id} ({synced}/{len(to_sync)})")

    total_time = time.time() - build_start
    log("")
    log("=" * 60)
    log("MODEL SYNC BUILD - COMPLETED")
    log("=" * 60)
    log(f"Synced: {synced}, Skipped: {skipped}, Total time: {format_duration(total_time)}")


if __name__ == "__main__":
    main()
