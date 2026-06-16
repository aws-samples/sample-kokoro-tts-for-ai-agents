"""Download Kokoro ONNX model and voices at Docker build time."""

import os
import urllib.request
from pathlib import Path

MODEL_DIR = os.environ.get("MODEL_DIR", "/app/models")
BASE_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"

FILES = [
    "kokoro-v1.0.onnx",
    "voices-v1.0.bin",
]


def main() -> None:
    out_dir = Path(MODEL_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    for filename in FILES:
        out_path = out_dir / filename
        if out_path.exists():
            print(f"SKIP {filename}: already exists")
            continue

        url = f"{BASE_URL}/{filename}"
        print(f"Downloading {url} ...")
        urllib.request.urlretrieve(url, str(out_path))
        size_mb = out_path.stat().st_size / (1024 * 1024)
        print(f"  Saved {filename} ({size_mb:.1f} MB)")

    print("Model download complete.")


if __name__ == "__main__":
    main()
