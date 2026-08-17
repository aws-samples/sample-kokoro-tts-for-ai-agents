# Copyright (c) 2026 Amazon.com
# This file is licensed under the MIT License.
# See the LICENSE file in the project root for full license information.

"""Human-friendly CLI for Kokoro-82M TTS synthesis.

Usage:
    uv run --project packages/models/tts-kokoro python -m tts_kokoro.cli \
        -t "Hello world" -t "Second sample"
"""

import re
import sys
from datetime import datetime
from pathlib import Path

import click
from loguru import logger

from tts_inference.types import AudioEncoding, SynthesisRequest, VoiceConfig
from tts_kokoro.synthesizer import LocalKokoroSynthesizer

MODEL_NAME = "kokoro-82m"


def _slugify(text: str, max_len: int = 40) -> str:
    slug = text[:max_len].lower()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    return slug.strip("-") or "audio"


def _resolve_output_dir(output_dir: str | None) -> Path:
    if output_dir:
        return Path(output_dir)
    project_root = Path(__file__).resolve().parents[5]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return project_root / "data" / "audio" / MODEL_NAME / timestamp


@click.command()
@click.option("-t", "--text", multiple=True, required=True, help="Text to synthesize (repeatable).")
@click.option("-o", "--output-dir", default=None, type=click.Path(), help="Output directory.")
@click.option("--voice-id", default=None, help="Voice preset [default: af_heart].")
@click.option("--speed", default=1.0, type=float, help="Speed multiplier [0.5-3.0].")
@click.option("--language", default="en", help="Language code.")
@click.option(
    "--encoding",
    default="wav",
    type=click.Choice(["wav", "mp3", "flac"], case_sensitive=False),
    help="Audio encoding.",
)
@click.option("--sample-rate", default=24000, type=int, help="Sample rate in Hz.")
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging.")
def main(
    text: tuple[str, ...],
    output_dir: str | None,
    voice_id: str | None,
    speed: float,
    language: str,
    encoding: str,
    sample_rate: int,
    verbose: bool,
) -> None:
    """Synthesize speech from text using Kokoro-82M."""
    logger.remove()
    logger.add(sys.stderr, level="DEBUG" if verbose else "INFO")

    out_path = _resolve_output_dir(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: {}", out_path)

    voice_config = VoiceConfig(
        voice_id=voice_id,
        language=language,
        speed=speed,
    )

    synthesizer = LocalKokoroSynthesizer()
    audio_encoding = AudioEncoding(encoding)

    for idx, sample in enumerate(text, start=1):
        if not sample.strip():
            logger.warning("Skipping empty text at position {}", idx)
            continue

        request = SynthesisRequest(
            text=sample,
            voice=voice_config,
            encoding=audio_encoding,
            sample_rate=sample_rate,
        )

        try:
            result = synthesizer.synthesize(request)
        except Exception as exc:
            logger.error("Failed on sample {}: {}", idx, exc)
            continue

        filename = f"{idx:03d}_{_slugify(sample)}.{encoding}"
        file_path = out_path / filename
        file_path.write_bytes(result.audio_bytes)

        click.echo(
            f"  [{idx:03d}] {file_path.name} "
            f"({result.duration_seconds:.2f}s, RTF={result.realtime_factor:.3f})"
        )

    click.echo(f"\nDone. Files written to: {out_path}")


if __name__ == "__main__":
    main()
