"""CLI entrypoint: python -m stt_whisper <audio_path>

Outputs JSON TranscriptionResult to stdout. Logging goes to stderr.
"""

import sys

from loguru import logger

logger.remove()
logger.add(sys.stderr)


def main() -> None:
    if len(sys.argv) < 2:
        logger.error("Usage: python -m stt_whisper <audio_path>")
        sys.exit(1)

    audio_path = sys.argv[1]

    from stt_whisper.extractor import LocalWhisperExtractor

    extractor = LocalWhisperExtractor()
    result = extractor.transcribe(audio_path)
    sys.stdout.write(result.model_dump_json(exclude_none=True))


if __name__ == "__main__":
    main()
