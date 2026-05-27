"""CLI entrypoint: python -m stt_qwen3_asr <audio_path>

Outputs JSON TranscriptionResult to stdout. Logging goes to stderr.
"""

import sys

from loguru import logger

logger.remove()
logger.add(sys.stderr)


def main() -> None:
    if len(sys.argv) < 2:
        logger.error("Usage: python -m stt_qwen3_asr <audio_path>")
        sys.exit(1)

    audio_path = sys.argv[1]

    from stt_qwen3_asr.extractor import LocalQwen3ASRExtractor

    extractor = LocalQwen3ASRExtractor()
    result = extractor.transcribe(audio_path)
    sys.stdout.write(result.model_dump_json(exclude_none=True))


if __name__ == "__main__":
    main()
