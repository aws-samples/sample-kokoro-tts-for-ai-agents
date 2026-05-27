"""CLI entrypoint: python -m tts_kokoro --request '<json>'

Outputs JSON SynthesisResult to stdout. Logging goes to stderr.
"""

import argparse
import sys

from loguru import logger

logger.remove()
logger.add(sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True, help="JSON SynthesisRequest")
    args = parser.parse_args()

    from tts_inference.types import SynthesisRequest
    from tts_kokoro.synthesizer import LocalKokoroSynthesizer

    request = SynthesisRequest.model_validate_json(args.request)
    synthesizer = LocalKokoroSynthesizer()
    result = synthesizer.synthesize(request)
    sys.stdout.write(result.model_dump_json(exclude_none=True))


if __name__ == "__main__":
    main()
