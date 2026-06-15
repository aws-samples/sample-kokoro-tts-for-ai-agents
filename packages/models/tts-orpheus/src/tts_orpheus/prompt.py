"""Orpheus prompt formatting for vLLM-compatible inference.

Constructs prompts with voice tokens and parses SNAC token IDs from
vLLM output.
"""

from __future__ import annotations

AVAILABLE_VOICES = ("tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe")
DEFAULT_VOICE = "tara"
NUM_CODEBOOKS = 7
CODEBOOK_SIZE = 4096

SOT_TOKEN_ID = 128259
EOT_TOKEN_ID = 128009
SOUND_START_IDS = [128260, 128261, 128257]
STOP_TOKEN_ID = 128258

AUDIO_TOKEN_OFFSET = 128266


def build_prompt(text: str, voice: str = DEFAULT_VOICE) -> str:
    """Build the Orpheus-specific prompt with custom token markers.

    vLLM auto-prepends <|begin_of_text|> (128000) so we omit it here.
    """
    return (
        "<custom_token_3>"
        f"{voice}: {text}"
        "<|eot_id|>"
        "<custom_token_4><custom_token_5><custom_token_1>"
    )


def token_ids_to_snac_codes(token_ids: list[int]) -> list[int]:
    """Convert raw vLLM token IDs to SNAC codes with codebook redistribution.

    Accepts integer token IDs (>= 128266 for audio). Subtracts the base offset
    and per-codebook offset to produce values in [0, 4096) for the SNAC decoder.
    """
    codes: list[int] = []
    for tid in token_ids:
        if tid < AUDIO_TOKEN_OFFSET:
            continue
        raw = tid - AUDIO_TOKEN_OFFSET
        code = raw - ((len(codes) % NUM_CODEBOOKS) * CODEBOOK_SIZE)
        if code < 0 or code >= CODEBOOK_SIZE:
            continue
        codes.append(code)
    return codes
