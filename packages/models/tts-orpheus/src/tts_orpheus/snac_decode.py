"""SNAC audio decoder for Orpheus TTS.

Converts groups of 7 SNAC token codes into PCM audio bytes.
The SNAC codec (hubertsiuzdak/snac_24khz) uses a hierarchical 3-level
quantization: 1 coarse + 2 mid + 4 fine codes per frame.
"""

from __future__ import annotations

import os

import numpy as np
import torch

SNAC_MODEL_ID = "hubertsiuzdak/snac_24khz"
SAMPLE_RATE = 24000
TOKENS_PER_FRAME = 7


class SnacDecoder:
    """Decodes SNAC token codes to audio waveform."""

    def __init__(self, model_path: str | None = None, device: str | None = None) -> None:
        self._model_path = model_path or os.environ.get("SNAC_MODEL_PATH", SNAC_MODEL_ID)
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model: object | None = None

    @property
    def model(self) -> object:
        if self._model is None:
            from snac import SNAC

            self._model = SNAC.from_pretrained(self._model_path).eval().to(self._device)
        return self._model

    def decode_frames(self, codes: list[int]) -> bytes | None:
        """Decode a buffer of SNAC codes into PCM int16 audio bytes.

        Expects a multiple-of-7 list of codes. Each group of 7 codes maps to:
        - codes[0]: coarse (codebook 0)
        - codes[1], codes[4]: mid (codebook 1)
        - codes[2], codes[3], codes[5], codes[6]: fine (codebook 2)
        """
        if len(codes) < TOKENS_PER_FRAME:
            return None

        num_frames = len(codes) // TOKENS_PER_FRAME
        frame_codes = codes[: num_frames * TOKENS_PER_FRAME]

        codes_0 = torch.zeros(num_frames, device=self._device, dtype=torch.int32)
        codes_1 = torch.zeros(num_frames * 2, device=self._device, dtype=torch.int32)
        codes_2 = torch.zeros(num_frames * 4, device=self._device, dtype=torch.int32)

        for j in range(num_frames):
            i = TOKENS_PER_FRAME * j
            codes_0[j] = frame_codes[i]
            codes_1[j * 2] = frame_codes[i + 1]
            codes_1[j * 2 + 1] = frame_codes[i + 4]
            codes_2[j * 4] = frame_codes[i + 2]
            codes_2[j * 4 + 1] = frame_codes[i + 3]
            codes_2[j * 4 + 2] = frame_codes[i + 5]
            codes_2[j * 4 + 3] = frame_codes[i + 6]

        codebooks = [
            codes_0.unsqueeze(0),
            codes_1.unsqueeze(0),
            codes_2.unsqueeze(0),
        ]

        if (
            torch.any(codebooks[0] < 0)
            or torch.any(codebooks[0] > 4096)
            or torch.any(codebooks[1] < 0)
            or torch.any(codebooks[1] > 4096)
            or torch.any(codebooks[2] < 0)
            or torch.any(codebooks[2] > 4096)
        ):
            return None

        with torch.inference_mode():
            audio_hat = self.model.decode(codebooks)

        audio_np = audio_hat.squeeze().detach().cpu().numpy()
        audio_int16 = (audio_np * 32767).astype(np.int16)
        return audio_int16.tobytes()
