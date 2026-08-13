"""Every bidi container must accept the frame type SageMaker actually sends.

``invoke_endpoint_with_bidirectional_stream`` forwards each ``RequestPayloadPart``
as a *binary* WebSocket frame. Starlette's ``receive_text()`` reads
``message["text"]`` unconditionally, which once raised ``KeyError: 'text'`` on
every production bidi request. The generic handler then forwarded ``str(e)``,
making the reply an error frame whose message was the literal string ``'text'``
— which reads as a complaint about the payload rather than a transport
mismatch, and is why it went unnoticed while the endpoint reported InService.

``test_kokoro_serve.py`` skips outside the container image. This one runs
everywhere: it extracts ``_receive_message`` from the source and exercises it
against fake ASGI messages, so it is still covered when torch and the model
assets that only exist inside the image are absent.
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

import pytest

_CONTAINERS = Path(__file__).resolve().parents[1] / "containers"

# Every container exposing a bidirectional WebSocket route.
HANDLERS = {
    "kokoro": "kokoro/serve.py",
}


class FakeWebSocketDisconnect(Exception):
    """Stand-in for starlette.websockets.WebSocketDisconnect."""

    def __init__(self, code: int = 1000) -> None:
        self.code = code


class FakeWebSocket:
    """Returns one scripted ASGI message from ``receive()``."""

    def __init__(self, message: dict) -> None:
        self._message = message

    async def receive(self) -> dict:
        return self._message


def _load_receive_message(relpath: str):
    """Extract and compile ``_receive_message`` from a container source file.

    Executing just this function keeps the test independent of torch, the model
    packages, and the on-disk voice assets that only exist inside the images.
    """
    source = (_CONTAINERS / relpath).read_text()
    match = re.search(
        r"^async def _receive_message.*?(?=^async def |^def |^@)",
        source,
        re.S | re.M,
    )
    assert match, f"{relpath} has no _receive_message; did it revert to receive_text()?"
    namespace: dict = {
        "WebSocket": object,
        "WebSocketDisconnect": FakeWebSocketDisconnect,
    }
    exec(match.group(0), namespace)  # noqa: S102 - compiling repo source under test
    return namespace["_receive_message"]


def _receive(relpath: str, message: dict) -> str:
    return asyncio.run(_load_receive_message(relpath)(FakeWebSocket(message)))


@pytest.mark.parametrize("relpath", HANDLERS.values(), ids=HANDLERS.keys())
class TestReceiveMessage:
    def test_a_binary_frame_decodes(self, relpath: str) -> None:
        # The exact shape SageMaker delivers, and the case that was broken.
        payload = json.dumps({"text": "hello", "voice": "af_heart"})
        message = {"type": "websocket.receive", "bytes": payload.encode("utf-8")}

        assert json.loads(_receive(relpath, message))["text"] == "hello"

    def test_a_text_frame_still_decodes(self, relpath: str) -> None:
        # The browser demo and Starlette's TestClient send text frames, so a fix
        # that swapped receive_text for receive_bytes would just invert the bug.
        payload = json.dumps({"text": "hello"})
        message = {"type": "websocket.receive", "text": payload}

        assert json.loads(_receive(relpath, message))["text"] == "hello"

    def test_multibyte_text_survives_a_binary_frame(self, relpath: str) -> None:
        # Must decode as UTF-8 rather than latin-1: a mojibake'd prompt would
        # synthesize the wrong audio while still looking like a success.
        text = "café ünïcode 日本語"
        payload = json.dumps({"text": text}, ensure_ascii=False)
        message = {"type": "websocket.receive", "bytes": payload.encode("utf-8")}

        assert json.loads(_receive(relpath, message))["text"] == text

    def test_a_disconnect_raises_rather_than_decoding_nothing(self, relpath: str) -> None:
        # The handlers already treat WebSocketDisconnect as a normal end of
        # session. Without this translation a disconnect would decode to "" and
        # be logged as an error, which is what the old code did on close.
        with pytest.raises(FakeWebSocketDisconnect):
            _receive(relpath, {"type": "websocket.disconnect", "code": 1001})

    def test_an_empty_binary_frame_decodes_to_empty_string(self, relpath: str) -> None:
        # Not an exception: the handler's own "text required" branch is what
        # should answer this, so it must get a value back.
        assert _receive(relpath, {"type": "websocket.receive", "bytes": b""}) == ""


@pytest.mark.parametrize("relpath", HANDLERS.values(), ids=HANDLERS.keys())
def test_no_container_reads_frames_as_text_only(relpath: str) -> None:
    """Guards the fix directly: ``receive_text()`` must not come back.

    Asserted on source rather than behavior because a reintroduced
    ``receive_text()`` elsewhere in the handler would break production bidi
    again without failing any of the tests above.
    """
    source = (_CONTAINERS / relpath).read_text()

    # Matches an actual await, not the prose in _receive_message's docstring
    # explaining why the call was removed.
    calls = re.findall(r"await\s+\w+\.receive_text\(\)", source)

    assert not calls, (
        f"{relpath} calls receive_text(), which raises KeyError: 'text' on the "
        "binary frames SageMaker's bidirectional transport sends"
    )
