"""Puts container/ on sys.path so tests can import its flat sibling modules directly.

The container's own files (``events.py``, ``bidi_bridge.py``, ``modes.py``,
``strands_agent.py``, ``server.py``) import each other with plain top-level
names (``import events``, ``from modes import run_bidi``), matching how they
actually run inside the image (``/app/*.py``, ``python server.py``) -- not as
an installed package. Tests need the same layout on ``sys.path``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_CONTAINER_DIR = Path(__file__).resolve().parent.parent / "container"

if str(_CONTAINER_DIR) not in sys.path:
    sys.path.insert(0, str(_CONTAINER_DIR))
