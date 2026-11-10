"""Small, pickle-free storage helpers used by datasets and checkpoints."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import numpy as np


def canonical_json(data: Mapping[str, Any]) -> str:
    """Serialize metadata deterministically."""
    return json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False)


def parse_json_scalar(value: np.ndarray | str) -> dict[str, Any]:
    """Decode a JSON string stored as a scalar NumPy array."""
    if isinstance(value, np.ndarray):
        value = str(value.item())
    return json.loads(value)


def atomic_savez(path: str | Path, **arrays: Any) -> None:
    """Atomically write a compressed ``.npz`` archive.

    A completed checkpoint is either the old archive or the new archive; an
    interruption cannot leave a half-written file at ``path``.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    with NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".npz",
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)

    try:
        np.savez_compressed(temporary_path, **arrays)
        os.replace(temporary_path, destination)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
