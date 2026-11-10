"""Load the two read-only analytic operators built for the bundled rasterizer."""

from functools import lru_cache
import hashlib
import importlib
import json
from pathlib import Path


@lru_cache(maxsize=1)
def load_analyzers():
    import diff_gaussian_rasterization as backend

    expected = json.loads((Path(__file__).parents[1] / "extensions/backend_abi.json").read_text())
    actual = hashlib.sha256(Path(backend.__file__).read_bytes()).hexdigest()
    if actual != expected["renderer_python_sha256"]:
        raise RuntimeError(
            "Install the bundled diff-gaussian-rasterization; its saved-buffer ABI is required."
        )
    try:
        return (
            importlib.import_module("analytic_anchor_energy"),
            importlib.import_module("analytic_slot_moments"),
        )
    except ImportError as exc:
        raise RuntimeError(
            "Build the final operators with: python -m pip install --no-build-isolation ./extensions/analytic"
        ) from exc
