"""Dynamic policy for model-artifact acquisition."""

import os
from pathlib import Path


def offline_enabled() -> bool:
    """Return whether model acquisition is disabled at this moment."""
    return os.environ.get("MOL_EMBED_OFFLINE") == "1"


def reject_offline_download(model: str, artifact: str, expected_path: str | Path | None = None) -> None:
    """Raise an actionable error when an unavailable artifact cannot be acquired."""
    location = f" at {expected_path}" if expected_path is not None else ""
    raise RuntimeError(
        f"{model} artifact {artifact}{location} is unavailable while MOL_EMBED_OFFLINE=1; "
        "provide a valid local asset or populate the model cache, or unset MOL_EMBED_OFFLINE."
    )
