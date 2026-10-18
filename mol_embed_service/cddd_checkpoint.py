"""Resolve the legacy CDDD encoder cache without importing its downloader package."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from filelock import FileLock

from .checkpoint_io import download_verified, verify_file
from .loading_policy import offline_enabled, reject_offline_download

MODEL_URL = "https://zenodo.org/records/14811055/files/encoder.onnx?download=1"
MODEL_MD5 = "35ef1902b85e67abfb7abed9fa06ffc2"


def get_model_path(name: str = "encoder") -> str:
    """Return a verified legacy-cache model path, acquiring it only when allowed."""
    if name != "encoder":
        raise ValueError(f"Unsupported CDDD model: {name}")
    path = Path.home() / ".cddd_onnx" / "models" / "encoder.onnx"
    if verify_file(path, md5=MODEL_MD5):
        return str(path)
    if offline_enabled():
        reject_offline_download("CDDD", "encoder.onnx", path)

    path.parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(path.parent / ".encoder.onnx.lock"))
    with lock:
        if verify_file(path, md5=MODEL_MD5):
            return str(path)
        fd, staging_name = tempfile.mkstemp(prefix=".encoder-", suffix=".part", dir=path.parent)
        os.close(fd)
        staging = Path(staging_name)
        staging.unlink()
        try:
            download_verified(MODEL_URL, staging, md5=MODEL_MD5)
            os.replace(staging, path)
        finally:
            staging.unlink(missing_ok=True)
    return str(path)
