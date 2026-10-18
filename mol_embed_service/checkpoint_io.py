"""Small verified streaming and staging primitives for model checkpoints."""

import hashlib
import os
import ssl
import tempfile
from pathlib import Path
from urllib.request import urlopen

_CHUNK_SIZE = 1024 * 1024
_TIMEOUT_SECONDS = 30


def is_lfs_pointer(path: str | Path) -> bool:
    """Detect Git LFS pointer text without reading a potentially large checkpoint."""
    try:
        with Path(path).open("rb") as stream:
            return stream.read(128).startswith(b"version https://git-lfs.github.com/spec/v1")
    except OSError:
        return False


def verify_file(
    path: str | Path,
    *,
    size: int | None = None,
    sha256: str | None = None,
    md5: str | None = None,
) -> bool:
    """Check file readability, optional exact length, and supplied digests."""
    path = Path(path)
    try:
        if not path.is_file() or is_lfs_pointer(path):
            return False
        if size is not None and path.stat().st_size != size:
            return False
        digests = {"sha256": hashlib.sha256(), "md5": hashlib.md5() if md5 else None}
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(_CHUNK_SIZE), b""):
                digests["sha256"].update(chunk)
                if digests["md5"]:
                    digests["md5"].update(chunk)
        return (sha256 is None or digests["sha256"].hexdigest() == sha256) and (
            md5 is None or digests["md5"].hexdigest() == md5
        )
    except OSError:
        return False


def stream_to_staging(
    url: str,
    staging_path: str | Path,
    *,
    size: int | None = None,
    sha256: str | None = None,
    md5: str | None = None,
    timeout: float = _TIMEOUT_SECONDS,
) -> Path:
    """Download HTTPS bytes to a staging file and verify before returning it."""
    if not url.lower().startswith("https://"):
        raise ValueError("Checkpoint downloads require HTTPS")
    destination = Path(staging_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sha_digest = hashlib.sha256()
    md5_digest = hashlib.md5() if md5 is not None else None
    count = 0
    try:
        with (
            urlopen(url, timeout=timeout, context=ssl.create_default_context()) as response,
            destination.open("xb") as output,
        ):
            while True:
                chunk = response.read(_CHUNK_SIZE)
                if not chunk:
                    break
                count += len(chunk)
                if size is not None and count > size:
                    raise ValueError(f"Downloaded artifact exceeds expected size {size}")
                output.write(chunk)
                sha_digest.update(chunk)
                if md5_digest:
                    md5_digest.update(chunk)
            output.flush()
            os.fsync(output.fileno())
        if size is not None and count != size:
            raise ValueError(f"Downloaded artifact has size {count}; expected {size}")
        if sha256 is not None and sha_digest.hexdigest() != sha256:
            raise ValueError("Downloaded artifact SHA256 does not match the pinned digest")
        if md5 is not None and md5_digest.hexdigest() != md5:
            raise ValueError("Downloaded artifact MD5 does not match the expected compatibility digest")
        if is_lfs_pointer(destination):
            raise ValueError("Downloaded artifact is a Git LFS pointer, not the checkpoint")
        return destination
    except BaseException:
        destination.unlink(missing_ok=True)
        raise


def download_verified(url: str, destination: str | Path, **verification) -> Path:
    """Stage, verify, then atomically replace one managed artifact."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{destination.name}-", suffix=".part", dir=destination.parent)
    os.close(fd)
    staging = Path(name)
    staging.unlink()
    try:
        stream_to_staging(url, staging, **verification)
        os.replace(staging, destination)
        return destination
    finally:
        staging.unlink(missing_ok=True)
