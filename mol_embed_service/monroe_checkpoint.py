"""Pinned Monroe checkpoint provenance and complete-bundle resolution."""

import json
import os
import shutil
import tempfile
from pathlib import Path

from filelock import FileLock

from .checkpoint_io import is_lfs_pointer, stream_to_staging, verify_file
from .loading_policy import offline_enabled, reject_offline_download

REVISION = "57238edfffea03808abe761a00cd9a75fa41bb95"
CONFIG_SIZE = 1192
CONFIG_SHA256 = "35417d83c1dac08fb1178cc65fe440b8ab785bd61ad67c0f83c7371b10aa4fe5"
WEIGHTS_SIZE = 300057823
WEIGHTS_SHA256 = "df996e8e98b12a3cda5fe5acbef3e1e9a4ea31946d3736461cd07e53a5bf205a"
BASE_URL = f"https://raw.githubusercontent.com/blazejba/Monroe/{REVISION}/checkpoint"
WEIGHTS_URL = f"https://media.githubusercontent.com/media/blazejba/Monroe/{REVISION}/checkpoint/weights.pt"


def default_cache_dir() -> Path:
    """Return the revision-specific service cache without creating it."""
    root = os.environ.get("XDG_CACHE_HOME")
    if root:
        base = Path(root).expanduser()
    else:
        base = Path.home() / ".cache"
    return base / "mol_embed_service" / "monroe" / REVISION


def _valid_managed_bundle(directory: Path) -> bool:
    config = directory / "config.json"
    weights = directory / "weights.pt"
    if not verify_file(config, size=CONFIG_SIZE, sha256=CONFIG_SHA256):
        return False
    if not verify_file(weights, size=WEIGHTS_SIZE, sha256=WEIGHTS_SHA256):
        return False
    try:
        with config.open(encoding="utf-8") as stream:
            return isinstance(json.load(stream), dict)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False


def _is_html_download_page(path: Path) -> bool:
    """Detect an HTML response saved in place of a local checkpoint."""
    try:
        with path.open("rb") as stream:
            content = stream.read(8192).lstrip().lower()
    except OSError:
        return False
    return content.startswith((b"<!doctype html", b"<html", b"<head", b"<body")) or (
        b"<html" in content[:1024] and b"</html>" in content
    )


def _validate_local(directory: Path) -> Path:
    if not directory.is_dir():
        raise FileNotFoundError(f"Monroe checkpoint directory does not exist: {directory}")
    for name in ("config.json", "weights.pt"):
        path = directory / name
        if not path.is_file() or not os.access(path, os.R_OK) or path.stat().st_size == 0:
            raise FileNotFoundError(f"Monroe checkpoint requires readable, nonempty {path}")
        if is_lfs_pointer(path):
            raise ValueError(f"Monroe checkpoint {path} is a Git LFS pointer; fetch the actual file")
        if name == "weights.pt" and _is_html_download_page(path):
            raise ValueError(
                f"Monroe checkpoint {path} contains an HTML download page, not the actual checkpoint. "
                "Obtain the raw weights.pt artifact, or unset MONROE_CHECKPOINT_DIR to use the "
                "verified managed download."
            )
    try:
        with (directory / "config.json").open(encoding="utf-8") as stream:
            if not isinstance(json.load(stream), dict):
                raise TypeError("configuration root must be a JSON object")
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid Monroe configuration at {directory / 'config.json'}: {exc}") from exc
    return directory


def _remove_backup(path: Path) -> None:
    if path.is_symlink() or (path.exists() and not path.is_dir()):
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def _download_bundle(target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(prefix=f".{target.name}-stage-", dir=target.parent))
    backup = target.with_name(f".{target.name}-old-{os.getpid()}")
    moved_old = False
    try:
        stream_to_staging(
            f"{BASE_URL}/config.json", staging_dir / "config.json", size=CONFIG_SIZE, sha256=CONFIG_SHA256
        )
        stream_to_staging(WEIGHTS_URL, staging_dir / "weights.pt", size=WEIGHTS_SIZE, sha256=WEIGHTS_SHA256)
        try:
            with (staging_dir / "config.json").open(encoding="utf-8") as stream:
                if not isinstance(json.load(stream), dict):
                    raise TypeError("Monroe checkpoint config must be a JSON object")
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Downloaded Monroe configuration is invalid: {exc}") from exc
        if not _valid_managed_bundle(staging_dir):
            raise ValueError("Downloaded Monroe checkpoint bundle failed verification")
        if backup.exists() or backup.is_symlink():
            _remove_backup(backup)
        if target.exists() or target.is_symlink():
            os.replace(target, backup)
            moved_old = True
        try:
            os.replace(staging_dir, target)
        except BaseException:
            if moved_old and not target.exists():
                os.replace(backup, target)
                moved_old = False
            raise
        if moved_old:
            _remove_backup(backup)
            moved_old = False
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        if moved_old and backup.exists() and not target.exists():
            os.replace(backup, target)
        elif (backup.exists() or backup.is_symlink()) and target.exists():
            _remove_backup(backup)


def resolve_checkpoint(checkpoint_dir: str | Path | None = None, *, cache_dir: str | Path | None = None) -> Path:
    """Resolve explicit, environment, or pinned managed Monroe checkpoint assets.

    Caller-selected directories are trusted local inputs: validate their shape, but do
    not impose the official digest or replace/repair them.
    """
    configured = checkpoint_dir if checkpoint_dir is not None else os.environ.get("MONROE_CHECKPOINT_DIR")
    if configured is not None:
        return _validate_local(Path(configured).expanduser())

    target = Path(cache_dir).expanduser() if cache_dir is not None else default_cache_dir()
    if _valid_managed_bundle(target):
        return target
    if offline_enabled():
        reject_offline_download("Monroe", "pinned config.json and weights.pt", target)

    # Lock path is a sibling, not a child of the directory being atomically replaced.
    lock = FileLock(str(target.parent / f".{target.name}.lock"), timeout=120)
    with lock:
        if _valid_managed_bundle(target):
            return target
        if offline_enabled():
            reject_offline_download("Monroe", "pinned config.json and weights.pt", target)
        _download_bundle(target)
        if not _valid_managed_bundle(target):
            raise ValueError(f"Published Monroe checkpoint bundle failed verification: {target}")
    return target
