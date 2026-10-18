"""Pinned MiniMol assets and immutable, service-owned checkpoint resolution."""

import os
import shutil
import tempfile
from importlib import resources
from pathlib import Path

from filelock import FileLock

from .checkpoint_io import stream_to_staging, verify_file
from .loading_policy import offline_enabled, reject_offline_download

REVISION = "41642328affbb35e4950b7b7a9020aa95919b503"
ASSET_SIZES = {
    "config.yaml": 13502,
    "base_shape.yaml": 19695,
    "state_dict.pth": 20791455,
}
ASSET_SHA256 = {
    "config.yaml": "334ace28d2056e75a4b7b1da45787874aeed7ff2a05a6e3b329df612f13eba8b",
    "base_shape.yaml": "6a5bb34227c6138be0d3c002e2a957de52174a3e2b83a6ed72e1229bf42912f3",
    "state_dict.pth": "a3061e1cdbbd5e544168fbf8c1e01836c8c19ac030891d495a2ae9f29cfe97a7",
}
BASE_URL = f"https://media.githubusercontent.com/media/graphcore-research/minimol/{REVISION}/minimol/ckpts/minimol_v1"
ASSET_URLS = {
    "config.yaml": f"https://raw.githubusercontent.com/graphcore-research/minimol/{REVISION}/minimol/ckpts/minimol_v1/config.yaml",
    "base_shape.yaml": f"https://raw.githubusercontent.com/graphcore-research/minimol/{REVISION}/minimol/ckpts/minimol_v1/base_shape.yaml",
    "state_dict.pth": f"{BASE_URL}/state_dict.pth",
}


def default_cache_dir() -> Path:
    """Return the revision-specific cache path without creating it."""
    root = os.environ.get("XDG_CACHE_HOME")
    base = Path(root).expanduser() if root else Path.home() / ".cache"
    return base / "mol_embed_service" / "minimol" / REVISION


def _valid_bundle(directory: Path) -> bool:
    return all(
        verify_file(directory / name, size=ASSET_SIZES[name], sha256=ASSET_SHA256[name])
        for name in ASSET_SIZES
    )


def _package_bundle() -> Path | None:
    try:
        location = resources.files("minimol.ckpts.minimol_v1")
        directory = Path(os.fspath(location))
    except (ImportError, ModuleNotFoundError, TypeError, OSError):
        return None
    return directory if directory.is_dir() and _valid_bundle(directory) else None


def _remove_backup(path: Path) -> None:
    if path.is_symlink() or (path.exists() and not path.is_dir()):
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def _download_bundle(target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}-stage-", dir=target.parent))
    backup = target.with_name(f".{target.name}-old-{os.getpid()}")
    moved_old = False
    try:
        for name, url in ASSET_URLS.items():
            stream_to_staging(url, staging / name, size=ASSET_SIZES[name], sha256=ASSET_SHA256[name])
        if not _valid_bundle(staging):
            raise ValueError("Downloaded MiniMol bundle failed pinned integrity verification")
        if backup.exists() or backup.is_symlink():
            _remove_backup(backup)
        if target.exists() or target.is_symlink():
            os.replace(target, backup)
            moved_old = True
        try:
            os.replace(staging, target)
        except BaseException:
            if moved_old and not target.exists():
                os.replace(backup, target)
                moved_old = False
            raise
        if moved_old:
            _remove_backup(backup)
            moved_old = False
    finally:
        if staging.exists():
            shutil.rmtree(staging)
        if moved_old and backup.exists() and not target.exists():
            os.replace(backup, target)
        elif (backup.exists() or backup.is_symlink()) and target.exists():
            _remove_backup(backup)


def resolve_checkpoint(*, cache_dir: str | Path | None = None) -> Path:
    """Prefer a complete installed bundle, then verified cache or online acquisition."""
    packaged = _package_bundle()
    if packaged is not None:
        return packaged
    target = Path(cache_dir).expanduser() if cache_dir is not None else default_cache_dir()
    if _valid_bundle(target):
        return target
    if offline_enabled():
        reject_offline_download("MiniMol", "pinned config.yaml, base_shape.yaml, and state_dict.pth", target)
    lock = FileLock(str(target.parent / f".{target.name}.lock"), timeout=120)
    with lock:
        if _valid_bundle(target):
            return target
        if offline_enabled():
            reject_offline_download("MiniMol", "pinned config.yaml, base_shape.yaml, and state_dict.pth", target)
        _download_bundle(target)
        if not _valid_bundle(target):
            raise ValueError(f"Published MiniMol checkpoint bundle failed verification: {target}")
    return target
