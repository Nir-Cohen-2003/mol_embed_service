import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from mol_embed_service.monroe_checkpoint import resolve_checkpoint


def setup_manifest(monkeypatch, config=b'{"model": "fixture"}', weights=b"fixture weights"):
    import mol_embed_service.monroe_checkpoint as module

    monkeypatch.setattr(module, "CONFIG_SIZE", len(config))
    monkeypatch.setattr(module, "CONFIG_SHA256", hashlib.sha256(config).hexdigest())
    monkeypatch.setattr(module, "WEIGHTS_SIZE", len(weights))
    monkeypatch.setattr(module, "WEIGHTS_SHA256", hashlib.sha256(weights).hexdigest())
    return module, config, weights


def _fetcher(payloads):
    def fetch(_url, stage, **_kwargs):
        Path(stage).write_bytes(payloads[Path(stage).name])
        return Path(stage)
    return fetch


def test_constructor_environment_then_managed_cache_precedence(tmp_path, monkeypatch):
    explicit = tmp_path / "explicit"
    env = tmp_path / "env"
    for directory in (explicit, env):
        directory.mkdir()
        (directory / "config.json").write_text("{}")
        (directory / "weights.pt").write_bytes(b"trusted local")
    monkeypatch.setenv("MONROE_CHECKPOINT_DIR", str(env))
    assert resolve_checkpoint(explicit) == explicit
    assert resolve_checkpoint() == env
    with pytest.raises(FileNotFoundError):
        resolve_checkpoint(tmp_path / "missing")


def test_offline_cache_miss_and_corrupt_bundle_do_not_write_or_fetch(tmp_path, monkeypatch):
    module, _, _ = setup_manifest(monkeypatch)
    target = tmp_path / "cache" / module.REVISION
    monkeypatch.setenv("MOL_EMBED_OFFLINE", "1")
    monkeypatch.setattr(module, "stream_to_staging", lambda *args, **kwargs: pytest.fail("network"))
    with pytest.raises(RuntimeError, match="MOL_EMBED_OFFLINE=1"):
        resolve_checkpoint(cache_dir=target)
    assert not target.parent.exists()
    target.mkdir(parents=True)
    (target / "config.json").write_text("{}")
    (target / "weights.pt").write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="MOL_EMBED_OFFLINE=1"):
        resolve_checkpoint(cache_dir=target)
    assert (target / "weights.pt").read_bytes() == b"corrupt"
    assert sorted(p.name for p in target.iterdir()) == ["config.json", "weights.pt"]


def test_verified_pair_downloaded_once_then_reused(tmp_path, monkeypatch):
    module, config, weights = setup_manifest(monkeypatch)
    target = tmp_path / "cache" / module.REVISION
    payloads = {"config.json": config, "weights.pt": weights}
    calls = []

    def fetch(url, stage, **kwargs):
        calls.append(url)
        stage.write_bytes(payloads[Path(stage).name])
        assert kwargs["size"] == len(payloads[Path(stage).name])
        return Path(stage)

    monkeypatch.setattr(module, "stream_to_staging", fetch)
    assert resolve_checkpoint(cache_dir=target) == target
    assert resolve_checkpoint(cache_dir=target) == target
    assert len(calls) == 2
    assert json.loads((target / "config.json").read_text()) == {"model": "fixture"}
    assert not list(target.parent.glob(".*stage-*"))


def test_concurrent_initialization_publishes_one_complete_bundle(tmp_path, monkeypatch):
    module, config, weights = setup_manifest(monkeypatch)
    target = tmp_path / "cache" / module.REVISION
    payloads = {"config.json": config, "weights.pt": weights}
    calls = []
    guard = threading.Lock()

    def fetch(url, stage, **kwargs):
        with guard:
            calls.append(url)
        Path(stage).write_bytes(payloads[Path(stage).name])
        return Path(stage)

    monkeypatch.setattr(module, "stream_to_staging", fetch)
    with ThreadPoolExecutor(max_workers=5) as pool:
        results = list(pool.map(lambda _: resolve_checkpoint(cache_dir=target), range(5)))
    assert results == [target] * 5
    assert len(calls) == 2
    assert (target / "config.json").is_file() and (target / "weights.pt").is_file()


def test_failed_download_leaves_old_cache_untouched(tmp_path, monkeypatch):
    module, _, _ = setup_manifest(monkeypatch)
    target = tmp_path / "cache" / module.REVISION
    target.mkdir(parents=True)
    (target / "sentinel").write_text("old")

    def fail(*args, **kwargs):
        raise OSError("interrupted")

    monkeypatch.setattr(module, "stream_to_staging", fail)
    with pytest.raises(OSError, match="interrupted"):
        resolve_checkpoint(cache_dir=target)
    assert (target / "sentinel").read_text() == "old"
    assert not list(target.parent.glob(".*stage-*"))


def test_offline_mode_is_read_dynamically_after_import(tmp_path, monkeypatch):
    _module, config, weights = setup_manifest(monkeypatch)
    target = tmp_path / "cache"
    target.mkdir()
    (target / "config.json").write_bytes(config)
    (target / "weights.pt").write_bytes(weights)
    monkeypatch.setenv("MOL_EMBED_OFFLINE", "1")
    assert resolve_checkpoint(cache_dir=target) == target


@pytest.mark.parametrize("kind", ["file", "symlink", "dangling-symlink"])
def test_invalid_managed_file_or_symlink_repaired_and_backup_removed(tmp_path, monkeypatch, kind):
    module, config, weights = setup_manifest(monkeypatch)
    payloads = {"config.json": config, "weights.pt": weights}
    monkeypatch.setattr(module, "stream_to_staging", _fetcher(payloads))
    target = tmp_path / module.REVISION
    referent = tmp_path / "referent"
    referent.write_text("do not change")
    if kind == "file":
        target.write_text("invalid")
    else:
        target.symlink_to(tmp_path / "absent" if kind == "dangling-symlink" else referent)
    assert resolve_checkpoint(cache_dir=target) == target
    assert module._valid_managed_bundle(target)
    assert referent.read_text() == "do not change"
    assert not list(tmp_path.glob(f".{target.name}-old-*"))


def test_offline_invalid_managed_file_is_not_repaired(tmp_path, monkeypatch):
    module, _, _ = setup_manifest(monkeypatch)
    target = tmp_path / module.REVISION
    target.write_text("leave intact")
    monkeypatch.setenv("MOL_EMBED_OFFLINE", "1")
    with pytest.raises(RuntimeError):
        resolve_checkpoint(cache_dir=target)
    assert target.is_file() and target.read_text() == "leave intact"
