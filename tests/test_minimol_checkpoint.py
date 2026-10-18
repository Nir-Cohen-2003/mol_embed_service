import hashlib
import io
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from mol_embed_service import minimol_checkpoint as checkpoint


def _manifest(monkeypatch):
    payloads = {"config.yaml": b"config", "base_shape.yaml": b"shape", "state_dict.pth": b"weights"}
    monkeypatch.setattr(checkpoint, "ASSET_SIZES", {name: len(data) for name, data in payloads.items()})
    monkeypatch.setattr(checkpoint, "ASSET_SHA256", {name: hashlib.sha256(data).hexdigest() for name, data in payloads.items()})
    return payloads


class _Response:
    def __init__(self, payload):
        self.stream = io.BytesIO(payload)

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self, size=-1):
        return self.stream.read(size)


def _fake_http(monkeypatch, payloads, *, corrupt=None):
    from mol_embed_service import checkpoint_io

    calls = []

    def open_url(url, **_kwargs):
        name = url.rsplit("/", 1)[-1]
        calls.append(name)
        data = payloads[name]
        if name == corrupt:
            data += b"corrupt"
        return _Response(data)

    monkeypatch.setattr(checkpoint_io, "urlopen", open_url)
    monkeypatch.setattr(checkpoint, "_package_bundle", lambda: None)
    return calls


def test_revision_pins_match_verified_raw_artifact_metadata():
    assert checkpoint.REVISION == "41642328affbb35e4950b7b7a9020aa95919b503"
    assert checkpoint.ASSET_SIZES == {"config.yaml": 13502, "base_shape.yaml": 19695, "state_dict.pth": 20791455}
    assert all(len(value) == 64 for value in checkpoint.ASSET_SHA256.values())


def test_offline_missing_package_bundle_fails_without_creating_cache(monkeypatch, tmp_path):
    cache = tmp_path / "missing"
    monkeypatch.setattr(checkpoint, "_package_bundle", lambda: None)
    monkeypatch.setenv("MOL_EMBED_OFFLINE", "1")
    monkeypatch.setattr(checkpoint, "_valid_bundle", lambda _: False)
    with pytest.raises(RuntimeError, match="MiniMol artifact"):
        checkpoint.resolve_checkpoint(cache_dir=cache)
    assert not cache.exists()


def test_valid_cache_is_reused_offline(monkeypatch, tmp_path):
    target = tmp_path / "cache"
    target.mkdir()
    payloads = _manifest(monkeypatch)
    for name, data in payloads.items():
        (target / name).write_bytes(data)
    monkeypatch.setattr(checkpoint, "_package_bundle", lambda: None)
    monkeypatch.setenv("MOL_EMBED_OFFLINE", "1")
    assert checkpoint.resolve_checkpoint(cache_dir=target) == target


def test_valid_package_bundle_is_preferred_offline(monkeypatch, tmp_path):
    package = tmp_path / "package"
    package.mkdir()
    payloads = _manifest(monkeypatch)
    for name, data in payloads.items():
        (package / name).write_bytes(data)
    monkeypatch.setattr(checkpoint, "_package_bundle", lambda: package)
    monkeypatch.setenv("MOL_EMBED_OFFLINE", "1")
    assert checkpoint.resolve_checkpoint(cache_dir=tmp_path / "cache") == package
    assert not (tmp_path / "cache").exists()


def test_corrupt_cache_fails_offline_without_mutation(monkeypatch, tmp_path):
    target = tmp_path / "cache"
    target.mkdir()
    (target / "state_dict.pth").write_bytes(b"corrupt")
    monkeypatch.setattr(checkpoint, "_package_bundle", lambda: None)
    monkeypatch.setenv("MOL_EMBED_OFFLINE", "1")
    with pytest.raises(RuntimeError, match="MiniMol artifact"):
        checkpoint.resolve_checkpoint(cache_dir=target)
    assert (target / "state_dict.pth").read_bytes() == b"corrupt"


def test_tiny_manifest_acquired_verified_and_reused(monkeypatch, tmp_path):
    payloads = _manifest(monkeypatch)
    calls = _fake_http(monkeypatch, payloads)
    target = tmp_path / "cache"
    assert checkpoint.resolve_checkpoint(cache_dir=target) == target
    assert checkpoint.resolve_checkpoint(cache_dir=target) == target
    assert sorted(calls) == sorted(payloads)
    assert all((target / name).read_bytes() == payload for name, payload in payloads.items())
    assert checkpoint._valid_bundle(target)
    assert not list(target.parent.glob(".*stage-*"))


def test_bad_http_payload_is_rejected_without_publishing(monkeypatch, tmp_path):
    payloads = _manifest(monkeypatch)
    _fake_http(monkeypatch, payloads, corrupt="base_shape.yaml")
    target = tmp_path / "cache"
    with pytest.raises(ValueError, match="size"):
        checkpoint.resolve_checkpoint(cache_dir=target)
    assert not target.exists()
    assert not list(target.parent.glob(".*stage-*"))


def test_interrupted_acquisition_cleans_staging_and_preserves_target(monkeypatch, tmp_path):
    payloads = _manifest(monkeypatch)
    calls = _fake_http(monkeypatch, payloads)
    target = tmp_path / "cache"
    target.mkdir()
    (target / "sentinel").write_text("old")
    from mol_embed_service import checkpoint_io

    original = checkpoint_io.urlopen

    def interrupted(url, **kwargs):
        if url.endswith("base_shape.yaml"):
            raise OSError("interrupted")
        return original(url, **kwargs)

    monkeypatch.setattr(checkpoint_io, "urlopen", interrupted)
    with pytest.raises(OSError, match="interrupted"):
        checkpoint.resolve_checkpoint(cache_dir=target)
    assert (target / "sentinel").read_text() == "old"
    assert not list(target.parent.glob(".*stage-*"))
    assert calls == ["config.yaml"]


def test_concurrent_acquisition_publishes_one_complete_bundle(monkeypatch, tmp_path):
    payloads = _manifest(monkeypatch)
    calls = _fake_http(monkeypatch, payloads)
    guard = threading.Lock()
    from mol_embed_service import checkpoint_io

    original = checkpoint_io.urlopen

    def guarded(url, **kwargs):
        with guard:
            return original(url, **kwargs)

    monkeypatch.setattr(checkpoint_io, "urlopen", guarded)
    target = tmp_path / "cache"
    with ThreadPoolExecutor(max_workers=5) as pool:
        results = list(pool.map(lambda _: checkpoint.resolve_checkpoint(cache_dir=target), range(5)))
    assert results == [target] * 5
    assert sorted(calls) == sorted(payloads)
    assert checkpoint._valid_bundle(target)


@pytest.mark.parametrize("kind", ["file", "symlink", "dangling-symlink"])
def test_repair_removes_owned_file_or_symlink_backup(monkeypatch, tmp_path, kind):
    payloads = _manifest(monkeypatch)
    _fake_http(monkeypatch, payloads)
    target = tmp_path / "cache"
    referent = tmp_path / "referent"
    referent.write_text("must survive")
    if kind == "file":
        target.write_text("invalid managed cache")
    else:
        target.symlink_to(tmp_path / "absent" if kind == "dangling-symlink" else referent)
    assert checkpoint.resolve_checkpoint(cache_dir=target) == target
    assert checkpoint._valid_bundle(target)
    assert referent.read_text() == "must survive"
    assert not list(tmp_path.glob(".*cache-old-*"))


def test_offline_invalid_cache_file_is_not_repaired(monkeypatch, tmp_path):
    target = tmp_path / "cache"
    target.write_text("leave unchanged")
    monkeypatch.setattr(checkpoint, "_package_bundle", lambda: None)
    monkeypatch.setenv("MOL_EMBED_OFFLINE", "1")
    with pytest.raises(RuntimeError):
        checkpoint.resolve_checkpoint(cache_dir=target)
    assert target.is_file() and target.read_text() == "leave unchanged"
