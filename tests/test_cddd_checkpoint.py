from pathlib import Path

import pytest

from mol_embed_service import cddd_checkpoint


def test_offline_missing_cache_fails_before_filesystem_or_network(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("MOL_EMBED_OFFLINE", "1")
    monkeypatch.setattr(cddd_checkpoint, "verify_file", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(cddd_checkpoint, "download_verified", lambda *_a, **_k: pytest.fail("network attempted"))
    with pytest.raises(RuntimeError, match="MOL_EMBED_OFFLINE=1"):
        cddd_checkpoint.get_model_path()
    assert not (tmp_path / ".cddd_onnx").exists()


def test_corrupt_legacy_cache_fails_offline_without_repair(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("MOL_EMBED_OFFLINE", "1")
    path = tmp_path / ".cddd_onnx/models/encoder.onnx"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"corrupt")
    monkeypatch.setattr(cddd_checkpoint, "verify_file", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(cddd_checkpoint, "download_verified", lambda *_a, **_k: pytest.fail("network attempted"))
    with pytest.raises(RuntimeError, match="encoder.onnx"):
        cddd_checkpoint.get_model_path()
    assert path.read_bytes() == b"corrupt"


def test_valid_legacy_cache_is_reused_offline(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("MOL_EMBED_OFFLINE", "1")
    path = tmp_path / ".cddd_onnx/models/encoder.onnx"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"cached")
    monkeypatch.setattr(cddd_checkpoint, "verify_file", lambda actual, **kw: actual == path and kw == {"md5": cddd_checkpoint.MODEL_MD5})
    assert Path(cddd_checkpoint.get_model_path()) == path


def test_online_acquisition_verifies_then_atomically_publishes(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("MOL_EMBED_OFFLINE", raising=False)
    path = tmp_path / ".cddd_onnx/models/encoder.onnx"
    calls = []
    monkeypatch.setattr(cddd_checkpoint, "verify_file", lambda actual, **kw: actual == path and path.exists())

    def download(url, staging, **verification):
        calls.append((url, verification))
        Path(staging).write_bytes(b"verified bytes")
        return Path(staging)

    monkeypatch.setattr(cddd_checkpoint, "download_verified", download)
    assert Path(cddd_checkpoint.get_model_path()) == path
    assert path.read_bytes() == b"verified bytes"
    assert calls == [(cddd_checkpoint.MODEL_URL, {"md5": cddd_checkpoint.MODEL_MD5})]
    assert not list(path.parent.glob("*.part"))
