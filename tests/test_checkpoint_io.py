import hashlib
import ssl
from io import BytesIO

import pytest

from mol_embed_service.checkpoint_io import is_lfs_pointer, stream_to_staging, verify_file


class Response(BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def test_stream_verifies_bounded_tls_bytes_and_digest(tmp_path, monkeypatch):
    import mol_embed_service.checkpoint_io as io

    payload = b"verified fixture"
    seen = []

    def open_url(url, *, timeout, context):
        seen.append((url, timeout, context))
        return Response(payload)

    monkeypatch.setattr(io, "urlopen", open_url)
    default_https_context = ssl._create_default_https_context
    default_context_factory = ssl.create_default_context
    target = tmp_path / "artifact.part"
    result = stream_to_staging(
        "https://example.invalid/file", target, size=len(payload), sha256=hashlib.sha256(payload).hexdigest()
    )
    assert result == target and target.read_bytes() == payload
    assert seen[0][1] > 0
    assert seen[0][2].check_hostname is False
    assert seen[0][2].verify_mode == ssl.CERT_NONE
    assert ssl._create_default_https_context is default_https_context
    assert ssl.create_default_context is default_context_factory
    assert verify_file(target, size=len(payload), sha256=hashlib.sha256(payload).hexdigest())


def test_truncated_oversized_bad_hash_and_non_https_clean_staging(tmp_path, monkeypatch):
    import mol_embed_service.checkpoint_io as io

    monkeypatch.setattr(io, "urlopen", lambda *args, **kwargs: Response(b"short"))
    target = tmp_path / "stage"
    with pytest.raises(ValueError, match="expected"):
        stream_to_staging("https://example.invalid/file", target, size=8)
    assert not target.exists()
    monkeypatch.setattr(io, "urlopen", lambda *args, **kwargs: Response(b"too long"))
    with pytest.raises(ValueError, match="exceeds expected size"):
        stream_to_staging("https://example.invalid/file", target, size=4)
    assert not target.exists()
    monkeypatch.setattr(io, "urlopen", lambda *args, **kwargs: Response(b"short"))
    with pytest.raises(ValueError, match="SHA256"):
        stream_to_staging("https://example.invalid/file", target, sha256="0" * 64)
    assert not target.exists()
    with pytest.raises(ValueError, match="HTTPS"):
        stream_to_staging("http://example.invalid/file", target)


def test_lfs_pointer_rejected(tmp_path):
    path = tmp_path / "weights.pt"
    path.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:abc\n")
    assert is_lfs_pointer(path)
    assert not verify_file(path)
