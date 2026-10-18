import pytest

from mol_embed_service.loading_policy import offline_enabled, reject_offline_download


def test_offline_setting_is_exact_and_dynamic(monkeypatch):
    monkeypatch.delenv("MOL_EMBED_OFFLINE", raising=False)
    assert not offline_enabled()
    monkeypatch.setenv("MOL_EMBED_OFFLINE", "true")
    assert not offline_enabled()
    monkeypatch.setenv("MOL_EMBED_OFFLINE", "1")
    assert offline_enabled()
    monkeypatch.setenv("MOL_EMBED_OFFLINE", "0")
    assert not offline_enabled()


def test_offline_error_names_artifact_and_remediation(tmp_path):
    with pytest.raises(RuntimeError, match=r"Monroe.*weights.pt.*MOL_EMBED_OFFLINE=1"):
        reject_offline_download("Monroe", "weights.pt", tmp_path / "weights.pt")
