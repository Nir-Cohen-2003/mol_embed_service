"""Per-call offline controls for Hugging Face model and tokenizer loaders."""

import os
import sys
import types

import pytest

from mol_embed_service import models


class _Loaded:
    def to(self, device):
        return self

    def eval(self):
        return self


@pytest.mark.parametrize("model_name", ["chemberta-v1", "chemberta-v2", "chemberta-v3"])
def test_chemberta_passes_dynamic_offline_to_model_and_tokenizer(monkeypatch, model_name):
    calls = []
    monkeypatch.setenv("MOL_EMBED_OFFLINE", "1")
    monkeypatch.setattr(models.AutoTokenizer, "from_pretrained", lambda *a, **kw: calls.append(("tokenizer", a, kw)) or _Loaded())
    monkeypatch.setattr(models.AutoModel, "from_pretrained", lambda *a, **kw: calls.append(("model", a, kw)) or _Loaded())
    models.ChemBERTaEmbedder(model_name, "cpu")
    assert [(kind, kw["local_files_only"]) for kind, _, kw in calls] == [("tokenizer", True), ("model", True)]
    monkeypatch.setenv("MOL_EMBED_OFFLINE", "0")
    calls.clear()
    models.ChemBERTaEmbedder(model_name, "cpu")
    assert all(kw["local_files_only"] is False for _, _, kw in calls)


def test_molformer_mol_jepa_and_mist_pass_offline_to_every_pretrained_call(monkeypatch):
    calls = []
    monkeypatch.setenv("MOL_EMBED_OFFLINE", "1")
    monkeypatch.setattr(models.AutoTokenizer, "from_pretrained", lambda *a, **kw: calls.append(("tokenizer", kw)) or _Loaded())
    monkeypatch.setattr(models.AutoModel, "from_pretrained", lambda *a, **kw: calls.append(("model", kw)) or _Loaded())
    monkeypatch.setitem(sys.modules, "smirk", types.SimpleNamespace(SmirkTokenizerFast=object))
    models.MolformerEmbedder("cpu")
    models.MolJEPAEmbedder("cpu")
    models.MistEmbedder("mist-28M", "cpu")
    assert [kind for kind, _ in calls] == ["tokenizer", "model", "model", "tokenizer", "model"]
    assert all(kwargs["local_files_only"] is True for _, kwargs in calls)


def test_hf_offline_flag_is_dynamic_and_does_not_mutate_global_hub_environment(monkeypatch):
    monkeypatch.delenv("MOL_EMBED_OFFLINE", raising=False)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    assert models.offline_enabled() is False
    monkeypatch.setenv("MOL_EMBED_OFFLINE", "1")
    assert models.offline_enabled() is True
    assert "HF_HUB_OFFLINE" not in os.environ
    assert "TRANSFORMERS_OFFLINE" not in os.environ


def test_chemeleon_offline_cache_miss_does_not_create_or_download(tmp_path, monkeypatch):
    from mol_embed_service import chemeleon_fingerprint as chemeleon

    monkeypatch.setenv("MOL_EMBED_OFFLINE", "1")
    monkeypatch.setattr(chemeleon, "urlretrieve", lambda *args: (_ for _ in ()).throw(AssertionError("network")))
    with pytest.raises(RuntimeError, match="MOL_EMBED_OFFLINE=1"):
        chemeleon._get_or_download(tmp_path / "cache" / "weights.pt", "https://invalid/weights.pt")
    assert not (tmp_path / "cache").exists()
