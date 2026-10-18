"""Offline regression coverage for Uni-Mol2 metadata and inference components."""

import hashlib
from dataclasses import replace
from io import BytesIO

import numpy as np
import pytest
import torch

from mol_embed_service.embedder import EMBEDDING_SIZES, embed_smiles, get_embedding_size
from mol_embed_service.models import BaseEmbedder, UniMol2Embedder
from mol_embed_service.unimol2_checkpoint import resolve_checkpoint
from mol_embed_service.unimol2_config import CONFIGS, UniMol2Config
from mol_embed_service.unimol2_features import collate, featurize_smiles
from mol_embed_service.unimol2_model import UniMol2Model


def test_all_pinned_variants_have_distinct_artifacts():
    assert [c.output_dim for c in CONFIGS.values()] == [768, 768, 1024, 1536, 1536]
    assert len({c.sha256 for c in CONFIGS.values()}) == 5
    assert all(c.directory in c.url for c in CONFIGS.values())


def test_explicit_artifact_must_match_size_and_digest(tmp_path):
    path = tmp_path / "weights.pt"
    path.write_bytes(b"not the official artifact")
    with pytest.raises(ValueError, match="does not match"):
        resolve_checkpoint("unimol2-84M", path)


def test_missing_explicit_artifact_fails_without_network(tmp_path, monkeypatch):
    monkeypatch.setenv("UNIMOL2_CHECKPOINT_PATH", str(tmp_path / "missing"))
    with pytest.raises(FileNotFoundError):
        resolve_checkpoint("unimol2-84M")


def test_unknown_variant_rejected_before_download():
    with pytest.raises(KeyError):
        resolve_checkpoint("unimol2")


def test_verified_download_is_published_and_cached(tmp_path, monkeypatch):
    import mol_embed_service.unimol2_checkpoint as checkpoint

    payload = b"tiny verified checkpoint"
    config = replace(
        CONFIGS["unimol2-84M"], bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest()
    )
    monkeypatch.setattr(checkpoint, "CONFIGS", {"unimol2-84M": config})
    default_https_context = checkpoint.ssl._create_default_https_context
    calls = []

    def open_url(url, timeout, context):
        calls.append((url, timeout, context))
        return BytesIO(payload)

    monkeypatch.setattr(checkpoint, "urlopen", open_url)
    first = resolve_checkpoint("unimol2-84M", cache_dir=tmp_path)
    second = resolve_checkpoint("unimol2-84M", cache_dir=tmp_path)
    assert first == second and first.read_bytes() == payload
    assert len(calls) == 1 and calls[0][0] == config.url and calls[0][1] > 0
    assert calls[0][2].check_hostname is False
    assert calls[0][2].verify_mode == checkpoint.ssl.CERT_NONE
    assert checkpoint.ssl._create_default_https_context is default_https_context
    assert not list(first.parent.glob(".checkpoint-*"))


@pytest.mark.parametrize(
    "name,width",
    [
        ("unimol2-84M", 768),
        ("unimol2-164M", 768),
        ("unimol2-310M", 1024),
        ("unimol2-570M", 1536),
        ("unimol2-1.1B", 1536),
    ],
)
def test_all_public_names_are_load_free(name, width):
    assert get_embedding_size(name) == width == EMBEDDING_SIZES[name]


@pytest.mark.parametrize(
    "name,layers,width,heads",
    [
        ("unimol2-84M", 12, 768, 48),
        ("unimol2-164M", 24, 768, 48),
        ("unimol2-310M", 32, 1024, 64),
        ("unimol2-570M", 32, 1536, 96),
        ("unimol2-1.1B", 64, 1536, 96),
    ],
)
def test_meta_models_cover_all_architecture_configs(name, layers, width, heads):
    with torch.device("meta"):
        model = UniMol2Model(name)
    assert len(model.encoder.layers) == layers
    assert model.config.width == width and model.config.heads == heads
    assert model.encoder.layers[0].self_attn.linear_q.weight.shape == (width, width)


def test_retained_graph_features_and_rounding():
    item = featurize_smiles("C[C@H](O)Cl")
    assert item["atom"].shape == (4, 8)
    assert item["edge"].shape == (4, 4, 3)
    assert item["pair_type"].shape == (4, 4, 2)
    batch = collate([item])
    assert batch["src_token"].shape == (1, 7)  # 5 incl CLS rounded up to 8
    assert batch["attn_bias"][0, 5, 0] == 0
    assert torch.isneginf(batch["attn_bias"][0, 0, 5])


def test_active_graph_feature_golden_values_and_disconnected_sentinel():
    ethanol = featurize_smiles("CCO")
    assert ethanol["atom"].tolist() == [
        [3, 23, 40, 54, 67, 85, 99, 115],
        [3, 23, 40, 53, 67, 85, 99, 115],
        [3, 21, 40, 52, 67, 85, 99, 115],
    ]
    assert ethanol["degree"].tolist() == [3, 4, 3]
    assert ethanol["path"].tolist() == [[1, 2, 3], [2, 1, 2], [3, 2, 1]]
    # Original active graph feature semantics: absent and diagonal edges remain 0
    # until the global +2; bonded features receive convert_to_single_emb +1 first.
    assert ethanol["edge"][0, 0].tolist() == [2, 2, 2]
    assert ethanol["edge"][0, 1].tolist() == [4, 20, 36]
    assert ethanol["edge"][1, 0].tolist() == [4, 20, 36]
    assert ethanol["edge"][0, 2].tolist() == [2, 2, 2]
    disconnected = featurize_smiles("C.C")
    assert disconnected["path"].tolist() == [[1, 510], [510, 1]]
    assert disconnected["edge"].tolist() == [[[2, 2, 2], [2, 2, 2]], [[2, 2, 2], [2, 2, 2]]]
    assert disconnected["pair_type"][0, 0].tolist() == [4, 132]


def test_invalid_is_none_and_large_molecule_rejected():
    assert featurize_smiles("not a smiles") is None
    with pytest.raises(ValueError, match="maximum is 256"):
        featurize_smiles("C" * 257)


def test_adapter_buckets_and_restores_rows_with_invalid_zero_fill():
    class FakeModel(torch.nn.Module):
        def forward(self, data):
            values = data["src_token"].sum(-1, keepdim=True).float()
            return values.expand(-1, 768)

    embedder = UniMol2Embedder.__new__(UniMol2Embedder)
    BaseEmbedder.__init__(embedder, device="cpu")
    embedder.config = CONFIGS["unimol2-84M"]
    embedder.model = FakeModel().eval()
    values = embedder.embed(["CCO", "bad", "C", "c1ccccc1"], batch_size=2)
    assert values.shape == (4, 768) and values.dtype == np.float32
    assert np.all(values[1] == 0) and np.all(values[[0, 2, 3], 0] > 0)
    with pytest.raises(ValueError, match="positive"):
        embedder.embed(["CCO"], batch_size=0)


def test_tiny_native_model_forward_is_finite():
    config = UniMol2Config(
        "tiny", "tiny", 1, 16, 4, 1, "0" * 64, pair_width=8, pair_hidden=4, kernels=8
    )
    model = UniMol2Model(config=config).eval()
    with torch.no_grad():
        model.encoder.layers[0].pair_tri_mul.linear_z.weight.normal_(0, 0.02)
    data = collate([featurize_smiles("CCO")])
    with torch.inference_mode():
        out = model(data)
    assert out.shape == (1, 16) and torch.isfinite(out).all()
    assert out.abs().sum() > 0


def test_tiny_model_batch_and_reordering_invariance():
    config = UniMol2Config(
        "tiny", "tiny", 1, 16, 4, 1, "0" * 64, pair_width=8, pair_hidden=4, kernels=8
    )
    model = UniMol2Model(config=config).eval()
    with torch.no_grad():
        model.encoder.layers[0].pair_tri_mul.linear_z.weight.normal_(0, 0.02)
    items = [featurize_smiles(s) for s in ("CCO", "c1ccccc1", "C.C")]
    with torch.inference_mode():
        together = model(collate(items))
        split = torch.cat([model(collate([item])) for item in items])
        reordered = model(collate([items[i] for i in (2, 0, 1)]))
    assert torch.allclose(together, split, atol=1e-6, rtol=1e-6)
    assert torch.allclose(together, reordered[[1, 2, 0]], atol=1e-6, rtol=1e-6)


def test_download_failure_and_truncation_cleanup(tmp_path, monkeypatch):
    import mol_embed_service.unimol2_checkpoint as checkpoint

    payload = b"verified payload"
    config = replace(
        CONFIGS["unimol2-84M"], bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest()
    )
    monkeypatch.setattr(checkpoint, "CONFIGS", {config.name: config})

    def broken_url(url, timeout, context):
        raise OSError("offline")

    monkeypatch.setattr(checkpoint, "urlopen", broken_url)
    with pytest.raises(OSError, match="offline"):
        resolve_checkpoint(config.name, cache_dir=tmp_path)
    target_dir = tmp_path / "mol_embed_service" / "unimol2" / config.directory
    assert not list(target_dir.rglob(".checkpoint-*"))

    monkeypatch.setattr(
        checkpoint, "urlopen", lambda url, timeout, context: BytesIO(payload[:-1])
    )
    with pytest.raises(ValueError, match="failed verification"):
        resolve_checkpoint(config.name, cache_dir=tmp_path)
    assert not list(target_dir.rglob(".checkpoint-*"))
    assert not list(target_dir.rglob("checkpoint.pt"))


def test_corrupt_cached_download_is_replaced(tmp_path, monkeypatch):
    import mol_embed_service.unimol2_checkpoint as checkpoint

    payload = b"replacement"
    config = replace(
        CONFIGS["unimol2-84M"], bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest()
    )
    monkeypatch.setattr(checkpoint, "CONFIGS", {config.name: config})
    target = (
        tmp_path / "mol_embed_service" / "unimol2" / config.directory
        / checkpoint.HF_REVISION / "checkpoint.pt"
    )
    target.parent.mkdir(parents=True)
    target.write_bytes(b"corrupt")
    monkeypatch.setattr(
        checkpoint, "urlopen", lambda url, timeout, context: BytesIO(payload)
    )
    assert resolve_checkpoint(config.name, cache_dir=tmp_path) == target
    assert target.read_bytes() == payload
    assert not list(target.parent.glob(".checkpoint-*"))


def test_public_dispatch_and_save_for_all_unimol2_variants(tmp_path, monkeypatch):
    import mol_embed_service.embedder as public_api

    dispatched = []

    class FakeEmbedder:
        def __init__(self, variant, device):
            self.variant = variant
            dispatched.append((variant, device))

        def embed(self, smiles, batch_size):
            return np.ones(
                (len(smiles), CONFIGS[self.variant].output_dim), dtype=np.float32
            )

    for name, width in EMBEDDING_SIZES.items():
        if not name.startswith("unimol2-"):
            continue
        monkeypatch.setattr(public_api, "UniMol2Embedder", FakeEmbedder)
        path = tmp_path / f"{name}.npy"
        embed_smiles(["CCO", "C"], name, str(path), batch_size=1, device="cpu")
        saved = np.load(path)
        assert saved.shape == (2, width) and saved.dtype == np.float32
        assert np.all(saved == 1)
        assert dispatched[-1] == (name, "cpu")


def test_parity_harness_detects_output_and_intermediate_failures():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "validate_unimol2", Path(__file__).parents[1] / "scripts" / "validate_unimol2.py"
    )
    harness = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(harness)
    report_has_failures = harness.report_has_failures

    row = {
        "nonfinite": 0,
        "allclose_atol_1e-5_rtol_1e-4": True,
        "stages": {"atom": 1e-7, "parts": [0.0, 2e-7]},
    }
    assert not report_has_failures({"rows": [row]})
    bad_output = {**row, "allclose_atol_1e-5_rtol_1e-4": False}
    assert report_has_failures({"rows": [bad_output]})
    nonfinite_output = {**row, "nonfinite": 1}
    assert report_has_failures({"rows": [nonfinite_output]})
    bad_stage = {**row, "stages": {"atom": float("nan")}}
    assert report_has_failures({"rows": [bad_stage]})
    bad_intermediate = {**row, "stages": {"atom": 1e-3}}
    assert report_has_failures({"rows": [bad_intermediate]})


def test_restricted_loader_rejects_key_shape_nonfinite_unknown_and_container(
    tmp_path, monkeypatch
):
    import mol_embed_service.unimol2_checkpoint as checkpoint
    import mol_embed_service.unimol2_model as model_module

    config = UniMol2Config(
        "tiny", "tiny", 1, 16, 4, 1, "0" * 64, pair_width=8, pair_hidden=4, kernels=8
    )
    monkeypatch.setattr(checkpoint, "CONFIGS", {"tiny": config})
    monkeypatch.setattr(model_module, "CONFIGS", {"tiny": config})
    model = UniMol2Model(config=config)
    state = model.state_dict()
    path = tmp_path / "tiny.pt"
    monkeypatch.setattr(checkpoint, "resolve_checkpoint", lambda *args: path)

    def check(bad_state, message):
        torch.save({"model": bad_state}, path)
        with pytest.raises(ValueError, match=message):
            checkpoint.load_checkpoint("tiny", path)

    wrong_key = dict(state)
    wrong_key["surprise.weight"] = torch.ones(1)
    check(wrong_key, "key coverage")
    wrong_shape = dict(state)
    key = next(iter(state))
    wrong_shape[key] = torch.zeros((1,))
    check(wrong_shape, "has shape")
    nonfinite = dict(state)
    key = next(k for k, value in state.items() if value.is_floating_point())
    nonfinite[key] = torch.full_like(state[key], float("nan"))
    check(nonfinite, "non-finite")
    check({**state, "unknown_head.weight": torch.ones(1)}, "key coverage")
    torch.save({"model": state, "map_location": "cpu"}, path)
    with pytest.raises(ValueError, match="container"):
        checkpoint.load_checkpoint("tiny", path)
    torch.save({"model": state, "kwargs": {"weights_only": False}}, path)
    with pytest.raises(ValueError, match="container"):
        checkpoint.load_checkpoint("tiny", path)
