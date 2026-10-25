"""Focused checks for the private Monroe inference subset."""
import json
import threading

import pytest
import torch

from mol_embed_service._vendor.monroe.eval.embed import _featurize_one, featurize_smiles
from mol_embed_service._vendor.monroe.model import ckpt
from mol_embed_service._vendor.monroe.preprocessing import build_eval_graphs


def test_private_resource_and_graph_smoke(capsys):
    from importlib.resources import files

    csv = files("mol_embed_service._vendor.monroe.assets").joinpath("periodic_table.csv")
    assert csv.is_file()
    graph = build_eval_graphs.build_mol("C[C@H](O)F")
    assert graph is not None
    assert graph["node_codes"].ndim == 2
    assert graph["edge_index"].shape[0] == 2
    assert "[info] Building molecule graph" not in capsys.readouterr().out


def test_featurization_falls_back_after_graph_builder_failure(monkeypatch):
    monkeypatch.setattr(build_eval_graphs, "build_mol", lambda value: (_ for _ in ()).throw(RuntimeError("primary failed")))
    monkeypatch.setattr(build_eval_graphs, "_build_from_smiles_fallback", lambda value: {"smiles": value})
    assert _featurize_one("CCO") == ("CCO", {"smiles": "CCO"})


def test_featurization_serially_from_thread_and_rejects_worker_pool(monkeypatch):
    graph = {"edge_index": torch.tensor([[0, 1], [1, 0]])}
    calls = []
    monkeypatch.setattr(build_eval_graphs.dm, "to_inchi", lambda value: "inchi")
    monkeypatch.setattr(
        build_eval_graphs,
        "build_single_graph",
        lambda **kwargs: calls.append((threading.get_ident(), kwargs)) or graph,
    )
    monkeypatch.setattr(
        build_eval_graphs,
        "_build_from_smiles_fallback",
        lambda value: pytest.fail("fallback should not run"),
    )
    result = []
    thread = threading.Thread(target=lambda: result.extend(featurize_smiles(["CCO"], n_workers=1)))
    thread.start()
    thread.join()
    assert result == [("CCO", graph)]
    assert len(calls) == 1
    assert calls[0][0] != threading.get_ident()
    assert calls[0][1]["inchi"] == "inchi"
    with pytest.raises(ValueError, match="n_workers=1"):
        featurize_smiles(["CCO"], n_workers=2)


def test_safe_loader_uses_cpu_weights_only_and_normalizes_compile_prefixes(tmp_path, monkeypatch):
    calls = {}
    source = {"encoder._orig_mod.weight": torch.ones(2, 2)}

    def load(path, **kwargs):
        calls.update(kwargs)
        return source

    monkeypatch.setattr(ckpt.torch, "load", load)
    assert list(ckpt._load_state_dict(tmp_path / "weights.pt")) == ["encoder.weight"]
    assert calls == {"map_location": "cpu", "weights_only": True}


def test_real_encoder_checkpoint_roundtrip_and_forward(tmp_path):
    from torch_geometric.data import Data

    from mol_embed_service._vendor.monroe.model.constants import (
        EDGE_FEAT_LIST_ONE_HOT,
        NODE_FEAT_LIST_FLOAT,
        NODE_FEAT_LIST_ONE_HOT,
    )
    from mol_embed_service._vendor.monroe.model.grit import GritTransformer

    config = {
        "encoder": {
            "rbf_dim": 1,
            "walk_len": 3,
            "emb_dim": 4,
            "hidden_dim": 4,
            "num_heads": 1,
            "num_layers": 1,
        }
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    torch.manual_seed(17)
    original = GritTransformer(
        node_feature_vocab=NODE_FEAT_LIST_ONE_HOT,
        edge_feature_vocab=EDGE_FEAT_LIST_ONE_HOT,
        node_float_dim=len(NODE_FEAT_LIST_FLOAT),
        **config["encoder"],
    )
    torch.save({f"encoder.{key}": value for key, value in original.state_dict().items()}, tmp_path / "weights.pt")

    loaded = ckpt.load_ckpt(str(tmp_path))
    assert not loaded.training
    for key, value in original.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], value)

    graph = Data(
        x=torch.ones(2, len(NODE_FEAT_LIST_FLOAT)),
        node_codes=torch.zeros(2, len(NODE_FEAT_LIST_ONE_HOT), dtype=torch.long),
        edge_index=torch.tensor([[0, 1], [1, 0]]),
        edge_codes=torch.zeros(2, len(EDGE_FEAT_LIST_ONE_HOT), dtype=torch.long),
        pos_in=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        batch=torch.zeros(2, dtype=torch.long),
    )
    embedding, node_embedding = loaded(graph)
    assert embedding.shape == (1, config["encoder"]["hidden_dim"])
    assert torch.isfinite(embedding).all()
    assert node_embedding is None


@pytest.mark.parametrize("config", [{}, {"encoder": None}, {"encoder": []}])
def test_loader_rejects_malformed_config(tmp_path, config):
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="encoder object"):
        ckpt.load_ckpt(str(tmp_path))


def test_loader_rejects_normalization_collision(tmp_path, monkeypatch):
    monkeypatch.setattr(
        ckpt.torch,
        "load",
        lambda *args, **kwargs: {"a._orig_mod.b": torch.ones(1), "a.b": torch.ones(1)},
    )
    with pytest.raises(ValueError, match="collision"):
        ckpt._load_state_dict(tmp_path / "weights.pt")


@pytest.mark.parametrize(
    "state, message",
    [
        ({"encoder.weight": torch.ones(2, 2)}, "missing="),
        ({"encoder.weight": torch.ones(2, 2), "encoder.bias": torch.ones(2), "encoder.extra": torch.ones(1)}, "unexpected="),
        ({"encoder.weight": torch.ones(3, 2), "encoder.bias": torch.ones(2)}, "shape/type mismatch"),
        ({"encoder.weight": torch.ones(2, 2), "encoder.bias": torch.ones(2), "other.weight": torch.ones(1)}, "Unexpected non-encoder"),
    ],
)
def test_loader_requires_complete_encoder_state(tmp_path, monkeypatch, state, message):
    monkeypatch.setattr(ckpt, "_build_encoder", lambda config: torch.nn.Linear(2, 2))
    monkeypatch.setattr(ckpt, "_load_state_dict", lambda path: state)
    (tmp_path / "config.json").write_text('{"encoder": {}}')
    (tmp_path / "weights.pt").write_bytes(b"unused")
    with pytest.raises(ValueError, match=message):
        ckpt.load_ckpt(str(tmp_path))
