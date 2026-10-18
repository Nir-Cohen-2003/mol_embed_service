import sys
import types
from typing import ClassVar

from mol_embed_service import minimol_backend


def test_load_state_dict_uses_cpu_and_weights_only(monkeypatch, tmp_path):
    import torch

    observed = {}

    def fake_load(path, **kwargs):
        observed["path"] = path
        observed.update(kwargs)
        return {"weight": torch.tensor([1.0])}

    monkeypatch.setattr(torch, "load", fake_load)
    path = tmp_path / "weights.pth"
    assert "weight" in minimol_backend._load_state_dict(path)
    assert observed == {"path": path, "map_location": "cpu", "weights_only": True}


def test_bridge_requires_reviewed_dependency_versions(monkeypatch):
    versions = {"minimol": "1.3.4", "graphium": "2.4.7"}
    monkeypatch.setattr(minimol_backend, "version", versions.__getitem__)
    minimol_backend._require_versions()
    versions["graphium"] = "2.5.0"
    try:
        minimol_backend._require_versions()
    except ImportError as exc:
        assert "graphium==2.4.7" in str(exc)
    else:
        raise AssertionError("incompatible Graphium version was accepted")


def test_constructor_routes_verified_assets_through_upstream_loader(monkeypatch, tmp_path):
    events = []
    asset_dir = tmp_path / "assets"
    asset_dir.mkdir()
    for name in ("config.yaml", "base_shape.yaml", "state_dict.pth"):
        (asset_dir / name).touch()
    monkeypatch.setattr(minimol_backend, "version", lambda package: {"minimol": "1.3.4", "graphium": "2.4.7"}[package])
    monkeypatch.setattr(minimol_backend, "resolve_checkpoint", lambda **kwargs: asset_dir)
    state = {"weights": object()}
    monkeypatch.setattr(minimol_backend, "_load_state_dict", lambda path: events.append(("state", path)) or state)

    class OmegaConf:
        @staticmethod
        def load(path):
            events.append(("config-load", path))
            return {"architecture": {}, "accelerator": {"type": "gpu"}}

        @staticmethod
        def to_container(cfg, *, resolve):
            events.append(("config-container", resolve))
            return cfg

    class Datamodule:
        in_dims: ClassVar[dict] = {"atom": 3}
        featurization = "features"
        task_norms = "norms"
        batch_size_training = 8

        def get_task_levels(self):
            return ["graph"]

    class Predictor:
        def load_state_dict(self, state_dict, *, strict):
            events.append(("load-state", state_dict, strict))

    class Fingerprinter:
        def __init__(self, predictor, head):
            events.append(("fingerprinter", predictor, head))
            self.predictor = predictor

        def setup(self):
            events.append(("setup",))

    class UpstreamMinimol:
        def set_training_mode_false(self, predictor):
            events.append(("inference-mode", predictor))

        def inherited_inference(self):
            return "upstream-inference"

        def inherited_training_helper(self):
            return "upstream-training-helper"

    def module(name, **attrs):
        value = types.ModuleType(name)
        value.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, value)

    def accelerator(cfg):
        events.append(("accelerator", cfg["accelerator"]["type"]))
        cfg.setdefault("accelerator", {})["type"] = "cpu"
        return cfg, "cpu"

    def datamodule(cfg, accelerator_type):
        events.append(("datamodule", accelerator_type))
        return Datamodule()

    def architecture(cfg, *, in_dims):
        events.append(("architecture", in_dims))
        return "Model", {"layers": 2}

    def metrics(cfg):
        events.append(("metrics", cfg is not None))
        return ["metric"]

    def predictor(**kwargs):
        events.append(("predictor", kwargs))
        return Predictor()

    module("graphium", __path__=[])
    module("graphium.config", __path__=[])
    module("graphium.config._loader", load_accelerator=accelerator, load_architecture=architecture,
           load_datamodule=datamodule, load_metrics=metrics, load_predictor=predictor)
    module("graphium.finetuning", __path__=[])
    module("graphium.finetuning.fingerprinting", Fingerprinter=Fingerprinter)
    module("minimol", __path__=[])
    module("minimol.model", Minimol=UpstreamMinimol)
    module("omegaconf", OmegaConf=OmegaConf)

    result = minimol_backend.create_minimol(cache_dir=tmp_path / "cache", batch_size=7)
    assert result.batch_size == 7
    assert result.inherited_inference() == "upstream-inference"
    assert result.inherited_training_helper() == "upstream-training-helper"
    assert [item[0] for item in events] == [
        "config-load", "config-container", "accelerator", "datamodule", "architecture", "metrics",
        "predictor", "inference-mode", "state", "load-state", "fingerprinter", "setup",
    ]
    assert events[0][1] == str(asset_dir / "config.yaml")
    assert events[2][1] == "cpu"
    predictor_args = events[6][1]
    assert predictor_args == {
        "config": {"architecture": {"mup_base_path": str((asset_dir / "base_shape.yaml").resolve())}, "accelerator": {"type": "cpu"}},
        "model_class": "Model", "model_kwargs": {"layers": 2}, "metrics": ["metric"],
        "task_levels": ["graph"], "accelerator_type": "cpu", "featurization": "features",
        "task_norms": "norms", "replicas": 1, "gradient_acc": 1, "global_bs": 8,
    }
    assert events[9] == ("load-state", state, False)
    assert events[10][2] == "gnn:15"
    assert events[11] == ("setup",)
