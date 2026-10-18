"""Constructor-only bridge to the pinned MiniMol implementation."""

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from .minimol_checkpoint import resolve_checkpoint


def _require_versions() -> None:
    expected = {"minimol": "1.3.4", "graphium": "2.4.7"}
    for package, required in expected.items():
        try:
            installed = version(package)
        except PackageNotFoundError as exc:
            raise ImportError(f"MiniMol requires {package}=={required}; install the pinned optional stack") from exc
        if installed != required:
            raise ImportError(
                f"MiniMol constructor bridge supports {package}=={required}, found {installed}; "
                "install the pinned MiniMol optional stack"
            )


def _load_state_dict(path: Path):
    """Load verified tensor-only checkpoint data on CPU without unsafe fallback."""
    import torch

    return torch.load(path, map_location="cpu", weights_only=True)


def create_minimol(*, cache_dir: str | Path | None = None, batch_size: int = 100):
    """Build the upstream model while routing its constructor to verified assets.

    The class inherits upstream inference and featurization methods unchanged;
    only initialization is adapted because upstream hardcodes package resources.
    """
    _require_versions()
    asset_dir = resolve_checkpoint(cache_dir=cache_dir)
    from graphium.config._loader import (
        load_accelerator,
        load_architecture,
        load_datamodule,
        load_metrics,
        load_predictor,
    )
    from graphium.finetuning.fingerprinting import Fingerprinter
    from minimol.model import Minimol
    from omegaconf import OmegaConf

    class CachedMinimol(Minimol):
        def __init__(self):
            self.batch_size = batch_size
            cfg = OmegaConf.to_container(OmegaConf.load(str(asset_dir / "config.yaml")), resolve=True)
            cfg["accelerator"]["type"] = "cpu"
            self.cfg, accelerator_type = load_accelerator(cfg)
            self.cfg["architecture"]["mup_base_path"] = str((asset_dir / "base_shape.yaml").resolve())
            self.datamodule = load_datamodule(self.cfg, accelerator_type)
            model_class, model_kwargs = load_architecture(cfg, in_dims=self.datamodule.in_dims)
            metrics = load_metrics(self.cfg)
            predictor = load_predictor(
                config=self.cfg,
                model_class=model_class,
                model_kwargs=model_kwargs,
                metrics=metrics,
                task_levels=self.datamodule.get_task_levels(),
                accelerator_type=accelerator_type,
                featurization=self.datamodule.featurization,
                task_norms=self.datamodule.task_norms,
                replicas=1,
                gradient_acc=1,
                global_bs=self.datamodule.batch_size_training,
            )
            self.set_training_mode_false(predictor)
            state_dict = _load_state_dict(asset_dir / "state_dict.pth")
            predictor.load_state_dict(state_dict, strict=False)
            self.predictor = Fingerprinter(predictor, "gnn:15")
            self.predictor.setup()

    return CachedMinimol()
