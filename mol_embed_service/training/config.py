"""Configuration for supervised molecular fine-tuning."""

import math
from dataclasses import dataclass, field
from pathlib import Path


def _primitive(value, path):
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{path} values must be finite")
        return value
    if isinstance(value, (tuple, list)):
        return [_primitive(item, f"{path}[]") for item in value]
    if isinstance(value, dict) and all(type(k) is str for k in value):
        return {k: _primitive(v, f"{path}.{k}") for k, v in sorted(value.items())}
    raise TypeError(f"{path} must contain only JSON-compatible primitive values")


def _finite_number(value, name, *, minimum=None, maximum=None, minimum_inclusive=True):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if minimum is not None and (value < minimum or (not minimum_inclusive and value == minimum)):
        op = "at least" if minimum_inclusive else "greater than"
        raise ValueError(f"{name} must be {op} {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be at most {maximum}")


@dataclass
class ActivationConfig:
    name: str = "ReLU"
    kwargs: dict = field(default_factory=dict)


@dataclass
class HeadConfig:
    hidden_dims: tuple[int, ...] = ()
    normalization: str = "none"
    normalization_kwargs: dict = field(default_factory=dict)
    activation: ActivationConfig = field(default_factory=ActivationConfig)
    dropout: float = 0.0


@dataclass
class OptimizerConfig:
    name: str = "AdamW"
    kwargs: dict = field(default_factory=lambda: {"lr": 1e-4})


@dataclass
class EarlyStoppingConfig:
    patience: int = 10
    min_delta: float = 0.0


@dataclass
class TrainingConfig:
    model: str
    task: str
    head: HeadConfig = field(default_factory=HeadConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    epochs: int = 20
    batch_size: int = 32
    train_backbone: bool = True
    early_stopping: EarlyStoppingConfig | None = field(default_factory=EarlyStoppingConfig)
    seed: int = 0
    device: str = "cpu"
    checkpoint_dir: str | None = None
    resume_from: str | None = None
    backbone_lr: float | None = None
    checkpoint_path: str | None = None
    revision: str | None = None

    def validate(self):
        if not isinstance(self.model, str) or self.model not in {
            "chemberta-v1", "chemberta-v2", "chemberta-v3", "chemeleon"
        }:
            raise ValueError(f"Unsupported trainable model: {self.model}")
        if not isinstance(self.task, str) or self.task not in {"regression", "binary"}:
            raise ValueError("task must be 'regression' or 'binary'")
        for key in ("epochs", "batch_size"):
            if type(getattr(self, key)) is not int or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be a positive integer")
        if type(self.seed) is not int or not 0 <= self.seed <= 2**32 - 1:
            raise ValueError("seed must be an integer in [0, 2**32-1] accepted by NumPy and PyTorch")
        if type(self.train_backbone) is not bool:
            raise TypeError("train_backbone must be bool")
        if not isinstance(self.head, HeadConfig):
            raise TypeError("head must be HeadConfig")
        if not isinstance(self.optimizer, OptimizerConfig):
            raise TypeError("optimizer must be OptimizerConfig")
        if not isinstance(self.head.activation, ActivationConfig):
            raise TypeError("head.activation must be ActivationConfig")
        widths = self.head.hidden_dims
        if not isinstance(widths, (tuple, list)) or any(type(x) is not int or x <= 0 for x in widths):
            raise ValueError("head.hidden_dims must contain positive integers")
        self.head.hidden_dims = tuple(widths)
        for obj, name in ((self.head.activation, "activation"), (self.optimizer, "optimizer")):
            if not isinstance(obj.name, str) or not obj.name:
                raise ValueError(f"{name}.name must be a non-empty string")
            if not isinstance(obj.kwargs, dict):
                raise TypeError(f"{name}.kwargs must be a mapping")
            obj.kwargs = _primitive(obj.kwargs, f"{name}.kwargs")
        if not isinstance(self.head.normalization, str):
            raise TypeError("head.normalization must be a string")
        if not isinstance(self.head.normalization_kwargs, dict):
            raise TypeError("head.normalization_kwargs must be a mapping")
        self.head.normalization_kwargs = _primitive(
            self.head.normalization_kwargs, "head.normalization_kwargs"
        )
        if "params" in self.optimizer.kwargs:
            raise ValueError("optimizer.params is controlled by the trainer")
        _finite_number(self.head.dropout, "head.dropout", minimum=0, maximum=1)
        if self.head.dropout == 1:
            raise ValueError("head.dropout must be less than 1")
        if self.backbone_lr is not None:
            _finite_number(self.backbone_lr, "backbone_lr", minimum=0, minimum_inclusive=False)
        if self.early_stopping is not None:
            e = self.early_stopping
            if not isinstance(e, EarlyStoppingConfig):
                raise TypeError("early_stopping must be EarlyStoppingConfig or None")
            if type(e.patience) is not int or e.patience <= 0:
                raise ValueError("early_stopping.patience must be a positive integer")
            _finite_number(e.min_delta, "early_stopping.min_delta", minimum=0)
        if self.model == "chemeleon" and self.revision is not None:
            raise ValueError("revision applies only to ChemBERTa")
        if self.model != "chemeleon" and self.checkpoint_path is not None:
            raise ValueError("checkpoint_path applies only to CheMeleon")
        for key in ("checkpoint_path", "checkpoint_dir", "resume_from"):
            value = getattr(self, key)
            if value is not None:
                if not isinstance(value, (str, Path)) or not str(value):
                    raise ValueError(f"{key} must be a usable path")
                setattr(self, key, str(value))
        if self.revision is not None and (not isinstance(self.revision, str) or not self.revision):
            raise ValueError("revision must be a non-empty string")
        if not isinstance(self.device, str) or not self.device:
            raise ValueError("device must be a non-empty string")
        try:
            import torch

            torch.device(self.device)
        except (RuntimeError, TypeError, ValueError) as exc:
            raise ValueError(f"device is not a valid PyTorch device: {self.device!r}") from exc

        # Preflight choices even when the current head shape does not exercise them.
        from .components import make_activation, make_norm, make_optimizer

        try:
            activation_widths = self.head.hidden_dims or (None,)
            for width in dict.fromkeys(activation_widths):
                make_activation(self.head.activation, width)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"head.activation: {exc}") from exc
        try:
            make_norm(self.head.normalization, 4, self.head.normalization_kwargs)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"head.normalization: {exc}") from exc
        try:
            import torch

            make_optimizer(
                self.optimizer.name, self.optimizer.kwargs,
                [torch.nn.Parameter(torch.zeros(1))],
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"optimizer: {exc}") from exc
        return self

    def canonical(self):
        import dataclasses

        result = dataclasses.asdict(self)
        result["head"]["hidden_dims"] = list(self.head.hidden_dims)
        return _primitive(result, "config")
