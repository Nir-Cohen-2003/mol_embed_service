"""Bounded PyTorch component construction for training heads."""

import copy

import torch
from torch import nn

# Only actual activations are admitted; nn namespace also contains layers such as Linear.
_ACTIVATION_MODULE = "torch.nn.modules.activation"
EXCLUDED_ACTIVATIONS = {"GLU", "Softmax2d", "MultiheadAttention"}
_UNSUPPORTED_OPTIMIZERS = {"LBFGS", "SparseAdam", "Muon"}


def activation_names():
    return sorted(
        name
        for name in dir(nn)
        if not name.startswith("_")
        and isinstance(getattr(nn, name), type)
        and issubclass(getattr(nn, name), nn.Module)
        and getattr(nn, name).__module__ == _ACTIVATION_MODULE
        and name not in EXCLUDED_ACTIVATIONS
        and name not in {"Module", "Sequential", "Identity"}
    )


def make_activation(spec, width=None):
    name, kwargs = spec.name, dict(spec.kwargs)
    if name == "Identity":
        if kwargs:
            raise ValueError("Identity does not accept activation kwargs")
        return nn.Identity()
    if name not in activation_names():
        raise ValueError(f"Unsupported activation {name!r}; accepted: {activation_names()}")
    if name in {"Softmax", "LogSoftmax", "Softmin"} and kwargs.get("dim") not in (1, -1):
        raise ValueError(f"{name} requires dim=1 or dim=-1 for dense hidden features")
    rng_state = torch.random.get_rng_state()
    try:
        try:
            module = getattr(nn, name)(**kwargs)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid activation {name} kwargs={kwargs}: {exc}") from exc
        if width is not None:
            probe = copy.deepcopy(module)
            with torch.enable_grad():
                leaf = torch.randn(3, width, requires_grad=True)
                sample = leaf + 0  # in-place activations must receive a non-leaf tensor
                output = probe(sample)
                if output.shape != sample.shape or not torch.isfinite(output).all():
                    raise ValueError("activation must return finite [batch,width] values")
                output.sum().backward()
                if leaf.grad is None or not torch.isfinite(leaf.grad).all():
                    raise ValueError("activation must provide finite input gradients")
    except Exception as exc:
        raise ValueError(
            f"Activation {name} violates [batch,width] differentiable contract: {exc}"
        ) from exc
    finally:
        torch.random.set_rng_state(rng_state)
    return module


def make_norm(name, width, kwargs=None):
    kwargs = dict(kwargs or {})
    if name == "none":
        if kwargs:
            raise ValueError("normalization_kwargs must be empty when normalization='none'")
        return nn.Identity()
    if name == "batch":
        cls, reserved = nn.BatchNorm1d, {"num_features"}
    elif name == "layer":
        cls, reserved = nn.LayerNorm, {"normalized_shape"}
    elif name == "rms":
        cls, reserved = getattr(nn, "RMSNorm", None), {"normalized_shape"}
        if cls is None:
            raise ValueError("normalization='rms' requires torch.nn.RMSNorm")
    else:
        raise ValueError("normalization must be batch, layer, rms, or none")
    if reserved.intersection(kwargs):
        raise ValueError(
            f"normalization dimension is supplied by the head, not kwargs: {reserved.intersection(kwargs)}"
        )
    if name == "batch" and kwargs.get("track_running_stats") is False:
        raise ValueError(
            "batch normalization requires track_running_stats=True for singleton prediction"
        )
    try:
        return cls(width, **kwargs)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {name} normalization kwargs={kwargs}: {exc}") from exc


def make_optimizer(name, kwargs, params):
    kwargs = dict(kwargs)
    if "params" in kwargs:
        raise ValueError("optimizer.params is supplied by the trainer")
    if not isinstance(name, str):
        raise TypeError(f"Optimizer name must be a string, got {type(name).__name__}")
    if name in _UNSUPPORTED_OPTIMIZERS:
        raise ValueError(f"Optimizer {name} is unsupported by dense mini-batch training")
    cls = getattr(torch.optim, name, None)
    if (
        not isinstance(cls, type)
        or not issubclass(cls, torch.optim.Optimizer)
        or name.startswith("_")
        or name == "Optimizer"
    ):
        names = sorted(
            n for n in dir(torch.optim)
            if isinstance(getattr(torch.optim, n), type)
            and issubclass(getattr(torch.optim, n), torch.optim.Optimizer)
            and n not in _UNSUPPORTED_OPTIMIZERS | {"Optimizer"}
            and not n.startswith("_")
        )
        raise ValueError(f"Unsupported optimizer {name!r}; accepted: {names}")
    try:
        return cls(params, **kwargs)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid optimizer {name} kwargs={kwargs}: {exc}") from exc
