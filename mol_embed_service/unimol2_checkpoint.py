"""Verified, atomic acquisition and restricted loading of official Uni-Mol2 weights."""

import hashlib
import os
import ssl
import tempfile
from pathlib import Path
from urllib.request import urlopen

import torch

from .loading_policy import offline_enabled, reject_offline_download
from .unimol2_config import CONFIGS, HF_REVISION


def _verified(path, config):
    digest = hashlib.sha256()
    size = 0
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            size += len(block)
            digest.update(block)
    if size != config.bytes or digest.hexdigest() != config.sha256:
        raise ValueError(
            f"Uni-Mol2 checkpoint does not match {config.name}: size={size}, sha256={digest.hexdigest()}"
        )


def resolve_checkpoint(name, checkpoint_path=None, cache_dir=None):
    config = CONFIGS[name]
    explicit = checkpoint_path or os.environ.get("UNIMOL2_CHECKPOINT_PATH")
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Uni-Mol2 checkpoint not found: {path}")
        _verified(path, config)
        return path
    root = Path(cache_dir or os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    if "XDG_CACHE_HOME" not in os.environ and cache_dir is None:
        root = Path.home() / ".cache"
    target = (
        root / "mol_embed_service" / "unimol2" / config.directory / HF_REVISION / "checkpoint.pt"
    )
    if target.is_file():
        try:
            _verified(target, config)
            return target
        except ValueError:
            if offline_enabled():
                reject_offline_download("Uni-Mol2", config.name, target)
    elif offline_enabled():
        reject_offline_download("Uni-Mol2", config.name, target)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=".checkpoint-", dir=target.parent)
    try:
        digest = hashlib.sha256()
        size = 0
        context = ssl._create_unverified_context()
        with os.fdopen(fd, "wb") as output, urlopen(
            config.url, timeout=60, context=context
        ) as response:
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                size += len(block)
                digest.update(block)
                output.write(block)
        if size != config.bytes or digest.hexdigest() != config.sha256:
            raise ValueError(
                f"Downloaded Uni-Mol2 artifact failed verification: size={size}, sha256={digest.hexdigest()}"
            )
        os.replace(temp, target)
        return target
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def load_checkpoint(name, path=None, cache_dir=None):
    config = CONFIGS[name]
    resolved = resolve_checkpoint(name, path, cache_dir)
    payload = torch.load(resolved, map_location="cpu", weights_only=True)
    if (
        not isinstance(payload, dict)
        or set(payload) != {"model"}
        or not isinstance(payload["model"], dict)
    ):
        raise ValueError(
            "Unexpected Uni-Mol2 checkpoint container; expected {'model': tensor mapping}"
        )
    state = payload["model"]
    if any(not isinstance(k, str) or not isinstance(v, torch.Tensor) for k, v in state.items()):
        raise ValueError("Uni-Mol2 checkpoint model must map string keys to tensors")
    from .unimol2_model import UniMol2Model

    with torch.device("meta"):
        model = UniMol2Model(name)
    expected = model.state_dict()
    allowed_unused = {
        "lm_head.weight",
        "lm_head.bias",
        "lm_head.dense.weight",
        "lm_head.dense.bias",
        "lm_head.layer_norm.weight",
        "lm_head.layer_norm.bias",
        "movement_pred_head.layer_norm.weight",
        "movement_pred_head.layer_norm.bias",
        "movement_pred_head.q_proj.weight",
        "movement_pred_head.k_proj.weight",
        "movement_pred_head.v_proj.weight",
        "movement_pred_head.force_proj1.weight",
        "movement_pred_head.linear_bias.weight",
        "movement_pred_head.linear_bias.bias",
        "movement_pred_head.pair_layer_norm.weight",
        "movement_pred_head.pair_layer_norm.bias",
    }
    unexpected = [key for key in state if key not in expected and key not in allowed_unused]
    missing = [key for key in expected if key not in state]
    if missing or unexpected:
        raise ValueError(
            f"Uni-Mol2 checkpoint key coverage failed: missing={missing[:8]}, unexpected={unexpected[:8]}"
        )
    for key, target in expected.items():
        value = state[key]
        if value.shape != target.shape:
            raise ValueError(
                f"Uni-Mol2 checkpoint tensor {key} has shape {tuple(value.shape)}, expected {tuple(target.shape)}"
            )
        if value.dtype != target.dtype:
            raise ValueError(
                f"Uni-Mol2 checkpoint tensor {key} has dtype {value.dtype}, expected {target.dtype}"
            )
        if not torch.isfinite(value).all():
            raise ValueError(f"Uni-Mol2 checkpoint tensor {key} contains non-finite values")
    del model
    return config, state
