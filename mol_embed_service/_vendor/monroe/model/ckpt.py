"""Safe, inference-only loader for pinned Monroe checkpoints."""
import json
from pathlib import Path

import torch

from mol_embed_service._vendor.monroe.model.constants import (
    EDGE_FEAT_LIST_ONE_HOT,
    NODE_FEAT_LIST_FLOAT,
    NODE_FEAT_LIST_ONE_HOT,
)
from mol_embed_service._vendor.monroe.model.grit import GritTransformer


def _resolve_ckpt_dir(path: str | Path) -> Path:
    ckpt_dir = Path(path)
    if ckpt_dir.is_file():
        ckpt_dir = ckpt_dir.parent
    return ckpt_dir


def _load_state_dict(weights_path: Path) -> dict:
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not all(
        isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in state.items()
    ):
        raise TypeError("Monroe checkpoint must contain a string-keyed tensor state dictionary")
    normalized = {}
    for key, value in state.items():
        # Upstream torch.compile state keys use the exact ._orig_mod. wrapper.
        key = key.replace("._orig_mod.", ".")
        if key.startswith("_orig_mod."):
            key = key.removeprefix("_orig_mod.")
        if key in normalized:
            raise ValueError(f"Monroe checkpoint key normalization collision: {key}")
        normalized[key] = value
    return normalized


def _build_encoder(config: dict) -> GritTransformer:
    if not isinstance(config, dict) or not isinstance(config.get("encoder"), dict):
        raise ValueError("Monroe config must contain an encoder object")
    encoder_cfg = dict(config["encoder"])
    if "edge_rbf_dim" in encoder_cfg and "rbf_dim" not in encoder_cfg:
        encoder_cfg["rbf_dim"] = encoder_cfg.pop("edge_rbf_dim")
    else:
        encoder_cfg.pop("edge_rbf_dim", None)
    for key in ("node_float_rbf", "node_float_missing"):
        encoder_cfg.pop(key, None)
    return GritTransformer(
        node_feature_vocab=NODE_FEAT_LIST_ONE_HOT,
        edge_feature_vocab=EDGE_FEAT_LIST_ONE_HOT,
        node_float_dim=len(NODE_FEAT_LIST_FLOAT),
        **encoder_cfg,
    )


def load_ckpt(path: str, use_ema: bool = False):
    """Load and fully validate the encoder from a checkpoint directory."""
    directory = _resolve_ckpt_dir(path)
    if use_ema:
        raise ValueError("Bundled Monroe inference supports weights.pt, not EMA weights")
    with (directory / "config.json").open(encoding="utf-8") as stream:
        config = json.load(stream)
    encoder = _build_encoder(config)
    state = _load_state_dict(directory / "weights.pt")
    allowed_prefixes = ("decoders.", "dataset_heads.")
    bad = [key for key in state if not key.startswith("encoder.") and not key.startswith(allowed_prefixes)]
    if bad:
        raise ValueError(f"Unexpected non-encoder Monroe checkpoint keys: {bad[:5]}")
    encoder_state = {key[len("encoder."):]: value for key, value in state.items() if key.startswith("encoder.")}
    expected = encoder.state_dict()
    missing = sorted(set(expected) - set(encoder_state))
    unexpected = sorted(set(encoder_state) - set(expected))
    if missing or unexpected:
        raise ValueError(f"Monroe encoder state mismatch; missing={missing[:5]}, unexpected={unexpected[:5]}")
    for key, target in expected.items():
        value = encoder_state[key]
        if not isinstance(value, torch.Tensor) or value.shape != target.shape:
            raise ValueError(f"Monroe encoder tensor shape/type mismatch for {key}")
    encoder.load_state_dict(encoder_state, strict=True)
    return encoder.eval()
