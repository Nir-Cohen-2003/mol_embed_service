"""Safe, self-contained model artifact serialization."""

import os
import tempfile
from pathlib import Path

import torch

_SCHEMA = 1
_MAX_ASSETS = 64
_MAX_ASSET_BYTES = 16 * 1024 * 1024
_MAX_TOTAL_ASSETS = 64 * 1024 * 1024


def _dependency_versions():
    from importlib.metadata import PackageNotFoundError, version

    result = {}
    for package in ("torch", "transformers", "tokenizers", "chemprop"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            continue
    return result


def _validate_data(value, path="payload"):
    if value is None or type(value) in (str, bool, int, float, bytes):
        if type(value) is float and not torch.isfinite(torch.tensor(value)).item():
            raise ValueError(f"{path} contains a non-finite float")
        return
    if isinstance(value, torch.Tensor):
        if value.layout != torch.strided or value.is_sparse:
            raise ValueError(f"{path} contains an unsupported tensor")
        if (value.is_floating_point() or value.is_complex()) and not torch.isfinite(value).all():
            raise ValueError(f"{path} contains non-finite tensor values")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_data(item, f"{path}[{index}]")
        return
    if isinstance(value, dict) and all(type(key) in (str, int) for key in value):
        for key, item in value.items():
            _validate_data(item, f"{path}.{key}")
        return
    raise TypeError(f"{path} contains unsupported value type {type(value).__name__}")


def _validate_target_scaler_state(state, outputs):
    mean = state.get("target_mean") if isinstance(state, dict) else None
    scale = state.get("target_scale") if isinstance(state, dict) else None
    if (not isinstance(mean, torch.Tensor) or not isinstance(scale, torch.Tensor)
            or tuple(mean.shape) != (outputs,) or tuple(scale.shape) != (outputs,)
            or not torch.isfinite(mean).all() or not torch.isfinite(scale).all()
            or torch.any(scale <= 0)):
        raise ValueError("Target scaler state must have the expected shape, finite means, and positive scales")


def model_payload(model):
    config = getattr(model, "training_config", None)
    if config is None:
        raise ValueError("Model lacks reconstruction metadata; cannot save a self-contained artifact")
    config.validate()
    if config.model not in {"chemberta-v1", "chemberta-v2", "chemberta-v3", "chemeleon"}:
        raise ValueError("Unsupported model backend")
    source = {"backend": config.model, "dependencies": _dependency_versions()}
    if config.model.startswith("chemberta-"):
        encoder = model.backbone.encoder
        model_config = encoder.config.to_dict()
        if model_config.get("auto_map"):
            raise ValueError("Custom-code auto_map configs cannot be exported safely")
        source["encoder_config"] = model_config
        source["preprocessing"] = "chemberta_mean_nonpadding_v1"
        resolved_revision = getattr(encoder.config, "_commit_hash", None)
        if resolved_revision:
            source["resolved_revision"] = resolved_revision
        try:
            from ..models import ChemBERTaEmbedder
            source["identifier"] = ChemBERTaEmbedder.MODEL_NAMES[config.model]
        except (ImportError, KeyError):
            pass
        if config.revision:
            source["requested_revision"] = config.revision
        files = {}
        with tempfile.TemporaryDirectory() as directory:
            model.backbone.tokenizer.save_pretrained(directory)
            for candidate in Path(directory).iterdir():
                if candidate.is_file():
                    files[candidate.name] = candidate.read_bytes()
        source["tokenizer_files"] = files
    elif config.model == "chemeleon":
        message_passing = model.backbone.encoder.message_passing
        if (type(message_passing.V_d_transform).__name__ != "Identity"
                or type(message_passing.graph_transform).__name__ != "Identity"):
            raise ValueError("CheMeleon artifacts support only identity graph/descriptor transforms")
        activation = message_passing.hparams.get("activation")
        if activation not in {"relu", "ReLU", "RELU"}:
            raise ValueError(f"Unsupported CheMeleon activation: {activation!r}")
        source["message_passing"] = {
            key: message_passing.hparams[key]
            for key in ("d_v", "d_e", "d_h", "bias", "depth", "dropout", "undirected", "d_vd")
        }
        source["message_passing"]["activation"] = "relu"
        source["pooling"] = "mean"
        source["normalization"] = "identity"
        source["preprocessing"] = "chemeleon_mean_v1"
        if message_passing.hparams.get("d_vd", 0) not in (None, 0):
            raise ValueError("CheMeleon artifacts do not support descriptor dimension d_vd")
        provenance = getattr(model.backbone, "checkpoint_provenance", None)
        if provenance is not None:
            source["checkpoint_provenance"] = dict(provenance)
    else:
        raise ValueError("Unsupported model backend")
    payload = {
        "schema": _SCHEMA, "source": source, "kind": "model",
        "config": {k: getattr(config, k) for k in ("model", "task", "revision", "checkpoint_path")},
        "head_config": {
            "hidden_dims": list(model.head_config.hidden_dims),
            "normalization": model.head_config.normalization,
            "normalization_kwargs": model.head_config.normalization_kwargs,
            "activation": {"name": model.head_config.activation.name, "kwargs": model.head_config.activation.kwargs},
            "dropout": model.head_config.dropout,
        },
        "embedding_dim": int(model.embedding_dim), "num_outputs": int(model.num_outputs),
        "train_backbone": model.train_backbone,
        "state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
    }
    _validate_data(payload)
    return payload


def atomic_save(payload, path):
    _validate_data(payload)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent)
    os.close(fd)
    try:
        torch.save(payload, tmp)
        with open(tmp, "rb") as stream:
            os.fsync(stream.fileno())
        os.replace(tmp, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def save_model(model, path):
    atomic_save(model_payload(model), path)


def load_model(path, device="cpu"):
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(f"Invalid or unsafe model artifact: {exc}") from exc
    return load_model_payload(payload, device)


def _validate_assets(files):
    if not isinstance(files, dict) or not files or len(files) > _MAX_ASSETS:
        raise ValueError("Invalid tokenizer assets")
    total = 0
    for name, content in files.items():
        if (type(name) is not str or name in {".", ".."} or "/" in name or "\\" in name
                or not name or type(content) is not bytes or len(content) > _MAX_ASSET_BYTES):
            raise ValueError("Unsafe tokenizer asset payload")
        total += len(content)
    if total > _MAX_TOTAL_ASSETS:
        raise ValueError("Tokenizer assets exceed size limit")


def load_model_payload(payload, device="cpu"):
    from .config import ActivationConfig, HeadConfig, TrainingConfig
    from .model import FineTunedModel
    _validate_data(payload)
    if not isinstance(payload, dict) or payload.get("schema") != _SCHEMA or payload.get("kind") != "model":
        raise ValueError("Unsupported model artifact schema/kind")
    try:
        c, h, source = payload["config"], payload["head_config"], payload["source"]
        if not isinstance(c, dict) or not isinstance(h, dict) or not isinstance(source, dict):
            raise TypeError("Invalid artifact configuration")
        if source.get("backend") != c.get("model"):
            raise ValueError("Artifact backend mismatch")
        cfg = TrainingConfig(model=c["model"], task=c["task"], device=device,
                             revision=c.get("revision"), checkpoint_path=c.get("checkpoint_path"))
        cfg.validate()
        outputs = payload["num_outputs"]
        if type(outputs) is not int or outputs <= 0:
            raise ValueError("Invalid task/output dimensions")
        if type(payload["embedding_dim"]) is not int or payload["embedding_dim"] <= 0:
            raise ValueError("Invalid embedding dimension")
        if type(payload["train_backbone"]) is not bool:
            raise ValueError("Invalid train_backbone value")
        head = HeadConfig(tuple(h["hidden_dims"]), h["normalization"], h["normalization_kwargs"],
                          ActivationConfig(**h["activation"]), h["dropout"])
        cfg.head = head
        cfg.validate()
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError(f"Invalid artifact configuration: {exc}") from exc
    if cfg.model.startswith("chemberta-"):
        from transformers import AutoConfig, AutoModel, AutoTokenizer
        model_config = source.get("encoder_config")
        if not isinstance(model_config, dict) or model_config.get("model_type") != "roberta" or model_config.get("auto_map"):
            raise ValueError("Invalid saved ChemBERTa config")
        _validate_assets(source.get("tokenizer_files"))
        with tempfile.TemporaryDirectory() as directory:
            for name, content in source["tokenizer_files"].items():
                (Path(directory) / name).write_bytes(content)
            tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True, trust_remote_code=False)
            values = dict(model_config); values.pop("model_type", None)
            model_config = AutoConfig.for_model("roberta", **values)
        resolved_revision = source.get("resolved_revision")
        if resolved_revision is not None:
            if type(resolved_revision) is not str or not resolved_revision:
                raise ValueError("Invalid saved ChemBERTa resolved revision")
            model_config._commit_hash = resolved_revision
        encoder = AutoModel.from_config(model_config, trust_remote_code=False)
        from .backbones import ChemBERTaBackbone
        backbone = ChemBERTaBackbone(encoder, tokenizer, int(encoder.config.hidden_size))
    else:
        from chemprop import featurizers, nn

        from .backbones import CheMeleonBackbone, CheMeleonEncoder
        schema = source.get("message_passing")
        if not isinstance(schema, dict) or source.get("pooling") != "mean" or source.get("normalization") != "identity":
            raise ValueError("Invalid CheMeleon reconstruction schema")
        if schema.get("d_vd", 0) not in (None, 0):
            raise ValueError("CheMeleon artifacts do not support descriptor dimension d_vd")
        backbone = CheMeleonBackbone(CheMeleonEncoder(nn.BondMessagePassing(**schema), nn.MeanAggregation()),
                                    featurizers.SimpleMoleculeMolGraphFeaturizer(),
                                    source.get("checkpoint_provenance"))
    if int(backbone.dimension) != payload["embedding_dim"]:
        raise ValueError("Artifact embedding dimension mismatch")
    model = FineTunedModel(backbone, head, cfg.task, outputs, train_backbone=payload["train_backbone"])
    state = payload.get("state")
    if not isinstance(state, dict) or not all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in state.items()):
        raise ValueError("Invalid model state")
    model.load_state_dict(state, strict=True)
    if model.task != cfg.task or model.num_outputs != outputs:
        raise ValueError("Artifact task/head dimensions are inconsistent")
    _validate_target_scaler_state(model.state_dict(), outputs)
    model.training_config = cfg
    model.to(device).eval()
    return model
