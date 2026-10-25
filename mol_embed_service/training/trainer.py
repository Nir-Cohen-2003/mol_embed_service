"""Explicit-split supervised training loop."""

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch

from .backbones import load_backbone
from .components import make_optimizer
from .model import FineTunedModel


def _checkpoint_model_payload(model):
    from .artifacts import model_payload

    return model_payload(model)


def _split(value, name):
    if not isinstance(value, tuple) or len(value) != 2:
        raise ValueError(f"{name} must be (smiles, targets)")
    smiles, y = value
    smiles = list(smiles)
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 1:
        y = y[:, None]
    if y.ndim != 2 or len(smiles) != len(y) or not len(y):
        raise ValueError(f"{name} has invalid SMILES/target shape")
    if not np.isfinite(y[~np.isnan(y)]).all():
        raise ValueError(f"{name} targets contain infinities")
    for i, s in enumerate(smiles):
        from rdkit import Chem

        if not isinstance(s, str) or not s.strip() or Chem.MolFromSmiles(s) is None:
            raise ValueError(f"Invalid SMILES in {name} row {i}: {s!r}")
    if np.isnan(y).all(axis=1).any():
        raise ValueError(f"{name} contains wholly unlabeled rows")
    return smiles, y


def finetune(config, train, validation):
    config.validate()
    xs, ys = _split(train, "train")
    xv, yv = _split(validation, "validation")
    if ys.shape[1] != yv.shape[1]:
        raise ValueError("train and validation target dimensions differ")
    if config.task == "binary":
        for split_name, targets in (("train", ys), ("validation", yv)):
            invalid = ~np.isnan(targets) & ~np.isin(targets, [0, 1])
            if invalid.any():
                row, column = np.argwhere(invalid)[0]
                raise ValueError(f"Binary target in {split_name} row {row}, target {column} must be 0 or 1")
    if np.isnan(ys).all(axis=0).any():
        raise ValueError("Every training target column must have observed labels")
    if np.isnan(yv).all():
        raise ValueError("Validation split has no observed labels")

    def fingerprint(smiles, targets):
        digest = hashlib.sha256(
            json.dumps(smiles, ensure_ascii=False, separators=(",", ":")).encode()
        )
        values = np.nan_to_num(targets, nan=0.0).astype("<f4")
        digest.update(values.tobytes())
        digest.update(np.isnan(targets).tobytes())
        return digest.hexdigest()

    data_identity = {"train": fingerprint(xs, ys), "validation": fingerprint(xv, yv)}
    if torch.is_inference_mode_enabled():
        raise RuntimeError("Training cannot run inside torch.inference_mode()")
    checkpoint = None
    if config.resume_from:
        try:
            checkpoint = torch.load(config.resume_from, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise ValueError(f"Invalid training checkpoint: {exc}") from exc
        required = {
            "kind", "schema", "config", "current_model", "data_identity", "completed_epochs",
            "current_state", "best_state", "optimizer_state", "best_loss", "best_epoch",
            "stale", "stopped", "history", "torch_rng", "python_rng", "numpy_rng",
            "torch_version", "dependencies", "cuda_rng", "global_step", "optimizer_names",
        }
        if not isinstance(checkpoint, dict) or not required.issubset(checkpoint):
            raise ValueError("Training checkpoint is malformed or missing required fields")
        if checkpoint.get("kind") != "training_checkpoint" or checkpoint.get("schema") != 1:
            raise ValueError("resume_from must be a version 1 training checkpoint")
        if checkpoint.get("current_model") is None:
            raise ValueError("Training checkpoint lacks embedded model reconstruction assets")
        completed = checkpoint.get("completed_epochs")
        if type(completed) is not int or completed < 0:
            raise ValueError("Training checkpoint has invalid completed epoch count")
        for key in ("global_step", "best_epoch", "stale"):
            if type(checkpoint.get(key)) is not int or checkpoint[key] < 0:
                raise ValueError(f"Training checkpoint has invalid {key}")
        history = checkpoint.get("history")
        if not isinstance(history, list) or len(history) != completed:
            raise ValueError("Training checkpoint history does not match completed epochs")
        for index, item in enumerate(history, 1):
            if (not isinstance(item, dict) or item.get("epoch") != index
                    or any(type(item.get(key)) not in (int, float) or not np.isfinite(item[key])
                           for key in ("train_loss", "validation_loss"))):
                raise ValueError("Training checkpoint contains malformed epoch history")
        for key in ("current_state", "best_state"):
            state = checkpoint.get(key)
            if not isinstance(state, dict) or not state or not all(
                isinstance(name, str) and isinstance(value, torch.Tensor)
                for name, value in state.items()
            ):
                raise ValueError(f"Training checkpoint has invalid {key}")
        if not isinstance(checkpoint.get("optimizer_state"), dict):
            raise ValueError("Training checkpoint has invalid optimizer state")
        from .artifacts import _validate_data, _validate_target_scaler_state
        for key in ("current_state", "best_state", "optimizer_state"):
            try:
                _validate_data(checkpoint[key], f"checkpoint.{key}")
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Training checkpoint has invalid {key}: {exc}") from exc
        outputs = checkpoint["current_model"].get("num_outputs") if isinstance(checkpoint["current_model"], dict) else None
        if type(outputs) is not int or outputs <= 0:
            raise ValueError("Training checkpoint has invalid embedded model output dimensions")
        for key in ("current_state", "best_state"):
            try:
                _validate_target_scaler_state(checkpoint[key], outputs)
            except ValueError as exc:
                raise ValueError(f"Training checkpoint has invalid {key} scaler: {exc}") from exc
        if type(checkpoint.get("best_loss")) not in (float, int) or not np.isfinite(checkpoint["best_loss"]):
            raise ValueError("Training checkpoint has invalid best loss")
        if type(checkpoint.get("stopped")) is not bool:
            raise ValueError("Training checkpoint has invalid stopped flag")
        expected = config.canonical()
        for key in ("epochs", "device", "checkpoint_dir", "resume_from"):
            expected.pop(key, None)
        if checkpoint.get("config") != expected or checkpoint.get("data_identity") != data_identity:
            raise ValueError("Resume config or ordered split data differs from checkpoint")
        if config.epochs < checkpoint["completed_epochs"]:
            raise ValueError("epochs cannot be lower than completed checkpoint epochs")
        from .artifacts import _dependency_versions, load_model_payload

        if checkpoint.get("dependencies") != _dependency_versions():
            raise ValueError("Strict resume requires matching runtime dependency versions")
        model = load_model_payload(checkpoint["current_model"], config.device)
    else:
        model = None
    if model is None:
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
        backbone = load_backbone(config)
        stats = ys.astype(np.float64)
        mean = np.nanmean(stats, axis=0) if config.task == "regression" else np.zeros(ys.shape[1])
        scale = np.nanstd(stats, axis=0) if config.task == "regression" else np.ones(ys.shape[1])
        if not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(np.abs(mean) > np.finfo(np.float32).max) or np.any(scale > np.finfo(np.float32).max):
            raise ValueError("Regression target statistics are not representable as float32")
        scale = np.where(scale == 0, 1, scale)
        mean32 = mean.astype(np.float32)
        scale32 = scale.astype(np.float32)
        if not np.isfinite(mean32).all() or not np.isfinite(scale32).all() or np.any(scale32 <= 0):
            raise ValueError("Regression target statistics are not representable as finite float32 with positive scales")
        model = FineTunedModel(
            backbone, config.head, config.task, ys.shape[1], mean32, scale32, config.train_backbone
        ).to(config.device)
    model.training_config = config
    if config.backbone_lr is not None and (
        not np.isfinite(config.backbone_lr) or config.backbone_lr <= 0
    ):
        raise ValueError("backbone_lr must be finite and positive")
    groups = []
    if config.train_backbone:
        groups.append(
            {
                "params": list(model.backbone.parameters()),
                **({"lr": config.backbone_lr} if config.backbone_lr is not None else {}),
            }
        )
    groups.append({"params": list(model.head.parameters())})
    optimizer = make_optimizer(config.optimizer.name, config.optimizer.kwargs, groups)
    optimizer_names = [
        [name for name, parameter in model.named_parameters() if any(parameter is p for p in group["params"])]
        for group in groups
    ]
    best = float("inf")
    best_state = None
    stale = 0
    history = []
    start_epoch = 0
    global_step = 0
    best_epoch = 0
    stopped = False
    if checkpoint is not None:
        if checkpoint.get("torch_version") != str(torch.__version__):
            raise ValueError("Strict resume requires the same PyTorch version")
        saved_cuda_count = checkpoint.get("cuda_device_count")
        current_cuda_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if type(saved_cuda_count) is not int or saved_cuda_count != current_cuda_count:
            raise ValueError("Strict CUDA resume requires the same device topology")
        start_epoch = checkpoint["completed_epochs"]
        global_step = checkpoint.get("global_step", 0)
        if config.epochs < start_epoch:
            raise ValueError("epochs cannot be lower than completed checkpoint epochs")
        if checkpoint["optimizer_names"] != optimizer_names:
            raise ValueError("Optimizer parameter ordering differs from checkpoint")
        model.load_state_dict(checkpoint["current_state"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        best = checkpoint["best_loss"]
        best_state = checkpoint["best_state"]
        best_epoch = checkpoint.get("best_epoch", 0)
        stale = checkpoint["stale"]
        history = checkpoint["history"]
        stopped = checkpoint.get("stopped", False)
        torch.set_rng_state(checkpoint["torch_rng"])
        if "python_rng" in checkpoint:
            random.setstate(tuple(checkpoint["python_rng"]))
        if torch.cuda.is_available() and checkpoint.get("cuda_rng") is not None:
            if len(checkpoint["cuda_rng"]) != torch.cuda.device_count():
                raise ValueError("Strict CUDA resume requires the same device count")
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        nstate = checkpoint["numpy_rng"]
        np.random.set_state(
            (nstate[0], np.asarray(nstate[1], dtype=np.uint32), nstate[2], nstate[3], nstate[4])
        )
    model.training_config = config
    batch_norm = config.head.normalization == "batch" and bool(config.head.hidden_dims)
    if batch_norm and (len(xs) < 2 or config.batch_size < 2):
        raise ValueError(
            "BatchNorm training requires at least two training rows and batch_size >= 2"
        )
    for epoch in range(start_epoch, config.epochs if not stopped else start_epoch):
        model.train()
        order = np.random.default_rng(config.seed + epoch).permutation(len(xs))
        batches = [
            order[start : start + config.batch_size]
            for start in range(0, len(order), config.batch_size)
        ]
        if batch_norm and len(batches) > 1 and len(batches[-1]) == 1:
            batches[-2] = np.concatenate((batches[-2], batches[-1]))
            batches.pop()
        total = count = 0
        for idx in batches:
            target = torch.as_tensor(ys[idx], device=config.device)
            mask = torch.isfinite(target)
            optimizer.zero_grad()
            if torch.is_inference_mode_enabled():
                raise RuntimeError("Training cannot run inside torch.inference_mode()")
            with torch.enable_grad():
                pred = model([xs[i] for i in idx])
                if config.task == "regression":
                    loss = (
                        (pred - (target - model.target_mean) / model.target_scale)[mask] ** 2
                    ).mean()
                else:
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        pred[mask], target[mask]
                    )
                if not torch.isfinite(pred).all() or not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite training output/loss at epoch {epoch + 1}, batch {global_step + 1}")
                loss.backward()
                for name, parameter in model.named_parameters():
                    if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                        raise FloatingPointError(f"Nonfinite gradient for {name} at epoch {epoch + 1}, batch {global_step + 1}")
                optimizer.step()
                from .artifacts import _validate_data
                try:
                    _validate_data(optimizer.state_dict(), "optimizer_state")
                except (TypeError, ValueError) as exc:
                    raise FloatingPointError(
                        f"Nonfinite optimizer state at epoch {epoch + 1}, batch {global_step + 1}: {exc}"
                    ) from exc
                for name, parameter in model.named_parameters():
                    if not torch.isfinite(parameter).all():
                        raise FloatingPointError(f"Nonfinite parameter {name} at epoch {epoch + 1}, batch {global_step + 1}")
            global_step += 1
            total += loss.item() * int(mask.sum())
            count += int(mask.sum())
        model.eval()
        totalv = countv = 0
        with torch.no_grad():
            for start in range(0, len(xv), config.batch_size):
                target = torch.as_tensor(
                    yv[start : start + config.batch_size], device=config.device
                )
                mask = torch.isfinite(target)
                pred = model(xv[start : start + config.batch_size])
                loss = (
                    ((pred - (target - model.target_mean) / model.target_scale)[mask] ** 2).mean()
                    if config.task == "regression"
                    else torch.nn.functional.binary_cross_entropy_with_logits(
                        pred[mask], target[mask]
                    )
                )
                totalv += loss.item() * int(mask.sum())
                countv += int(mask.sum())
        val = totalv / countv
        if not np.isfinite(val):
            raise FloatingPointError(f"Nonfinite validation loss at epoch {epoch + 1}")
        history.append({"epoch": epoch + 1, "train_loss": total / count, "validation_loss": val})
        if val < best - (config.early_stopping.min_delta if config.early_stopping else 0):
            best = val
            best_epoch = epoch + 1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stale = 0
            if config.checkpoint_dir:
                from .artifacts import save_model

                save_model(model, Path(config.checkpoint_dir) / "best.pt")
        else:
            stale += 1
        should_stop = bool(config.early_stopping and stale >= config.early_stopping.patience)
        if config.checkpoint_dir:
            checkpoint_dir = Path(config.checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            config_data = config.canonical()
            for key in ("epochs", "device", "checkpoint_dir", "resume_from"):
                config_data.pop(key, None)
            from .artifacts import _dependency_versions, atomic_save

            payload = {
                "schema": 1,
                "kind": "training_checkpoint",
                "config": config_data,
                "current_model": _checkpoint_model_payload(model),
                "data_identity": data_identity,
                "completed_epochs": epoch + 1,
                "current_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "best_state": {k: v.detach().cpu() for k, v in best_state.items()},
                "optimizer_state": optimizer.state_dict(),
                "best_loss": best,
                "best_epoch": best_epoch,
                "stale": stale,
                "stopped": should_stop,
                "history": history,
                "torch_rng": torch.get_rng_state(),
                "python_rng": list(random.getstate()),
                "torch_version": str(torch.__version__),
                "dependencies": _dependency_versions(),
                "optimizer_names": optimizer_names,
                "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "global_step": global_step,
                "numpy_rng": (
                    np.random.get_state()[0],
                    np.random.get_state()[1].tolist(),
                    np.random.get_state()[2],
                    np.random.get_state()[3],
                    np.random.get_state()[4],
                ),
            }
            atomic_save(payload, checkpoint_dir / "latest.pt")
        if should_stop:
            break
    if best_state is None:
        raise ValueError("No finite validation loss was observed")
    model.load_state_dict(best_state)
    model.history = history
    model.training_config = config
    model.eval()
    return model
