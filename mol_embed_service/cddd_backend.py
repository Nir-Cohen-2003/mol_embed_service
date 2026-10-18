"""Strict ONNX Runtime backend for CDDD embeddings."""

from __future__ import annotations

import numpy as np
import torch

from .cddd_inference import CDDDInferenceModel, preprocess_smiles


class CDDDBackend:
    """Own one explicitly configured CDDD ONNX Runtime encoder session."""

    def __init__(self, device, session_options_factory=None):
        import onnxruntime as ort

        from .cddd_checkpoint import get_model_path

        self.device = torch.device(device)
        if self.device.type not in {"cpu", "cuda"}:
            raise ValueError(f"CDDD supports only CPU or CUDA devices, got {self.device}")
        available = ort.get_available_providers()
        if self.device.type == "cuda" and "CUDAExecutionProvider" not in available:
            raise RuntimeError(
                "CDDD CUDA was requested but ONNX Runtime CUDAExecutionProvider is unavailable; "
                f"available providers: {available}. Install a compatible CUDA-enabled ONNX Runtime."
            )

        options = (session_options_factory or ort.SessionOptions)()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        provider = "CPUExecutionProvider"
        providers = [provider]
        if self.device.type == "cuda":
            provider = (
                "CUDAExecutionProvider",
                {"device_id": self.device.index or 0, "use_tf32": "0"},
            )
            providers = [provider, "CPUExecutionProvider"]
        try:
            session = ort.InferenceSession(
                get_model_path("encoder"), sess_options=options, providers=providers
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to create CDDD {self.device} ONNX Runtime session with providers {providers}"
            ) from exc
        actual = session.get_providers()
        if self.device.type == "cuda" and "CUDAExecutionProvider" not in actual:
            raise RuntimeError(
                "CDDD CUDA was requested, but the created ONNX Runtime session did not attach "
                f"CUDAExecutionProvider (session providers: {actual})"
            )
        if self.device.type == "cpu" and actual != ["CPUExecutionProvider"]:
            raise RuntimeError(f"CDDD CPU session unexpectedly attached providers {actual}")
        if self.device.type == "cuda":
            # ORT otherwise retries a failed CUDA run with a recreated CPU session.
            session.disable_fallback()
        self.model = CDDDInferenceModel(session)

    def embed(self, smiles_list: list[str], batch_size: int) -> np.ndarray:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        result = np.zeros((len(smiles_list), 512), dtype=np.float32)
        if not smiles_list:
            return result
        valid = []
        valid_indices = []
        for index, smiles in enumerate(smiles_list):
            if not smiles:
                continue
            try:
                processed = preprocess_smiles(smiles)
            except Exception:  # noqa: BLE001, S112 - malformed inputs are rejected
                continue
            if not (isinstance(processed, float) and np.isnan(processed)):
                valid.append(smiles)
                valid_indices.append(index)
        dummy = "CN1C=NC2=C1C(=O)N(C)C(=O)N2C"
        for start in range(0, len(valid), batch_size):
            rows = valid[start : start + batch_size]
            try:
                output = np.asarray(self.model.seq_to_emb([dummy] + rows)[1:])
            except Exception as exc:
                raise RuntimeError(f"CDDD inference failed for rows {valid_indices[start:start + batch_size]}") from exc
            expected = (len(rows), 512)
            if output.shape != expected:
                raise ValueError(f"CDDD returned shape {output.shape}; expected {expected}")
            if not np.isfinite(output).all():
                raise ValueError(f"CDDD returned nonfinite values for rows {valid_indices[start:start + batch_size]}")
            result[valid_indices[start : start + batch_size]] = output
        return result
