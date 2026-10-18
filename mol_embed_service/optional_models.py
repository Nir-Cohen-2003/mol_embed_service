"""Lazy adapters for optional molecular embedding backends."""

import importlib
import os
import threading
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem

from .models import BaseEmbedder

_MONROE_DIM = 720
_MINIMOL_DIM = 512
_GRASP_DIM = 768
_GRASP_MODEL = "caithmac/GRASP"
_GRASP_REVISION = "4804c0a558307c944da5e8de1d182b4b4c994dae"

_minimol_lock = threading.RLock()
_minimol_backend = None


def _validate_inputs(model: str, smiles_list: list[str], batch_size: int) -> list[str]:
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError(f"{model} batch_size must be a positive integer, got {batch_size!r}")
    if isinstance(smiles_list, (str, bytes)):
        raise TypeError(f"{model} input must be a sequence of SMILES, not a bare string")
    try:
        smiles = list(smiles_list)
    except TypeError as exc:
        raise ValueError(f"{model} input must be a sequence of SMILES") from exc
    if not smiles:
        raise ValueError(f"{model} input cannot be empty")
    invalid = []
    for index, value in enumerate(smiles):
        molecule = Chem.MolFromSmiles(value) if isinstance(value, str) and value else None
        if molecule is None or molecule.GetNumAtoms() == 0:
            invalid.append(index)
    if invalid:
        raise ValueError(f"{model} invalid or empty SMILES at row indices {invalid}")
    return smiles


def _validated_matrix(model: str, result, rows: int, width: int) -> np.ndarray:
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            array = np.asarray(result, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{model} returned embeddings that cannot be converted to float32") from exc
    if array.shape != (rows, width):
        raise ValueError(f"{model} returned shape {array.shape}; expected {(rows, width)}")
    if not np.isfinite(array).all():
        raise ValueError(f"{model} returned non-finite float32 embeddings")
    return np.ascontiguousarray(array, dtype=np.float32)


def _dependency_error(name: str, install: str) -> ImportError:
    return ImportError(f"{name} is optional; install it using the instructions in the README ({install}).")


class MonroeEmbedder(BaseEmbedder):
    """Monroe encoder backed by an explicitly configured local checkpoint."""

    def __init__(self, device: str = "cuda", *, checkpoint_dir: str | Path | None = None):
        super().__init__(device)
        configured = checkpoint_dir if checkpoint_dir is not None else os.environ.get("MONROE_CHECKPOINT_DIR")
        if configured is None:
            raise ValueError("Monroe requires checkpoint_dir or MONROE_CHECKPOINT_DIR; see README checkpoint instructions")
        self.checkpoint_dir = Path(configured).expanduser()
        for filename in ("config.json", "weights.pt"):
            path = self.checkpoint_dir / filename
            if not path.is_file() or not os.access(path, os.R_OK) or path.stat().st_size == 0:
                raise FileNotFoundError(f"Monroe checkpoint requires readable, nonempty {path}")
            if filename == "weights.pt":
                with path.open("rb") as checkpoint:
                    if checkpoint.read(120).startswith(b"version https://git-lfs.github.com/spec/v1"):
                        raise ValueError(f"Monroe checkpoint {path} is a Git LFS pointer; fetch the actual LFS weight file")
        try:
            embed_module = importlib.import_module("monroe.eval.embed")
            checkpoint_module = importlib.import_module("monroe.model.ckpt")
        except ImportError as exc:
            raise _dependency_error("Monroe", "the pinned isolated installation") from exc
        self._embed_smiles = embed_module.embed_smiles
        self.encoder = checkpoint_module.load_ckpt(str(self.checkpoint_dir))

    def embed(self, smiles_list: list[str], batch_size: int) -> np.ndarray:
        smiles = _validate_inputs("Monroe", smiles_list, batch_size)
        result = self._embed_smiles(smiles, self.encoder, device=str(self.device), batch_size=batch_size, n_workers=1)
        rows = []
        for index, value in enumerate(smiles):
            if value not in result:
                raise ValueError(f"Monroe missing embedding for row {index} ({value!r})")
            vector = np.asarray(result[value])
            if vector.shape != (_MONROE_DIM,):
                raise ValueError(f"Monroe malformed embedding at row {index}: shape {vector.shape}, expected ({_MONROE_DIM},)")
            rows.append(vector)
        return _validated_matrix("Monroe", rows, len(smiles), _MONROE_DIM)


class MiniMolEmbedder(BaseEmbedder):
    """MiniMol wrapper with a conservative CUDA-hidden CPU support boundary."""

    def __init__(self, device: str = "cuda"):
        if device not in ("cpu", "cuda"):
            raise ValueError(f"MiniMol supports only unindexed 'cpu' or 'cuda' requests, got {device!r}")
        if torch.cuda.is_available():
            raise ValueError("MiniMol is unsupported when CUDA is visible: hide CUDA before process startup (for example CUDA_VISIBLE_DEVICES='')")
        super().__init__(device)

    def embed(self, smiles_list: list[str], batch_size: int) -> np.ndarray:
        global _minimol_backend
        smiles = _validate_inputs("MiniMol", smiles_list, batch_size)
        if torch.cuda.is_available():
            raise ValueError("MiniMol is unsupported when CUDA is visible: hide CUDA before process startup")
        with _minimol_lock:
            if torch.cuda.is_available():
                raise ValueError("MiniMol is unsupported when CUDA is visible: hide CUDA before process startup")
            if _minimol_backend is None:
                try:
                    minimol_module = importlib.import_module("minimol")
                except ImportError as exc:
                    raise _dependency_error("MiniMol", "the pinned isolated installation") from exc
                _minimol_backend = minimol_module.Minimol(batch_size=100)
            chunks = []
            with torch.no_grad():
                for start in range(0, len(smiles), batch_size):
                    input_chunk = smiles[start:start + batch_size]
                    values = _minimol_backend(input_chunk)
                    try:
                        chunk = torch.stack(list(values)).detach().cpu().numpy()
                        chunk = _validated_matrix("MiniMol", chunk, len(input_chunk), _MINIMOL_DIM)
                    except (TypeError, RuntimeError, ValueError) as exc:
                        raise ValueError(
                            f"MiniMol returned malformed output for input chunk starting at {start}: {exc}"
                        ) from exc
                    chunks.append(chunk)
            result = np.concatenate(chunks, axis=0)
        return _validated_matrix("MiniMol", result, len(smiles), _MINIMOL_DIM)


class GRASPEmbedder(BaseEmbedder):
    """GRASP final-layer CLS encoder."""

    def __init__(self, device: str = "cuda"):
        super().__init__(device)
        try:
            grasp_module = importlib.import_module("grasp")
        except ImportError as exc:
            raise _dependency_error("GRASP", "the pinned isolated installation") from exc
        self.encoder = grasp_module.GRASPEncoder.from_pretrained(
            _GRASP_MODEL, device=str(self.device), revision=_GRASP_REVISION
        )

    def embed(self, smiles_list: list[str], batch_size: int) -> np.ndarray:
        smiles = _validate_inputs("GRASP", smiles_list, batch_size)
        result = self.encoder.encode(smiles, batch_size=batch_size)
        return _validated_matrix("GRASP", result, len(smiles), _GRASP_DIM)
