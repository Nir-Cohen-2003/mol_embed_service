"""CheMeleon fingerprint module using the official ChemProp 2.2+ API."""

import hashlib
import os
from pathlib import Path
from typing import Sequence, Union
from urllib.request import urlretrieve
import numpy as np
import torch
from chemprop import featurizers, nn
from chemprop.data import BatchMolGraph
from chemprop.models import MPNN
from chemprop.nn import RegressionFFN
from rdkit.Chem import Mol, MolFromSmiles

CHEMELEON_TARGET_SEMANTIC_VERSION = "chemeleon-mean-aggregation-v1"
CHEMELEON_SERVICE_VERSION = "0.1.0"


def resolve_checkpoint_provenance(checkpoint_path: Union[str, Path]) -> dict[str, str]:
    """Return explicit checkpoint path and content provenance for target caches."""
    if not isinstance(checkpoint_path, (str, Path)) or not str(checkpoint_path):
        raise ValueError(f"CheMeleon checkpoint_path must be explicit and non-empty, got {checkpoint_path!r}")
    resolved = Path(checkpoint_path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"CheMeleon checkpoint does not exist: configured={checkpoint_path!r}, resolved={resolved}")
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "checkpoint_path": str(resolved),
        "checkpoint_sha256": digest.hexdigest(),
        "semantic_version": CHEMELEON_TARGET_SEMANTIC_VERSION,
    }


def _get_or_download(path: Union[str, Path], url: str) -> Path:
    """Return the path to a file, downloading it if necessary."""
    path = Path(path)
    if path.is_dir():
        path = path / Path(url).name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(url, path)
    return path


class CheMeleonFingerprint:
    """CheMeleon learned fingerprint generator.

    The legacy no-argument path retains generic service behavior. Automatic
    training targets must pass an explicit checkpoint through the target API.
    """

    _CKPT_URL = "https://zenodo.org/records/15460715/files/chemeleon_mp.pt?download=1"

    def __init__(self, checkpoint_path: Union[str, Path, None] = None, device: Union[str, torch.device, None] = None, *, normalize: bool = False):
        self._norm = normalize
        ckpt = _get_or_download(checkpoint_path or Path.home() / ".chemprop" / "chemeleon_mp.pt", self._CKPT_URL)
        self.checkpoint_path = ckpt.resolve()
        self.provenance = resolve_checkpoint_provenance(self.checkpoint_path)
        ckpt_data = torch.load(ckpt, map_location="cpu", weights_only=True)
        self.featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
        agg = nn.MeanAggregation()
        mp = nn.BondMessagePassing(**ckpt_data["hyper_parameters"])
        mp.load_state_dict(ckpt_data["state_dict"])
        self.model = MPNN(message_passing=mp, agg=agg, predictor=RegressionFFN(input_dim=mp.output_dim))
        self.model.eval()
        if device is not None:
            self.model.to(device=device)

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @torch.inference_mode()
    def __call__(self, inputs: Sequence[Union[str, Mol]]) -> np.ndarray:
        graphs = [self.featurizer(MolFromSmiles(m) if isinstance(m, str) else m) for m in inputs]
        bmg = BatchMolGraph(graphs)
        bmg.to(device=self.device)
        emb = self.model.fingerprint(bmg).numpy(force=True).astype(np.float32, copy=False)
        if self._norm:
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            emb = emb / np.where(norms == 0, 1, norms)
        return emb

    @property
    def target_contract(self) -> dict[str, object]:
        return {"dimension": 2048, "dtype": "float32", "pooling": "mean", "normalization": "none"}
