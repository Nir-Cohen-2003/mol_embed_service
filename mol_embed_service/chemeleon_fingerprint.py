"""CheMeleon fingerprint module using the official ChemProp 2.2+ API."""

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

    Automatically downloads model weights from Zenodo on first use.
    """

    _CKPT_URL = "https://zenodo.org/records/15460715/files/chemeleon_mp.pt?download=1"

    def __init__(
        self,
        checkpoint_path: Union[str, Path, None] = None,
        device: Union[str, torch.device, None] = None,
        *,
        normalize: bool = False,
    ):
        self._norm = normalize

        ckpt = _get_or_download(
            checkpoint_path or Path.home() / ".chemprop" / "chemeleon_mp.pt",
            self._CKPT_URL,
        )
        ckpt_data = torch.load(ckpt, map_location="cpu", weights_only=True)

        self.featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
        agg = nn.MeanAggregation()
        mp = nn.BondMessagePassing(**ckpt_data["hyper_parameters"])
        mp.load_state_dict(ckpt_data["state_dict"])

        self.model = MPNN(
            message_passing=mp,
            agg=agg,
            predictor=RegressionFFN(input_dim=mp.output_dim),  # not actually used
        )
        self.model.eval()

        if device is not None:
            self.model.to(device=device)

    @property
    def device(self) -> torch.device:
        """Return the device the model is on."""
        return next(self.model.parameters()).device

    @torch.inference_mode()
    def __call__(self, inputs: Sequence[Union[str, Mol]]) -> np.ndarray:
        """Generate CheMeleon fingerprints for a list of SMILES strings or RDKit Mols.

        Args:
            inputs: Sequence of SMILES strings or RDKit Mol objects.

        Returns:
            np.ndarray of shape (len(inputs), 2048).
        """
        graphs = [
            self.featurizer(MolFromSmiles(m) if isinstance(m, str) else m)
            for m in inputs
        ]
        bmg = BatchMolGraph(graphs)
        bmg.to(device=self.device)

        emb = self.model.fingerprint(bmg).numpy(force=True)

        if self._norm:
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            emb = emb / np.where(norms == 0, 1, norms)

        return emb
