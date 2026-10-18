"""Minimal CDDD SMILES preprocessing and ONNX input-format adapter."""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.SaltRemover import SaltRemover

# These IDs define the encoder's fixed serialized input vocabulary.
VOCABULARY = {
    "#": 1, "%": 2, "(": 4, ")": 3, "+": 5, "-": 6,
    "0": 8, "1": 7, "2": 10, "3": 9, "4": 12, "5": 11,
    "6": 14, "7": 13, "8": 16, "9": 15, ":": 17,
    "</s>": 0, "<s>": 39, "=": 18, "@": 19, "B": 21,
    "Br": 38, "C": 20, "Cl": 37, "F": 22, "H": 24,
    "I": 23, "N": 26, "O": 25, "P": 27, "S": 28,
    "[": 29, "]": 30, "c": 31, "i": 32, "n": 34,
    "o": 33, "p": 35, "s": 36,
}
_TOKEN_PATTERN = re.compile(r"Cl|Br|[#%)(+\-1032547698:=@CBFIHONPS\[\]cionps]")
_ORGANIC_ATOMS = {5, 6, 7, 8, 9, 15, 16, 17, 35, 53}
_SALT_REMOVER = SaltRemover()


def preprocess_smiles(smiles: str) -> str | float:
    """Normalize and filter one SMILES according to the encoder input domain."""
    try:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            return float("nan")
        molecule = _SALT_REMOVER.StripMol(molecule, dontRemoveEverything=True)
        if molecule is None:
            return float("nan")
        normalized = Chem.MolToSmiles(molecule, isomericSmiles=False)
        if "." in normalized:
            fragments = Chem.GetMolFrags(Chem.MolFromSmiles(normalized), asMols=True)
            if not fragments:
                return float("nan")
            molecule = max(fragments, key=lambda fragment: fragment.GetNumAtoms())
            normalized = Chem.MolToSmiles(molecule)
        molecule = Chem.MolFromSmiles(normalized)
        if molecule is None:
            return float("nan")
        atoms = {atom.GetAtomicNum() for atom in molecule.GetAtoms()}
        if not atoms <= _ORGANIC_ATOMS:
            return float("nan")
        if not (-5 < Descriptors.MolLogP(molecule) < 7):
            return float("nan")
        if not (12 < Descriptors.MolWt(molecule) < 600):
            return float("nan")
        if not (3 < Descriptors.HeavyAtomCount(molecule) < 50):
            return float("nan")
        return Chem.MolToSmiles(molecule)
    except Exception:  # noqa: BLE001 - malformed chemistry is an invalid row
        return float("nan")


def tokenize(smiles: str) -> list[int]:
    """Encode recognized tokens, preserving the model's historical skip behavior."""
    return [VOCABULARY[token] for token in _TOKEN_PATTERN.findall(smiles)]


def make_inputs(smiles_batch: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Create right-padded int32 token IDs and int32 unpadded sequence lengths."""
    sequences = [np.asarray([39, *tokenize(smiles), 0], dtype=np.int32) for smiles in smiles_batch]
    lengths = np.asarray([len(sequence) for sequence in sequences], dtype=np.int32)
    width = int(lengths.max())
    inputs = np.zeros((len(sequences), width), dtype=np.int32)
    for row, sequence in enumerate(sequences):
        inputs[row, : len(sequence)] = sequence
    return inputs, lengths


@dataclass
class HyperParameters:
    batch_size: int = 128


class CDDDInferenceModel:
    """Run the established token format through an explicitly configured ORT session."""

    def __init__(self, session):
        self.encoder_session = session
        self.hparams = HyperParameters()

    def seq_to_emb(self, smiles_list: list[str], batch_size: int | None = None) -> np.ndarray:
        size = batch_size or self.hparams.batch_size
        if size <= 0:
            raise ValueError(f"batch_size must be positive, got {size}")
        processed = [preprocess_smiles(smiles) for smiles in smiles_list]
        valid = [smiles for smiles in processed if not pd.isna(smiles)]
        if not valid:
            raise ValueError("No valid SMILES found after preprocessing")
        output = []
        for start in range(0, len(valid), size):
            tokens, lengths = make_inputs(valid[start : start + size])
            output.extend(self.encoder_session.run(None, {
                "Input/Placeholder:0": tokens,
                "Input/Placeholder_1:0": lengths,
            })[0])
        return np.asarray(output)
