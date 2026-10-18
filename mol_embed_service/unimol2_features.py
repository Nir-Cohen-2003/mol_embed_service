"""Deterministic retained-graph Uni-Mol2 input features.

Derived from DP Technology Uni-Mol at https://github.com/deepmodeling/Uni-Mol/tree/90f52c41299a1a582da0f9765e9f87aa21faa16a
Copyright (c) DP Technology. Licensed under the MIT License (see project attribution).
"""

import logging

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

_CHIRAL = [
    "CHI_UNSPECIFIED",
    "CHI_TETRAHEDRAL_CW",
    "CHI_TETRAHEDRAL_CCW",
    "CHI_TRIGONALBIPYRAMIDAL",
    "CHI_OCTAHEDRAL",
    "CHI_SQUAREPLANAR",
    "CHI_OTHER",
]
_DEGREE = list(range(11)) + ["misc"]
_CHARGE = list(range(-5, 6)) + ["misc"]
_H = list(range(9)) + ["misc"]
_RADICAL = list(range(5)) + ["misc"]
_HYBRID = ["SP", "SP2", "SP3", "SP3D", "SP3D2", "misc"]
_BOND = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC", "misc"]
_STEREO = ["STEREONONE", "STEREOZ", "STEREOE", "STEREOCIS", "STEREOTRANS", "STEREOANY"]


def _idx(values, item):
    try:
        return values.index(item)
    except ValueError:
        return len(values) - 1


def featurize_smiles(smiles):
    """Return token, retained graph features and centered 3D coordinates, or None if invalid/empty."""
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    mol = Chem.RemoveAllHs(mol)
    n = mol.GetNumAtoms()
    if n == 0:
        return None
    if n > 256:
        raise ValueError(f"molecule has {n} heavy atoms; maximum is 256")
    if any(a.GetAtomicNum() <= 0 or a.GetAtomicNum() >= 128 for a in mol.GetAtoms()):
        raise ValueError("unsupported atomic number for Uni-Mol2 token embedding")
    x = []
    for a in mol.GetAtoms():
        values = [
            _idx(_CHIRAL, str(a.GetChiralTag())),
            _idx(_DEGREE, a.GetTotalDegree()),
            _idx(_CHARGE, a.GetFormalCharge()),
            _idx(_H, a.GetTotalNumHs()),
            _idx(_RADICAL, a.GetNumRadicalElectrons()),
            _idx(_HYBRID, str(a.GetHybridization())),
            int(a.GetIsAromatic()),
            int(a.IsInRing()),
        ]
        x.append(values)
    x = np.asarray(x, dtype=np.int64)
    adj = np.zeros((n, n), dtype=np.int64)
    edge = np.zeros((n, n, 3), dtype=np.int64)
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        adj[i, j] = adj[j, i] = 1
        bf = [
            _idx(_BOND, str(b.GetBondType())),
            _idx(_STEREO, str(b.GetStereo())),
            int(b.GetIsConjugated()),
        ]
        edge[i, j] = edge[j, i] = bf
    atom_offsets = np.arange(8) * 16 + 1
    atom = x + atom_offsets
    edge_offsets = np.array([1, 17, 33])
    bonded = adj.astype(bool)
    edge[bonded] = edge[bonded] + edge_offsets + 1
    degree = adj.sum(-1)
    atom = atom + 2
    edge = edge + 2
    degree = degree + 2
    # Floyd-Warshall with active dataset's disconnected sentinel 509, then +1.
    dist = np.full((n, n), 509, dtype=np.int64)
    dist[adj.astype(bool)] = 1
    np.fill_diagonal(dist, 0)
    for k in range(n):
        dist = np.minimum(dist, dist[:, k, None] + dist[None, k, :])
    dist = np.minimum(dist, 509) + 1
    # Compatibility quirk: pair typing derives from encoded chirality column.
    chir = atom[:, 0]
    pair = np.stack(
        (np.broadcast_to(chir[:, None] + 1, (n, n)), np.broadcast_to(chir[None, :] + 129, (n, n))),
        axis=-1,
    )
    # ETKDGv3, fixed seed, one thread; add/remove H preserving heavy atom order.
    with_h = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    params.numThreads = 1
    if AllChem.EmbedMolecule(with_h, params) != 0:
        raise ValueError("ETKDGv3 conformer generation failed")
    if AllChem.MMFFHasAllMoleculeParams(with_h):
        status = AllChem.MMFFOptimizeMolecule(with_h, mmffVariant="MMFF94", maxIters=200)
        if status == 1:
            logging.getLogger(__name__).warning(
                "MMFF94 optimization did not converge; retaining finite ETKDGv3 geometry"
            )
        elif status != 0:
            raise ValueError(f"MMFF94 optimization failed with code {status}")
    xyz = np.asarray(with_h.GetConformer().GetPositions()[:n], dtype=np.float32)
    if xyz.shape != (n, 3) or not np.isfinite(xyz).all():
        raise ValueError("invalid conformer coordinates")
    xyz -= xyz.mean(0, keepdims=True)
    return {
        "token": np.asarray([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=np.int64),
        "atom": atom,
        "edge": edge,
        "degree": degree,
        "path": dist,
        "pair_type": pair,
        "xyz": xyz,
    }


def collate(items, device="cpu"):
    n = len(items)
    lengths = [len(x["token"]) for x in items]
    L = ((max(lengths) + 4) // 4) * 4
    out = {
        "src_token": torch.zeros((n, L - 1), dtype=torch.long, device=device),
        "atom_feat": torch.zeros((n, L - 1, 8), dtype=torch.long, device=device),
        "degree": torch.zeros((n, L - 1), dtype=torch.long, device=device),
        "edge_feat": torch.zeros((n, L - 1, L - 1, 3), dtype=torch.long, device=device),
        "shortest_path": torch.zeros((n, L - 1, L - 1), dtype=torch.long, device=device),
        "pair_type": torch.zeros((n, L - 1, L - 1, 2), dtype=torch.long, device=device),
        "src_pos": torch.zeros((n, L - 1, 3), dtype=torch.float32, device=device),
        "atom_mask": torch.zeros((n, L - 1), dtype=torch.long, device=device),
        "attn_bias": torch.full((n, L, L), float("-inf"), device=device),
    }
    for b, x in enumerate(items):
        m = len(x["token"])
        out["src_token"][b, :m] = torch.as_tensor(x["token"], device=device)
        out["atom_feat"][b, :m] = torch.as_tensor(x["atom"], device=device)
        out["degree"][b, :m] = torch.as_tensor(x["degree"], device=device)
        out["edge_feat"][b, :m, :m] = torch.as_tensor(x["edge"], device=device)
        out["shortest_path"][b, :m, :m] = torch.as_tensor(x["path"], device=device)
        out["pair_type"][b, :m, :m] = torch.as_tensor(x["pair_type"], device=device)
        out["src_pos"][b, :m] = torch.as_tensor(x["xyz"], device=device)
        out["atom_mask"][b, :m] = 1
        out["attn_bias"][b, : m + 1, : m + 1] = 0
        out["attn_bias"][b, m + 1 :, : m + 1] = 0
    return out
