"""Inference-only Monroe SMILES graph construction (serial, no process-global alarms)."""
import datamol as dm
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm
from mol_embed_service._vendor.monroe.model.featurizer import build_from_mol_and_coords, build_single_graph

def _build_from_smiles_fallback(smi: str) -> dict[str, np.ndarray] | None:
    """Build molecular graph directly from SMILES, bypassing InChI round-trip.

    Used as a fallback when build_single_graph fails (e.g., metal-oxygen bonds
    lost during InChI conversion + rdDetermineBonds reconstruction).
    """
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        mol = Chem.AddHs(mol, addCoords=False)
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
        # Try 3D embedding
        try:
            from mol_embed_service._vendor.monroe.model.featurizer import predict_structure
            conf = predict_structure(mol)
            pos_rdkit = np.asarray(conf.GetPositions(), dtype=np.float32)
        except Exception:
            AllChem.Compute2DCoords(mol)
            conf = mol.GetConformer(0)
            pos_rdkit = np.asarray(conf.GetPositions(), dtype=np.float32)
        return build_from_mol_and_coords(mol, pos_rdkit=pos_rdkit, stereo_augmentation=True, symmetrize=True)
    except Exception as exc:
        tqdm.write(f"[warn] SMILES fallback also failed for {smi}: {exc}")
        return None


def build_mol(smi: str) -> dict[str, np.ndarray] | None:
    """Build molecular graph from SMILES."""
    try:
        try:
            inchi = dm.to_inchi(smi)
        except Exception as exc:
            tqdm.write(f"[warn] Failed to convert SMILES to InChI for {smi}: {exc}")
            mol = Chem.MolFromSmiles(smi, sanitize=False)
            Chem.SanitizeMol(mol, sanitizeOps=Chem.SANITIZE_PROPERTIES)
            Chem.SanitizeMol(
                mol,
                sanitizeOps=Chem.SANITIZE_SYMMRINGS
                | Chem.SANITIZE_CLEANUP
                | Chem.SANITIZE_ADJUSTHS,
            )
            Chem.Kekulize(mol, clearAromaticFlags=True)
            Chem.SanitizeMol(
                mol, sanitizeOps=Chem.SANITIZE_SETAROMATICITY | Chem.SANITIZE_KEKULIZE
            )
            Chem.SetAromaticity(mol, Chem.AromaticityModel.AROMATICITY_DEFAULT)
            Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
            inchi = Chem.MolToInchi(mol)

        # Serial service path intentionally avoids process-global signal handlers.
        mol_graph = build_single_graph(
            inchi=inchi,
            atomic_numbers=None,
            coords=None,
            charge=None,
            stereo_augmentation=True,
            allow_2d_fallback_on_timeout=True,
            symmetrize=True,
        )

        # Fallback: if InChI round-trip lost all bonds (e.g., metal compounds),
        # rebuild directly from SMILES which preserves the original bond information
        if mol_graph is not None and mol_graph["edge_index"].shape[1] == 0:
            tqdm.write(f"[warn] InChI round-trip lost bonds for {smi[:50]}, trying SMILES fallback")
            mol_graph = _build_from_smiles_fallback(smi)

    except Exception as exc:
        tqdm.write(f"[warn] Failed to build graph for {smi}: {exc}")
        mol_graph = _build_from_smiles_fallback(smi)
    return mol_graph
