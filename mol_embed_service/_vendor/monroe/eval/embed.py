"""Embed SMILES with a pretrained Monroe encoder.

The entry point is :func:`embed_smiles`: SMILES in, one 720-d vector per molecule out.
Featurization (conformer generation, which dominates the cost) runs across a process
pool; the encoder forward pass is a batched GPU call on top.

    from mol_embed_service._vendor.monroe.model.ckpt import load_ckpt
    from mol_embed_service._vendor.monroe.eval.embed import embed_smiles

    encoder = load_ckpt("checkpoint")
    embeddings = embed_smiles(["CCO", "c1ccccc1"], encoder)
"""

import numpy as np
import torch
from torch_geometric.data import Batch, Data

from mol_embed_service._vendor.monroe.preprocessing import build_eval_graphs as _graphs


def _featurize_one(smiles: str) -> tuple[str, dict] | None:
    """Graph for one molecule, or None if it cannot be built."""
    try:
        graph = _graphs.build_mol(smiles)
        if graph is None:
            graph = _graphs._build_from_smiles_fallback(smiles)
    except Exception:
        try:
            graph = _graphs._build_from_smiles_fallback(smiles)
        except Exception:
            return None
    return (smiles, graph) if graph is not None else None


def featurize_smiles(smiles: list[str], n_workers: int | None = None) -> list[tuple[str, dict]]:
    """Build graphs serially; only the service-supported n_workers=1 is accepted."""
    if n_workers not in (None, 1):
        raise ValueError("Bundled Monroe inference supports n_workers=1 only")
    return [graph for value in smiles if (graph := _featurize_one(value)) is not None]


def to_pyg(graph: dict) -> Data:
    """Graph dict from the featurizer -> the Data object the encoder expects."""
    return Data(
        x=torch.as_tensor(graph["node_float"], dtype=torch.float32),
        node_codes=torch.as_tensor(graph["node_codes"], dtype=torch.long),
        edge_index=torch.as_tensor(graph["edge_index"], dtype=torch.long),
        edge_codes=torch.as_tensor(graph["edge_codes"], dtype=torch.long),
        pos_in=torch.as_tensor(np.asarray(graph["pos_rdkit"], dtype=np.float32)),
    )


def embed_smiles(
    smiles: list[str],
    encoder: torch.nn.Module,
    device: torch.device | str | None = None,
    batch_size: int = 32,
    n_workers: int | None = None,
) -> dict[str, np.ndarray]:
    """SMILES -> {smiles: graph-level embedding}.

    Molecules the featurizer cannot handle are left out of the returned dict, so
    always look up by SMILES rather than assuming the input order is preserved.
    """
    if device is None:
        device = next(encoder.parameters()).device
    encoder = encoder.to(device).eval()

    built = featurize_smiles(smiles, n_workers=n_workers)
    if not built:
        return {}

    graphs = [to_pyg(graph) for _, graph in built]
    vectors = []
    with torch.no_grad():
        for start in range(0, len(graphs), batch_size):
            batch = Batch.from_data_list(graphs[start:start + batch_size]).to(device)
            graph_embedding, _ = encoder(batch)
            vectors.append(graph_embedding.cpu().numpy())

    return dict(zip([smi for smi, _ in built], np.concatenate(vectors)))
