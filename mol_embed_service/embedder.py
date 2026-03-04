"""Main embedding API implementation."""

from typing import List, Literal
from pathlib import Path
import numpy as np
from .models import ChemBERTaEmbedder, CDDDEmbedder, ChemformerEmbedder


ModelType = Literal[
    "chemberta-v1",
    "chemberta-v2",
    "chemberta-v3",
    "cddd",
    "chemformer"
]


def embed_smiles(
    smiles_list: List[str],
    model: ModelType,
    output_path: str,
    batch_size: int = 32,
    device: str = "cuda",
) -> None:
    """Generate molecular embeddings and save to .npy file.

    Args:
        smiles_list: List of SMILES strings to embed
        model: Model type to use for embedding
        output_path: Path to save embeddings (.npy file)
        batch_size: Batch size for inference (default: 32)
        device: Device to use ('cuda' or 'cpu', default: 'cuda')

    Raises:
        ValueError: If invalid model type or empty SMILES list
        FileNotFoundError: If output directory doesn't exist

    Example:
        >>> from mol_embed_service import embed_smiles
        >>> smiles = ["CCO", "c1ccccc1", "CC(=O)O"]
        >>> embed_smiles(smiles, "chemberta-v1", "embeddings.npy", batch_size=16)
    """
    if not smiles_list:
        raise ValueError("SMILES list cannot be empty")

    # Ensure output directory exists
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Select and initialize model
    print(f"Initializing {model} on {device}...")
    if model.startswith("chemberta"):
        embedder = ChemBERTaEmbedder(version=model, device=device)
    elif model == "cddd":
        embedder = CDDDEmbedder(device=device)
    elif model == "chemformer":
        embedder = ChemformerEmbedder(device=device)
    else:
        raise ValueError(
            f"Unknown model: {model}. "
            f"Must be one of: chemberta-v1, chemberta-v2, chemberta-v3, cddd, chemformer"
        )

    # Generate embeddings
    print(f"Generating embeddings (batch_size={batch_size})...")
    embeddings = embedder.embed(smiles_list, batch_size=batch_size)

    # Save to file
    np.save(output_file, embeddings)
    print(f"Embeddings saved to {output_file}")
    print(f"Shape: {embeddings.shape}, dtype: {embeddings.dtype}")


if __name__ == "__main__":
    # Example usage
    test_smiles = [
        "CCO",                          # Ethanol
        "c1ccccc1",                     # Benzene
        "CC(=O)O",                      # Acetic acid
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O" # Ibuprofen
    ]

    embed_smiles(
        test_smiles,
        model="chemberta-v1",
        output_path="test_embeddings.npy",
        batch_size=2
    )