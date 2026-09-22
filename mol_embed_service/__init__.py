"""Molecular Embedding Service - GPU-accelerated embedding generation."""

from .embedder import (
    embed_smiles,
    get_embedding_size,
    get_chemeleon_provenance,
    chemeleon_target_array,
    EMBEDDING_SIZES,
    ModelType,
    EmbeddingDim,
)

__version__ = "0.1.0"
__all__ = [
    "embed_smiles", "get_embedding_size", "get_chemeleon_provenance",
    "chemeleon_target_array", "EMBEDDING_SIZES", "ModelType", "EmbeddingDim",
]
