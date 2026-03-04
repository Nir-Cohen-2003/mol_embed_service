"""Molecular Embedding Service - GPU-accelerated embedding generation."""

from .embedder import embed_smiles

__version__ = "0.1.0"
__all__ = ["embed_smiles"]
