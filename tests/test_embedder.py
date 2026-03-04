import os

import numpy as np
import pytest
import torch

from mol_embed_service.embedder import embed_smiles

SAMPLE_SMILES = [
    "CCO",  # Ethanol
    "c1ccccc1",  # Benzene
    "CC(=O)O",  # Acetic acid
    "CC(C)CC1=CC=C(C=C1)C[C@H](C)C(=O)O",  # Ibuprofen (Canonical)
    "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",  # Caffeine (Canonical)
]


@pytest.mark.parametrize(
    "model, expected_dim",
    [
        ("chemberta-v1", 768),
        ("chemberta-v2", 384),
        ("chemberta-v3", 384),
        ("cddd", 512),
        ("chemformer", 384),
    ],
)
def test_all_models_embedding_and_saving(model, expected_dim, tmp_path):
    output_path = tmp_path / f"test_{model}.npy"

    # We use CPU for tests to ensure they run in CI environments without GPU
    # embed_smiles will automatically fallback to CPU if CUDA is not available
    embed_smiles(
        smiles_list=SAMPLE_SMILES,
        model=model,
        output_path=str(output_path),
        batch_size=2,
        device="cpu",
    )

    assert output_path.exists()
    embeddings = np.load(output_path)

    # Verify shape
    assert embeddings.shape == (len(SAMPLE_SMILES), expected_dim)

    # Verify it's a valid numpy array
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.dtype in [np.float32, np.float64]

    # All models must return valid numbers
    assert not np.isnan(embeddings).any(), f"NaNs found in {model} embeddings"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize(
    "model", ["chemberta-v1", "chemberta-v2", "chemberta-v3", "cddd", "chemformer"]
)
def test_gpu_embedding(model, tmp_path):
    output_path = tmp_path / f"test_gpu_{model}.npy"

    embed_smiles(
        smiles_list=SAMPLE_SMILES,
        model=model,
        output_path=str(output_path),
        batch_size=2,
        device="cuda",
    )

    assert output_path.exists()
    embeddings = np.load(output_path)
    assert embeddings.shape[0] == len(SAMPLE_SMILES)
    assert not np.isnan(embeddings).any(), f"NaNs found in {model} GPU embeddings"


def test_embed_smiles_empty_list():
    with pytest.raises(ValueError, match="SMILES list cannot be empty"):
        embed_smiles([], "chemberta-v1", "test.npy")


def test_embed_smiles_invalid_model():
    with pytest.raises(ValueError, match="Unknown model"):
        embed_smiles(["CCO"], "non-existent-model", "test.npy")
