# Molecular Embedding Service

GPU-accelerated molecular embedding generation for ChemBERTa (v1-3), CDDD, MolFormer, CheMeleon, MIST, and Mol-JEPA.

## Features

- **9 selectable models**: ChemBERTa-v1, ChemBERTa-v2, ChemBERTa-v3, CDDD, MolFormer, CheMeleon, MIST (1.8B & 28M), and Mol-JEPA
- **GPU Acceleration**: Efficient batched inference with CUDA support
- **Embedding Size API**: Query dimensions programmatically via `EMBEDDING_SIZES` and `get_embedding_size()`
- **Mol-JEPA input handling**: Empty or malformed SMILES receive aligned zero rows; Mol-JEPA errors for valid molecules propagate
- **Extensible**: Easy to add new models in the future

## Installation

# Clone repository
git clone <repo-url>
cd mol-embed-service

# Install with Pixi
pixi install

# Activate environment
pixi shell

## Usage

### Python API

from mol_embed_service import embed_smiles, ModelType, EmbeddingDim

smiles_list = ["CCO", "c1ccccc1", "CC(=O)O"]

# Generate embeddings
embed_smiles(
    smiles_list=smiles_list,
    model="chemberta-v1",  # or v2, v3, cddd, molformer, chemeleon, mist-1.8B, mist-28M, mol-jepa
    output_path="embeddings.npy",
    batch_size=32,
    device="cuda"
)

# Load embeddings
import numpy as np
embeddings = np.load("embeddings.npy")
print(embeddings.shape)  # (3, embedding_dim)

### Available Models

| Model | Version | Embedding Dim | Notes |
|-------|---------|---------------|-------|
| `chemberta-v1` | ZINC base | 768 | Original ChemBERTa |
| `chemberta-v2` | 77M MLM | 384 | Masked Language Model |
| `chemberta-v3` | 77M MTR | 384 | Multi-task Regression |
| `cddd` | ONNX | 512 | Continuous descriptors |
| `molformer` | MoLFormer-XL | 768 | Transformer encoder |
| `chemeleon` | ChemProp MPNN | 2048 | Learned MPNN fingerprints |
| `mist-1.8B` | MIST 1.8B | 2304 | RoBERTa-PreLayerNorm encoder |
| `mist-28M` | MIST 28M | 512 | Lightweight MIST encoder |
| `mol-jepa` | Flogrammer/Mol-JEPA | 512 | Documented CLS representation |

> **Note on Embedding Dimensions**: The vector size is determined by the underlying pretrained model weights and is **not user-configurable**. ChemBERTa-v1 and MoLFormer output 768-dim vectors, CDDD outputs 512-dim vectors, ChemBERTa-v2/v3 output 384-dim vectors, CheMeleon outputs 2048-dim vectors, MIST-1.8B outputs 2304-dim vectors, MIST-28M outputs 512-dim vectors, and Mol-JEPA returns its 512-dimensional `out.cls` representation (not 5125). The library exports `ModelType` and `EmbeddingDim` type aliases for static type checking.

### Embedding Size API

You can query embedding dimensions programmatically without instantiating models:

```python
from mol_embed_service import EMBEDDING_SIZES, get_embedding_size

# Access all embedding sizes
print(EMBEDDING_SIZES)
# {'chemberta-v1': 768, 'chemberta-v2': 384, ...}

# Get dimension for a specific model
dim = get_embedding_size("chemberta-v1")
print(dim)  # 768

# Use for pre-allocating arrays or validation
import numpy as np
embeddings = np.zeros((num_molecules, get_embedding_size("cddd")))
```

> **Note**: If you use the `cddd` model, ensure `cddd-onnx` is included in your environment (it is included in the default package dependencies).
>
> **Note**: If you use the `mist-1.8B` or `mist-28M` models, the `smirk` package is required (included by default). Building `smirk` requires a Rust compiler. Install Rust from [rust-lang.org](https://www.rust-lang.org/tools/install) before installing this package if `smirk` wheels are not available for your platform.
>
> **Note on Mol-JEPA**: Mol-JEPA is downloaded on first use through the Hugging Face cache (about 182 MB). This service loads `Flogrammer/Mol-JEPA` with pinned remote code at revision `4c912b450175f31b5ba913a5dc921c03b27b985a` and uses the official project's [512-dimensional CLS output](https://huggingface.co/Flogrammer/Mol-JEPA/blob/4c912b450175f31b5ba913a5dc921c03b27b985a/README.md). The exact immutable weight URL is [`model.safetensors`](https://huggingface.co/Flogrammer/Mol-JEPA/resolve/4c912b450175f31b5ba913a5dc921c03b27b985a/model.safetensors). It requires `transformers>=4.50.3`, `torch-geometric`, `molfeat`, and `safetensors`; custom code execution is required. For Mol-JEPA, empty or malformed SMILES are zero-filled while errors for RDKit-valid molecules are propagated.
>
> Mol-JEPA is from [Boehringer-Ingelheim/mol-jepa](https://github.com/Boehringer-Ingelheim/mol-jepa) and is licensed **CC BY-NC 4.0** ([model card/license](https://huggingface.co/Flogrammer/Mol-JEPA/blob/4c912b450175f31b5ba913a5dc921c03b27b985a/README.md)). The package source remains MIT, but that does not cover the checkpoint or override its terms. Release/deployment requires compliance with the model terms, and commercial use requires separate authorization.

### Parameters

- `smiles_list` (List[str]): SMILES strings to embed
- `model` (str): Model identifier
- `output_path` (str): Output .npy file path
- `batch_size` (int, default=32): Inference batch size
- `device` (str, default="cuda"): "cuda" or "cpu"

## Development

# Run tests
pixi run test

# Lint
pixi run lint

## Requirements

- Python 3.10+
- CUDA 12.1+ (for GPU support)
- 4GB+ GPU memory recommended

## License

MIT