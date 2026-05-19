# Molecular Embedding Service

GPU-accelerated molecular embedding generation for ChemBERTa (v1-3), CDDD, MolFormer, CheMeleon, and MIST.

## Features

- **7 Models**: ChemBERTa-v1, ChemBERTa-v2, ChemBERTa-v3, CDDD, MolFormer, CheMeleon, MIST (1.8B & 28M)
- **GPU Acceleration**: Efficient batched inference with CUDA support
- **Embedding Size API**: Query dimensions programmatically via `EMBEDDING_SIZES` and `get_embedding_size()`
- **Clean Input**: Optimized for pre-validated SMILES strings
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
    model="chemberta-v1",  # or v2, v3, cddd, molformer, chemeleon, mist-1.8B, mist-28M
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

> **Note on Embedding Dimensions**: The vector size is determined by the underlying pretrained model weights and is **not user-configurable**. ChemBERTa-v1 and MoLFormer output 768-dim vectors, CDDD outputs 512-dim vectors, ChemBERTa-v2/v3 output 384-dim vectors, CheMeleon outputs 2048-dim vectors, MIST-1.8B outputs 2304-dim vectors, and MIST-28M outputs 512-dim vectors. The library exports `ModelType` and `EmbeddingDim` type aliases for static type checking.

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

### Parameters

- `smiles_list` (List[str]): SMILES strings to embed
- `model` (str): Model identifier
- `output_path` (str): Output .npy file path
- `batch_size` (int, default=32): Inference batch size
- `device` (str, default="cuda"): "cuda" or "cpu"

## Development

# Run tests
pixi run test

# Format code
pixi run format

# Lint
pixi run lint

## Requirements

- Python 3.10+
- CUDA 12.1+ (for GPU support)
- 4GB+ GPU memory recommended

## License

MIT