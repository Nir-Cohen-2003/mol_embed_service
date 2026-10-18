# Molecular Embedding Service

Molecular embedding generation for ChemBERTa (v1-3), CDDD, MolFormer, CheMeleon, MIST, Mol-JEPA, Uni-Mol2, and optional Monroe, MiniMol, and GRASP backends.

## Features

- **17 selectable model variants**: ChemBERTa-v1/v2/v3, CDDD, MolFormer, CheMeleon, MIST (1.8B & 28M), Mol-JEPA, five Uni-Mol2 sizes, Monroe, MiniMol, and GRASP
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
    model="chemberta-v1",  # also: monroe, minimol, grasp, and the models below
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
| `monroe` | Monroe | 720 | Graph attention pooled representation; explicit local checkpoint required |
| `minimol` | MiniMol | 512 | Graphium node fingerprints with global max pooling; optional dependency |
| `grasp` | GRASP | 768 | Final-layer CLS representation; optional dependency |
| `unimol2-84M` | Uni-Mol2 84M | 768 | FP32 virtual-node/CLS; automatic verified local cache |
| `unimol2-164M` | Uni-Mol2 164M | 768 | FP32 virtual-node/CLS |
| `unimol2-310M` | Uni-Mol2 310M | 1024 | FP32 virtual-node/CLS |
| `unimol2-570M` | Uni-Mol2 570M | 1536 | FP32 virtual-node/CLS |
| `unimol2-1.1B` | Uni-Mol2 1.1B | 1536 | FP32 virtual-node/CLS |

> **Uni-Mol2**: Five exact selectors are available (`unimol2-84M`, `-164M`, `-310M`, `-570M`, `-1.1B`), returning unnormalized FP32 CLS vectors with dimensions 768/768/1024/1536/1536. First use downloads the selected verified artifact to a size- and revision-specific cache (about 0.337/0.659/1.241/2.281/4.518 GB). For these Uni-Mol2 HTTPS downloads only, TLS certificate and hostname verification are disabled; each artifact is still checked against its pinned byte count and SHA-256 digest before publication. `UNIMOL2_CHECKPOINT_PATH` or the adapter's explicit path overrides the cache and must point to that exact official artifact; corrupt cache files are replaced only after a verified download. Artifacts use Hugging Face revision [`bf37d253c0f5ac61f468137eb00059978a104ff0`](https://huggingface.co/dptech/Uni-Mol2/tree/bf37d253c0f5ac61f468137eb00059978a104ff0), with digests recorded in `mol_embed_service/unimol2_config.py`; architecture/feature source is pinned at [`90f52c41299a1a582da0f9765e9f87aa21faa16a`](https://github.com/deepmodeling/Uni-Mol/tree/90f52c41299a1a582da0f9765e9f87aa21faa16a). The service retains graph features and generates one centered ETKDGv3 conformer (seed 42, MMFF94 when available), caps input at 256 heavy atoms, zero-fills invalid/empty SMILES, and raises with row context for valid-molecule geometry failures. It uses per-molecule CLS-inclusive lengths rounded to four for batching and FP32 inference. RDKit versions can change generated conformers; this is a documented service policy, not identical upstream training geometry. The local native-PyTorch implementation includes source-derived MIT attribution; the model card also declares MIT and requests citation to the Uni-Mol2 paper.
>
> **Note on Embedding Dimensions**: Dimensions are determined by the pretrained models: ChemBERTa-v1/MoLFormer/GRASP 768, Monroe 720, CDDD/ChemBERTa-v2/v3/MiniMol/MIST-28M/Mol-JEPA 512 (except v2/v3 are 384), CheMeleon 2048, and MIST-1.8B 2304. The library exports `ModelType` and `EmbeddingDim` type aliases.

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

### Optional model installations and limitations

The `monroe`, `minimol`, and `grasp` adapters are optional: their upstream packages are not added to this service's global dependencies. Install each backend in a separate, isolated environment and follow its pinned upstream instructions; these stacks have not been jointly dependency-resolved. Adapters import upstream libraries only when selected. The new adapters reject invalid/empty SMILES with row indices and fail the whole call for malformed outputs; unlike Mol-JEPA they do not zero-fill bad rows.

- **Monroe**: use source revision [`57238edfffea03808abe761a00cd9a75fa41bb95`](https://github.com/blazejba/Monroe/tree/57238edfffea03808abe761a00cd9a75fa41bb95). Acquire the checkpoint's Git LFS files from that project's checkpoint repository; code revision does not identify checkpoint contents. Configure its directory containing readable, nonempty `config.json` and `weights.pt` either with `MONROE_CHECKPOINT_DIR=/path/to/checkpoint` or the adapter's `checkpoint_dir` constructor option. LFS pointer files are rejected; no checkpoint is downloaded automatically. The upstream helper uses a fork-based multiprocessing pool even with `n_workers=1`, and moves the encoder before pool creation. Fork-platform dependence and CUDA/threaded/prefork-serving safety are unverified. Load only trusted checkpoint files.
- **MiniMol**: use source revision [`41642328affbb35e4950b7b7a9020aa95919b503`](https://github.com/graphcore-research/minimol/tree/41642328affbb35e4950b7b7a9020aa95919b503). It depends on Graphium 2.4.7, Hydra, and Torch-specific compiled extensions; source declares 1.3.4 while PyPI has reported 1.3.5, so wheel/source parity is unknown. This adapter keeps one process-local backend instance initialized with its upstream microbatch size 100; requests are chunked by the supplied `batch_size`. Its support boundary is deliberately conservative: only CUDA-unavailable/hidden CPU environments are supported. There, `device="cpu"` runs on CPU, and an unavailable `device="cuda"` request retains the base adapter's CPU fallback. If CUDA is visible, **both** requests are rejected before backend loading/use. This is not a claim that CUDA visibility causes GPU inference: actual network and input placement are not established, so CUDA execution is unsupported. Hide CUDA before process startup, e.g. `CUDA_VISIBLE_DEVICES=''`, to use MiniMol's CPU path.
- **GRASP**: use code revision [`fa36c355cac87bc5b18a9deff34dda65ba541c0b`](https://github.com/caithmac/GRASP/tree/fa36c355cac87bc5b18a9deff34dda65ba541c0b) and pinned [Hugging Face checkpoint revision `4804c0a558307c944da5e8de1d182b4b4c994dae`](https://huggingface.co/caithmac/GRASP/tree/4804c0a558307c944da5e8de1d182b4b4c994dae). It returns final-layer CLS vectors, rejects molecules above 511 atoms, and does not encode stereochemistry; the atom limit is not evidence of generalization to that size. The model terms identify **CC BY-NC 4.0** ([checkpoint LICENSE](https://huggingface.co/caithmac/GRASP/blob/4804c0a558307c944da5e8de1d182b4b4c994dae/LICENSE), [upstream NOTICE](https://github.com/caithmac/GRASP/blob/fa36c355cac87bc5b18a9deff34dda65ba541c0b/NOTICE)); the upstream README also contains a conflicting MIT sentence. Review the governing terms: public availability is not commercial-use permission. The service's MIT license does not override model terms.

Manual smoke check after installing/configuring the desired backend (MiniMol requires CUDA hidden before startup):

```python
import numpy as np
from mol_embed_service import embed_smiles

smiles = ["CCO", "c1ccccc1", "CCO"]
embed_smiles(smiles, "grasp", "optional.npy", batch_size=2, device="cpu")
values = np.load("optional.npy")
assert values.shape == (3, 768)
assert values.dtype == np.float32 and np.isfinite(values).all()
assert np.allclose(values[0], values[2])

# For MiniMol, launch the process with CUDA hidden before Python starts:
# CUDA_VISIBLE_DEVICES='' python smoke.py
# Then use the same input in the supported CPU-only environment:
# embed_smiles(smiles, "minimol", "minimol.npy", batch_size=2, device="cpu")
```

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