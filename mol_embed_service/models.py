"""Model wrapper classes for different embedding models."""

import sys
import logging
from typing import List, Optional
from abc import ABC, abstractmethod
import torch
import numpy as np
import tqdm
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM


def _in_interactive_terminal() -> bool:
    """Return True if stderr is attached to an interactive terminal."""
    return sys.stderr.isatty()


class BaseEmbedder(ABC):
    """Base class for molecular embedding models."""

    def __init__(self, device: str = "cuda"):
        if device == "cuda" and not torch.cuda.is_available():
            print(
                "Warning: device='cuda' requested but CUDA is not available. Falling back to CPU."
            )
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

    @abstractmethod
    def embed(self, smiles_list: List[str], batch_size: int) -> np.ndarray:
        """Generate embeddings for a list of SMILES strings."""
        pass


class UniMol2Embedder(BaseEmbedder):
    """FP32 Uni-Mol2 CLS embeddings for one exact size-qualified model."""

    def __init__(
        self,
        variant: str = "unimol2-84M",
        device: str = "cuda",
        checkpoint_path=None,
        cache_dir=None,
    ):
        super().__init__(device)
        from .unimol2_checkpoint import load_checkpoint
        from .unimol2_config import CONFIGS, SOURCE_REVISION
        from .unimol2_model import UniMol2Model

        if variant not in CONFIGS:
            raise ValueError(f"Unknown Uni-Mol2 variant: {variant}")
        config, state = load_checkpoint(variant, checkpoint_path, cache_dir)
        self.config = config
        from rdkit import rdBase

        logging.getLogger(__name__).info(
            "Uni-Mol2 %s sha256=%s source_revision=%s rdkit=%s features=retained-etkdgv3-seed42-mmff94-max256-length-bucket-fp32",
            config.name,
            config.sha256,
            SOURCE_REVISION,
            rdBase.rdkitVersion,
        )
        self.model = UniMol2Model(config=config).float()
        self.model.load_state_dict(
            {k: v for k, v in state.items() if k in self.model.state_dict()}, strict=True
        )
        self.model.to(self.device).eval()

    def embed(self, smiles_list: list[str], batch_size: int) -> np.ndarray:
        from .unimol2_features import collate, featurize_smiles

        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        result = np.zeros((len(smiles_list), self.config.output_dim), dtype=np.float32)
        buckets = {}
        for i, smiles in enumerate(smiles_list):
            try:
                item = featurize_smiles(smiles)
            except Exception as exc:
                raise ValueError(f"Uni-Mol2 preprocessing failed at row {i}: {exc}") from exc
            if item is not None:
                n = len(item["token"])
                length = ((n + 4) // 4) * 4
                buckets.setdefault(length, []).append((i, item))
        with torch.inference_mode():
            for rows in buckets.values():
                for start in range(0, len(rows), batch_size):
                    batch = rows[start : start + batch_size]
                    data = collate([item for _, item in batch], self.device)
                    output = self.model(data)
                    if (
                        output.shape != (len(batch), self.config.output_dim)
                        or not torch.isfinite(output).all()
                    ):
                        raise ValueError("Uni-Mol2 produced invalid representation tensor")
                    values = output.float().cpu().numpy()
                    for (row, _), value in zip(batch, values):
                        result[row] = value
        return result


class ChemBERTaEmbedder(BaseEmbedder):
    """ChemBERTa v1/v2/v3 embedder using HuggingFace models."""

    MODEL_NAMES = {
        "chemberta-v1": "seyonec/ChemBERTa-zinc-base-v1",
        "chemberta-v2": "DeepChem/ChemBERTa-77M-MLM",
        "chemberta-v3": "DeepChem/ChemBERTa-77M-MTR",
    }

    def __init__(self, version: str = "chemberta-v1", device: str = "cuda"):
        super().__init__(device)
        model_name = self.MODEL_NAMES.get(version)
        if not model_name:
            raise ValueError(f"Unknown ChemBERTa version: {version}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def embed(self, smiles_list: List[str], batch_size: int) -> np.ndarray:
        """Generate embeddings using mean pooling of last hidden state."""
        embeddings = []

        with torch.no_grad():
            for i in tqdm.tqdm(
                range(0, len(smiles_list), batch_size),
                desc=f"Embedding with {self.__class__.__name__}",
                disable=not _in_interactive_terminal(),
            ):
                batch = smiles_list[i : i + batch_size]
                inputs = self.tokenizer(
                    batch, padding=True, truncation=True, max_length=512, return_tensors="pt"
                ).to(self.device)

                outputs = self.model(**inputs)
                # Mean pooling over sequence length
                attention_mask = inputs["attention_mask"]
                token_embeddings = outputs.last_hidden_state
                input_mask_expanded = (
                    attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                )
                sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                batch_embeddings = (sum_embeddings / sum_mask).cpu().numpy()

                embeddings.append(batch_embeddings)

        return np.vstack(embeddings)


class CDDDEmbedder(BaseEmbedder):
    """CDDD embedder using a strictly configured ONNX Runtime session."""

    def __init__(self, device: str = "cuda"):
        from .cddd_backend import CDDDBackend

        requested_device = torch.device(device)
        super().__init__("cpu")
        self.device = requested_device
        self._backend = CDDDBackend(requested_device)
        self.model = self._backend.model

    def embed(self, smiles_list: List[str], batch_size: int) -> np.ndarray:
        """Generate CDDD embeddings (512-dimensional)."""
        return self._backend.embed(smiles_list, batch_size)


class MolformerEmbedder(BaseEmbedder):
    """MoLFormer embedder using HuggingFace models."""

    MODEL_NAME = "ibm/MoLFormer-XL-both-10pct"

    def __init__(self, device: str = "cuda"):
        super().__init__(device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            self.MODEL_NAME, trust_remote_code=True, deterministic_eval=True
        ).to(self.device)
        self.model.eval()

    def embed(self, smiles_list: List[str], batch_size: int) -> np.ndarray:
        """Generate embeddings using mean pooling of last hidden state."""
        embeddings = []

        with torch.no_grad():
            for i in tqdm.tqdm(
                range(0, len(smiles_list), batch_size),
                desc=f"Embedding with {self.__class__.__name__}",
                disable=not _in_interactive_terminal(),
            ):
                batch = smiles_list[i : i + batch_size]
                inputs = self.tokenizer(batch, padding=True, return_tensors="pt").to(self.device)

                outputs = self.model(**inputs)
                # Mean pooling over sequence length
                attention_mask = inputs["attention_mask"]
                token_embeddings = outputs.last_hidden_state
                input_mask_expanded = (
                    attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                )
                sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                batch_embeddings = (sum_embeddings / sum_mask).cpu().numpy()

                embeddings.append(batch_embeddings)

        return np.vstack(embeddings)


class CheMeleonEmbedder(BaseEmbedder):
    """CheMeleon embedder using ChemProp MPNN."""

    EMBEDDING_DIM = 2048

    def __init__(self, device: str = "cuda"):
        super().__init__(device)
        from .chemeleon_fingerprint import CheMeleonFingerprint

        self.model = CheMeleonFingerprint(device=self.device)

    def embed(self, smiles_list: List[str], batch_size: int) -> np.ndarray:
        """Generate CheMeleon embeddings (2048-dimensional)."""
        embeddings = []

        for i in tqdm.tqdm(
            range(0, len(smiles_list), batch_size),
            desc=f"Embedding with {self.__class__.__name__}",
            disable=not _in_interactive_terminal(),
        ):
            batch = smiles_list[i : i + batch_size]
            valid_indices = []
            valid_smiles = []

            for j, s in enumerate(batch):
                from rdkit import Chem

                mol = Chem.MolFromSmiles(s) if s else None
                if mol is not None:
                    valid_indices.append(j)
                    valid_smiles.append(s)

            batch_embeddings = np.zeros((len(batch), self.EMBEDDING_DIM), dtype=np.float32)

            if valid_smiles:
                emb = self.model(valid_smiles)
                for idx, valid_idx in enumerate(valid_indices):
                    batch_embeddings[valid_idx] = emb[idx]

            embeddings.append(batch_embeddings)

        return np.vstack(embeddings)


class MolJEPAEmbedder(BaseEmbedder):
    """Mol-JEPA CLS embeddings from the pinned Hugging Face checkpoint.

    Empty or malformed SMILES receive zero rows so output remains aligned with
    the input. Errors raised while processing RDKit-valid molecules propagate
    to the caller rather than being silently converted to zero rows.
    """

    MODEL_NAME = "Flogrammer/Mol-JEPA"
    MODEL_REVISION = "4c912b450175f31b5ba913a5dc921c03b27b985a"
    EMBEDDING_DIM = 512

    def __init__(self, device: str = "cuda"):
        super().__init__(device)
        self.model = AutoModel.from_pretrained(
            self.MODEL_NAME,
            revision=self.MODEL_REVISION,
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()

    def embed(self, smiles_list: list[str], batch_size: int) -> np.ndarray:
        """Generate row-aligned 512-dimensional ``out.cls`` embeddings."""
        from rdkit import Chem

        embeddings = []
        with torch.no_grad():
            for i in tqdm.tqdm(
                range(0, len(smiles_list), batch_size),
                desc=f"Embedding with {self.__class__.__name__}",
                disable=not _in_interactive_terminal(),
            ):
                batch = smiles_list[i : i + batch_size]
                valid_indices = []
                valid_smiles = []
                for j, smiles in enumerate(batch):
                    mol = Chem.MolFromSmiles(smiles) if smiles else None
                    if mol is not None and mol.GetNumAtoms() > 0:
                        valid_indices.append(j)
                        valid_smiles.append(smiles)

                batch_embeddings = np.zeros((len(batch), self.EMBEDDING_DIM), dtype=np.float32)
                if valid_smiles:
                    output = self.model(valid_smiles)
                    cls = output.cls.detach().cpu().numpy()
                    expected_shape = (len(valid_smiles), self.EMBEDDING_DIM)
                    if cls.shape != expected_shape:
                        raise ValueError(
                            f"Mol-JEPA out.cls has shape {cls.shape}; expected {expected_shape}"
                        )
                    cls = cls.astype(np.float32, copy=False)
                    for row, valid_idx in enumerate(valid_indices):
                        batch_embeddings[valid_idx] = cls[row]

                embeddings.append(batch_embeddings)

        return np.vstack(embeddings)


class MistEmbedder(BaseEmbedder):
    """MIST (Molecular Insight SMILES Transformer) embedder."""

    MODEL_NAMES = {
        "mist-1.8B": "mist-models/mist-1.8B-dh61satt",
        "mist-28M": "mist-models/mist-28M-ti624ev1",
    }

    EMBEDDING_DIMS = {
        "mist-1.8B": 2304,
        "mist-28M": 512,
    }

    def __init__(self, version: str = "mist-1.8B", device: str = "cuda"):
        super().__init__(device)
        model_name = self.MODEL_NAMES.get(version)
        if not model_name:
            raise ValueError(f"Unknown MIST version: {version}")

        self.version = version
        self.embedding_dim = self.EMBEDDING_DIMS[version]

        # Import SmirkTokenizerFast so AutoTokenizer can resolve it
        from smirk import SmirkTokenizerFast  # noqa: F401

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(self.device)
        self.model.eval()

    @staticmethod
    def _kekulize(smiles: str) -> Optional[str]:
        from rdkit import Chem

        mol = Chem.MolFromSmiles(smiles) if smiles else None
        if mol is None:
            return None
        Chem.Kekulize(mol)
        return Chem.MolToSmiles(mol, kekuleSmiles=True)

    def embed(self, smiles_list: List[str], batch_size: int) -> np.ndarray:
        """Generate MIST embeddings using [CLS] token of last hidden state."""
        embeddings = []

        with torch.no_grad():
            for i in tqdm.tqdm(
                range(0, len(smiles_list), batch_size),
                desc=f"Embedding with {self.__class__.__name__}({self.version})",
                disable=not _in_interactive_terminal(),
            ):
                batch = smiles_list[i : i + batch_size]
                valid_indices = []
                valid_smiles = []

                for j, s in enumerate(batch):
                    k = self._kekulize(s)
                    if k is not None:
                        valid_indices.append(j)
                        valid_smiles.append(k)

                batch_embeddings = np.zeros((len(batch), self.embedding_dim), dtype=np.float32)

                if valid_smiles:
                    inputs = self.tokenizer(
                        valid_smiles,
                        padding=True,
                        truncation=True,
                        max_length=512,
                        return_tensors="pt",
                    ).to(self.device)

                    outputs = self.model(**inputs)
                    # Extract [CLS] / first-token embedding
                    cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

                    for idx, valid_idx in enumerate(valid_indices):
                        batch_embeddings[valid_idx] = cls_embeddings[idx]

                embeddings.append(batch_embeddings)

        return np.vstack(embeddings)
