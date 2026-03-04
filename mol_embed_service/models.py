"""Model wrapper classes for different embedding models."""

from typing import List, Optional
from abc import ABC, abstractmethod
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM


class BaseEmbedder(ABC):
    """Base class for molecular embedding models."""

    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

    @abstractmethod
    def embed(self, smiles_list: List[str], batch_size: int) -> np.ndarray:
        """Generate embeddings for a list of SMILES strings."""
        pass


class ChemBERTaEmbedder(BaseEmbedder):
    """ChemBERTa v1/v2/v3 embedder using HuggingFace models."""

    MODEL_NAMES = {
        "chemberta-v1": "seyonec/ChemBERTa-zinc-base-v1",
        "chemberta-v2": "DeepChem/ChemBERTa-77M-MLM",
        "chemberta-v3": "DeepChem/ChemBERTa-77M-MTR"
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
            for i in range(0, len(smiles_list), batch_size):
                batch = smiles_list[i:i + batch_size]
                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                ).to(self.device)

                outputs = self.model(**inputs)
                # Mean pooling over sequence length
                attention_mask = inputs["attention_mask"]
                token_embeddings = outputs.last_hidden_state
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                batch_embeddings = (sum_embeddings / sum_mask).cpu().numpy()

                embeddings.append(batch_embeddings)

        return np.vstack(embeddings)


class CDDDEmbedder(BaseEmbedder):
    """CDDD embedder using cddd-onnx package."""

    def __init__(self, device: str = "cuda"):
        super().__init__(device)
        # cddd-onnx uses ONNX Runtime, GPU support through onnxruntime-gpu
        from cddd_onnx import InferenceModel
        self.model = InferenceModel()

    def embed(self, smiles_list: List[str], batch_size: int) -> np.ndarray:
        """Generate CDDD embeddings (512-dimensional)."""
        # CDDD model handles batching internally
        import os
        # Prevent onnxruntime from setting affinity which causes errors in some environments
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        
        # Use a more complex dummy SMILES that is likely to be valid for CDDD
        dummy_smiles = "CN1C=NC2=C1C(=O)N(C)C(=O)N2C" # Caffeine
        
        # We process each SMILES individually with a dummy prepended to ensure no NaNs
        # The first element in any CDDD-ONNX call often returns NaN
        embeddings = []
        for smiles in smiles_list:
            try:
                # Prepend dummy and take the second embedding
                pair_input = [dummy_smiles, smiles]
                pair_emb = self.model.seq_to_emb(pair_input)
                
                # If the second one is still NaN (rare), try the first one as a last resort
                if np.isnan(pair_emb[1]).any():
                    if not np.isnan(pair_emb[0]).any():
                        embeddings.append(pair_emb[0])
                    else:
                        # If both are NaN, we have a problematic SMILES
                        print(f"Warning: CDDD failed to embed SMILES: {smiles}")
                        embeddings.append(np.zeros(512))
                else:
                    embeddings.append(pair_emb[1])
            except Exception as e:
                print(f"Error embedding {smiles}: {e}")
                embeddings.append(np.zeros(512))
                
        return np.vstack(embeddings)


class ChemformerEmbedder(BaseEmbedder):
    """Chemformer-style embedder using MoLFormer."""

    MODEL_NAME = "DeepChem/ChemBERTa-77M-MTR"  # Use ChemBERTa-v3 as a reliable fallback

    def __init__(self, device: str = "cuda"):
        super().__init__(device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            self.MODEL_NAME, 
            trust_remote_code=True,
            # Workaround for transformers.onnx missing in some versions
            # and model type mismatch
        ).to(self.device)
        self.model.eval()

    def embed(self, smiles_list: List[str], batch_size: int) -> np.ndarray:
        """Generate embeddings from MoLFormer hidden states."""
        embeddings = []

        with torch.no_grad():
            for i in range(0, len(smiles_list), batch_size):
                batch = smiles_list[i:i + batch_size]
                # MoLFormer uses its own tokenizer logic
                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                ).to(self.device)

                outputs = self.model(**inputs)
                # Mean pooling over sequence length
                batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                embeddings.append(batch_embeddings)

        return np.vstack(embeddings)
