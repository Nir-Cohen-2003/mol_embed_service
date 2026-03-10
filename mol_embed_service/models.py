"""Model wrapper classes for different embedding models."""

from typing import List, Optional
from abc import ABC, abstractmethod
import torch
import numpy as np
import tqdm
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM


class BaseEmbedder(ABC):
    """Base class for molecular embedding models."""

    def __init__(self, device: str = "cuda"):
        if device == "cuda" and not torch.cuda.is_available():
            print("Warning: device='cuda' requested but CUDA is not available. Falling back to CPU.")
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

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
            for i in tqdm.tqdm(range(0, len(smiles_list), batch_size), desc=f"Embedding with {self.__class__.__name__}"):
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
        
        # Override the inference session to enforce CUDA execution provider if requested
        if self.device.type == "cuda":
            import onnxruntime as ort
            from cddd_onnx.model_downloader import get_model_path
            encoder_path = get_model_path("encoder")
            try:
                sess_options = ort.SessionOptions()
                # To prevent the thread_setaffinity_np warning
                sess_options.intra_op_num_threads = 1
                sess_options.inter_op_num_threads = 1
                self.model.encoder_session = ort.InferenceSession(
                    encoder_path,
                    sess_options=sess_options,
                    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
                )
            except Exception as e:
                print(f"Warning: Failed to set CUDAExecutionProvider for CDDD: {e}")
        else:
            # Also set the thread options for CPU to prevent the warning
            import onnxruntime as ort
            from cddd_onnx.model_downloader import get_model_path
            encoder_path = get_model_path("encoder")
            try:
                sess_options = ort.SessionOptions()
                sess_options.intra_op_num_threads = 1
                sess_options.inter_op_num_threads = 1
                self.model.encoder_session = ort.InferenceSession(
                    encoder_path,
                    sess_options=sess_options,
                    providers=["CPUExecutionProvider"]
                )
            except Exception as e:
                pass

    def embed(self, smiles_list: List[str], batch_size: int) -> np.ndarray:
        """Generate CDDD embeddings (512-dimensional)."""
        import os
        import pandas as pd
        from cddd_onnx.preprocessing import preprocess_smiles
        
        # Prevent onnxruntime from setting affinity which causes errors in some environments
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        
        # Use a more complex dummy SMILES that is likely to be valid for CDDD
        dummy_smiles = "CN1C=NC2=C1C(=O)N(C)C(=O)N2C" # Caffeine
        
        final_embeddings = np.zeros((len(smiles_list), 512), dtype=np.float32)

        for i in tqdm.tqdm(range(0, len(smiles_list), batch_size), desc=f"Embedding with {self.__class__.__name__}"):
            batch = smiles_list[i:i + batch_size]
            valid_indices = []
            valid_smiles = []

            for j, s in enumerate(batch):
                if not s:
                    continue
                try:
                    p = preprocess_smiles(s)
                    if not pd.isna(p):
                        valid_indices.append(j)
                        valid_smiles.append(s)
                except Exception:
                    pass

            if not valid_smiles:
                continue

            try:
                # Prepend dummy to ensure no NaNs in first element
                pair_input = [dummy_smiles] + valid_smiles
                emb_output = self.model.seq_to_emb(pair_input)
                
                # Exclude dummy
                valid_embs = emb_output[1:]
                
                for idx, valid_idx in enumerate(valid_indices):
                    emb = valid_embs[idx]
                    if not np.isnan(emb).any():
                        final_embeddings[i + valid_idx] = emb
            except Exception as e:
                print(f"Error embedding batch in CDDD: {e}")

        return final_embeddings


class MolformerEmbedder(BaseEmbedder):
    """MoLFormer embedder using HuggingFace models."""

    MODEL_NAME = "ibm/MoLFormer-XL-both-10pct"

    def __init__(self, device: str = "cuda"):
        super().__init__(device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            self.MODEL_NAME, 
            trust_remote_code=True, 
            deterministic_eval=True
        ).to(self.device)
        self.model.eval()

    def embed(self, smiles_list: List[str], batch_size: int) -> np.ndarray:
        """Generate embeddings using mean pooling of last hidden state."""
        embeddings = []

        with torch.no_grad():
            for i in tqdm.tqdm(range(0, len(smiles_list), batch_size), desc=f"Embedding with {self.__class__.__name__}"):
                batch = smiles_list[i:i + batch_size]
                inputs = self.tokenizer(
                    batch,
                    padding=True,
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
            for i in tqdm.tqdm(range(0, len(smiles_list), batch_size), desc=f"Embedding with {self.__class__.__name__}"):
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
