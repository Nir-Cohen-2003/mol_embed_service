"""Fine-tuned encoder and prediction interface."""

import numpy as np
import torch
from torch import nn

from .components import make_activation, make_norm
from .config import HeadConfig


class FineTunedModel(nn.Module):
    def __init__(
        self, backbone, head_config, task, outputs, mean=None, scale=None, train_backbone=True
    ):
        super().__init__()
        if task not in {"regression", "binary"}:
            raise ValueError("task must be 'regression' or 'binary'")
        if type(outputs) is not int or outputs <= 0:
            raise ValueError("outputs must be a positive integer")
        if type(train_backbone) is not bool:
            raise TypeError("train_backbone must be bool")
        if not isinstance(head_config, HeadConfig):
            raise TypeError("head_config must be HeadConfig")
        dimension = getattr(backbone, "dimension", None)
        if type(dimension) is not int or dimension <= 0:
            raise ValueError("backbone.dimension must be a positive integer")
        self.backbone = backbone
        self.task = task
        self.num_outputs = outputs
        self.embedding_dim = dimension
        self.head_config = head_config
        layers = []
        width = self.embedding_dim
        for hidden in head_config.hidden_dims:
            layers.extend(
                [
                    nn.Linear(width, hidden),
                    make_norm(head_config.normalization, hidden, head_config.normalization_kwargs),
                    make_activation(head_config.activation, hidden),
                    nn.Dropout(head_config.dropout),
                ]
            )
            width = hidden
        layers.append(nn.Linear(width, outputs))
        self.head = nn.Sequential(*layers)
        target_mean = torch.as_tensor(
            mean if mean is not None else np.zeros(outputs), dtype=torch.float32
        )
        target_scale = torch.as_tensor(
            scale if scale is not None else np.ones(outputs), dtype=torch.float32
        )
        if target_mean.shape != (outputs,) or not torch.isfinite(target_mean).all():
            raise ValueError("mean must contain one finite value per output")
        if (
            target_scale.shape != (outputs,)
            or not torch.isfinite(target_scale).all()
            or torch.any(target_scale <= 0)
        ):
            raise ValueError("scale must contain one finite positive value per output")
        self.register_buffer("target_mean", target_mean)
        self.register_buffer("target_scale", target_scale)
        self.train_backbone = train_backbone
        for p in self.backbone.parameters():
            p.requires_grad_(train_backbone)

    def train(self, mode=True):
        super().train(mode)
        if not self.train_backbone:
            self.backbone.eval()
        return self

    def forward(self, smiles):
        if self.train_backbone:
            features = self.backbone(smiles)
        else:
            with torch.no_grad():
                features = self.backbone(smiles)
        return self.head(features)

    @staticmethod
    def _validate_smiles(smiles):
        from rdkit import Chem

        for index, value in enumerate(smiles):
            molecule = Chem.MolFromSmiles(value) if isinstance(value, str) and value.strip() else None
            if molecule is None or molecule.GetNumAtoms() == 0:
                raise ValueError(f"Invalid SMILES at row {index}: {value!r}")

    def _module_modes(self):
        return [(module, module.training) for module in self.modules()]

    @staticmethod
    def _restore_modes(modes):
        for module, training in modes:
            module.training = training

    def embed(self, smiles):
        smiles = list(smiles)
        self._validate_smiles(smiles)
        if not smiles:
            return np.empty((0, self.embedding_dim), dtype=np.float32)
        modes = self._module_modes()
        self.eval()
        try:
            with torch.no_grad():
                return self.backbone(smiles).float().cpu().numpy()
        finally:
            self._restore_modes(modes)

    def predict(self, smiles):
        smiles = list(smiles)
        self._validate_smiles(smiles)
        if not smiles:
            return np.empty((0, self.num_outputs), dtype=np.float32)
        modes = self._module_modes()
        self.eval()
        try:
            with torch.no_grad():
                values = self(smiles).float()
                if self.task == "binary":
                    values = values.sigmoid()
                else:
                    values = values * self.target_scale + self.target_mean
                return values.cpu().numpy()
        finally:
            self._restore_modes(modes)
