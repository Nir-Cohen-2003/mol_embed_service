"""Differentiable adapters for the initially supported pretrained encoders."""

from torch import nn


class Backbone(nn.Module):
    def __init__(self, encoder, dimension):
        super().__init__()
        self.encoder = encoder
        self.dimension = dimension

    def forward(self, smiles):
        raise NotImplementedError


class ChemBERTaBackbone(Backbone):
    def __init__(self, encoder, tokenizer, dimension):
        super().__init__(encoder, dimension)
        self.tokenizer = tokenizer

    def forward(self, smiles):
        from ..tensor_features import chemberta_features

        return chemberta_features(
            self.encoder, self.tokenizer, smiles, next(self.encoder.parameters()).device
        )


class CheMeleonEncoder(nn.Module):
    """Only the fingerprint path; excludes the unused Chemprop predictor."""

    def __init__(self, message_passing, aggregation):
        super().__init__()
        self.message_passing = message_passing
        self.aggregation = aggregation

    def fingerprint(self, graph):
        values = self.message_passing(graph)
        return self.aggregation(values, graph.batch)


class CheMeleonBackbone(Backbone):
    def __init__(self, model, featurizer, checkpoint_provenance=None):
        message_passing = model.message_passing
        d_vd = message_passing.hparams.get("d_vd", 0)
        if d_vd not in (None, 0):
            raise ValueError("CheMeleon backbone does not support descriptor dimension d_vd")
        from chemprop import featurizers

        supported_shape = featurizers.SimpleMoleculeMolGraphFeaturizer().shape
        dimensions = (message_passing.hparams.get("d_v"), message_passing.hparams.get("d_e"))
        if dimensions != supported_shape:
            raise ValueError(
                "CheMeleon backbone requires SimpleMoleculeMolGraphFeaturizer dimensions "
                f"{supported_shape}, got {dimensions}"
            )
        super().__init__(model, int(message_passing.output_dim))
        self.featurizer = featurizer
        self.checkpoint_provenance = checkpoint_provenance

    def forward(self, smiles):
        from ..tensor_features import chemeleon_features

        return chemeleon_features(
            self.encoder, self.featurizer, smiles, next(self.encoder.parameters()).device
        )


def load_backbone(config):
    if config.model.startswith("chemberta-"):
        from transformers import AutoModel, AutoTokenizer

        from ..loading_policy import offline_enabled
        from ..models import ChemBERTaEmbedder

        name = ChemBERTaEmbedder.MODEL_NAMES[config.model]
        tokenizer = AutoTokenizer.from_pretrained(
            name, revision=config.revision, local_files_only=offline_enabled()
        )
        encoder = AutoModel.from_pretrained(
            name, revision=config.revision, local_files_only=offline_enabled()
        ).to(config.device)
        return ChemBERTaBackbone(encoder, tokenizer, int(encoder.config.hidden_size))
    from ..chemeleon_fingerprint import CheMeleonFingerprint

    fingerprint = CheMeleonFingerprint(checkpoint_path=config.checkpoint_path, device=config.device)
    encoder = CheMeleonEncoder(fingerprint.model.message_passing, fingerprint.model.agg)
    encoder.load_state_dict(
        {
            k: v
            for k, v in fingerprint.model.state_dict().items()
            if k.startswith(("message_passing.", "agg."))
        }
    )
    return CheMeleonBackbone(encoder, fingerprint.featurizer, fingerprint.provenance)
