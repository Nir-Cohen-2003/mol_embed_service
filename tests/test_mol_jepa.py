import numpy as np
import pytest
import torch

from mol_embed_service.embedder import embed_smiles, get_embedding_size
from mol_embed_service.models import MolJEPAEmbedder

MODEL_NAME = "Flogrammer/Mol-JEPA"
MODEL_REVISION = "4c912b450175f31b5ba913a5dc921c03b27b985a"


class FakeOutput:
    def __init__(self, cls):
        self.cls = cls
        self.predictions = torch.full((cls.shape[0], 12, 512), -7.0)
        self.embeddings = torch.full((cls.shape[0], 13, 512), -9.0)


class FakeModel:
    def __init__(self, cls_factory=None, error=None):
        self.cls_factory = cls_factory or self._default_cls
        self.error = error
        self.calls = []
        self.was_eval = False
        self.device = None
        self.grad_enabled = []

    @staticmethod
    def _default_cls(smiles):
        return torch.tensor([[float(i + 1)] * 512 for i in range(len(smiles))], dtype=torch.float64)

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        self.was_eval = True
        return self

    def __call__(self, smiles):
        self.calls.append(list(smiles))
        self.grad_enabled.append(torch.is_grad_enabled())
        if self.error:
            raise self.error
        return FakeOutput(self.cls_factory(smiles))


def install_fake(monkeypatch, fake_model):
    loader_calls = []

    def from_pretrained(*args, **kwargs):
        loader_calls.append((args, kwargs))
        return fake_model

    monkeypatch.setattr("mol_embed_service.models.AutoModel.from_pretrained", from_pretrained)
    monkeypatch.setattr(
        "mol_embed_service.models.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: pytest.fail("Mol-JEPA must not load a tokenizer"),
    )
    return loader_calls


def test_mol_jepa_cls_embedding_and_pinned_loader(monkeypatch, tmp_path):
    fake = FakeModel()
    loader_calls = install_fake(monkeypatch, fake)
    output_path = tmp_path / "mol_jepa.npy"
    smiles = ["CCO", "CC", "c1ccccc1", "C=O", "C#N"]

    embed_smiles(smiles, "mol-jepa", str(output_path), batch_size=2, device="cpu")

    assert loader_calls == [
        (
            (MODEL_NAME,),
            {"revision": MODEL_REVISION, "trust_remote_code": True},
        )
    ]
    assert fake.device == torch.device("cpu")
    assert fake.was_eval
    assert fake.grad_enabled == [False, False, False]
    assert fake.calls == [smiles[:2], smiles[2:4], smiles[4:]]

    embeddings = np.load(output_path)
    assert embeddings.shape == (5, 512)
    assert embeddings.dtype == np.float32
    assert np.array_equal(embeddings[0], np.ones(512, dtype=np.float32))
    assert np.array_equal(embeddings[1], np.full(512, 2, dtype=np.float32))
    assert np.array_equal(embeddings[2], np.ones(512, dtype=np.float32))


def test_mol_jepa_embedding_size():
    assert MolJEPAEmbedder.MODEL_NAME == MODEL_NAME
    assert MolJEPAEmbedder.MODEL_REVISION == MODEL_REVISION
    assert MolJEPAEmbedder.EMBEDDING_DIM == 512
    assert get_embedding_size("mol-jepa") == 512


def test_mol_jepa_invalid_rows_preserve_alignment(monkeypatch, tmp_path):
    fake = FakeModel()
    install_fake(monkeypatch, fake)
    smiles = ["", "CCO", "invalid_smiles_that_will_fail", "", "CC", "C"]
    output_path = tmp_path / "invalid.npy"

    embed_smiles(smiles, "mol-jepa", str(output_path), batch_size=2, device="cpu")

    embeddings = np.load(output_path)
    assert embeddings.shape == (len(smiles), 512)
    assert np.array_equal(embeddings[0], np.zeros(512, dtype=np.float32))
    assert np.array_equal(embeddings[2], np.zeros(512, dtype=np.float32))
    assert np.array_equal(embeddings[3], np.zeros(512, dtype=np.float32))
    assert fake.calls == [["CCO"], ["CC", "C"]]
    assert np.count_nonzero(embeddings[1]) == 512
    assert np.count_nonzero(embeddings[4]) == 512


def test_mol_jepa_upstream_error_propagates(monkeypatch, tmp_path):
    fake = FakeModel(error=RuntimeError("upstream failure"))
    install_fake(monkeypatch, fake)
    output_path = tmp_path / "should_not_exist.npy"

    with pytest.raises(RuntimeError, match="upstream failure"):
        embed_smiles(["CCO"], "mol-jepa", str(output_path), device="cpu")

    assert not output_path.exists()


@pytest.mark.parametrize("shape", [(1, 511), (2, 512)])
def test_mol_jepa_rejects_wrong_cls_shape(monkeypatch, tmp_path, shape):
    def wrong_cls(smiles):
        return torch.zeros(shape, dtype=torch.float32)

    fake = FakeModel(cls_factory=wrong_cls)
    install_fake(monkeypatch, fake)
    output_path = tmp_path / "wrong_shape.npy"

    with pytest.raises(ValueError, match="out.cls has shape"):
        embed_smiles(["CCO"], "mol-jepa", str(output_path), device="cpu")

    assert not output_path.exists()
