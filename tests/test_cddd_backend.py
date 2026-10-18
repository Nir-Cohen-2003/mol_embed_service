import numpy as np
import pytest

from mol_embed_service.cddd_backend import CDDDBackend


class FakeSession:
    def __init__(self, providers):
        self.providers = [p[0] if isinstance(p, tuple) else p for p in providers]
        self.fallback_disabled = False
        self.cpu_retries = 0

    def get_providers(self):
        return self.providers

    def disable_fallback(self):
        self.fallback_disabled = True

    def run(self, *_args, **_kwargs):
        if not self.fallback_disabled:
            self.cpu_retries += 1
            return [np.ones((1, 512), dtype=np.float32)]
        raise RuntimeError("CUDA execution provider failed")


def fake_modules(monkeypatch, available, created=None, error=None, actual=None):
    created = created if created is not None else []
    ort = type("ORT", (), {})()
    ort.get_available_providers = lambda: available
    ort.SessionOptions = type("Options", (), {})

    def create(path, sess_options, providers):
        if error:
            raise error
        session = FakeSession(actual if actual is not None else providers)
        created.append((path, sess_options, providers, session))
        return session

    ort.InferenceSession = create
    monkeypatch.setitem(__import__("sys").modules, "onnxruntime", ort)
    monkeypatch.setattr("mol_embed_service.cddd_checkpoint.get_model_path", lambda name="encoder": "/cached/encoder.onnx")
    return created


def test_cuda_ep_must_be_available(monkeypatch):
    fake_modules(monkeypatch, ["CPUExecutionProvider"])
    with pytest.raises(RuntimeError, match="CUDAExecutionProvider is unavailable"):
        CDDDBackend("cuda")


def test_cuda_ep_must_attach_and_index_is_forwarded(monkeypatch):
    created = fake_modules(monkeypatch, ["CUDAExecutionProvider", "CPUExecutionProvider"])
    backend = CDDDBackend("cuda:2")
    assert created[0][2][0] == ("CUDAExecutionProvider", {"device_id": 2, "use_tf32": "0"})
    assert backend.model.encoder_session is created[0][3]
    assert created[0][1].intra_op_num_threads == 1
    assert len(created) == 1


def test_cpu_only_provider_and_cuda_construction_errors(monkeypatch):
    created = fake_modules(monkeypatch, ["CPUExecutionProvider"])
    CDDDBackend("cpu")
    assert created[0][2] == ["CPUExecutionProvider"]
    fake_modules(monkeypatch, ["CUDAExecutionProvider"], error=OSError("missing library"))
    with pytest.raises(RuntimeError, match="Failed to create") as exc:
        CDDDBackend("cuda")
    assert isinstance(exc.value.__cause__, OSError)


def test_cuda_session_that_silently_uses_cpu_is_rejected(monkeypatch):
    fake_modules(monkeypatch, ["CUDAExecutionProvider", "CPUExecutionProvider"], actual=["CPUExecutionProvider"])
    with pytest.raises(RuntimeError, match="did not attach CUDAExecutionProvider"):
        CDDDBackend("cuda")


def test_cuda_inference_failure_does_not_retry_on_cpu(monkeypatch):
    created = fake_modules(monkeypatch, ["CUDAExecutionProvider", "CPUExecutionProvider"])
    backend = CDDDBackend("cuda")
    session = created[0][3]
    monkeypatch.setattr("mol_embed_service.cddd_backend.preprocess_smiles", lambda value: value)
    backend.model.seq_to_emb = lambda _smiles: session.run("embedding", {})
    with pytest.raises(RuntimeError, match="inference failed") as exc:
        backend.embed(["CC"], 1)
    assert isinstance(exc.value.__cause__, RuntimeError)
    assert "CUDA execution provider failed" in str(exc.value.__cause__)
    assert session.fallback_disabled and session.cpu_retries == 0


def test_empty_input_and_batch_validation(monkeypatch):
    fake_modules(monkeypatch, ["CPUExecutionProvider"])
    backend = CDDDBackend("cpu")
    assert backend.embed([], 4).shape == (0, 512)
    with pytest.raises(ValueError, match="positive"):
        backend.embed([], 0)


def test_inference_errors_are_not_converted_to_zero(monkeypatch):
    fake_modules(monkeypatch, ["CPUExecutionProvider"])
    backend = CDDDBackend("cpu")
    monkeypatch.setattr("mol_embed_service.cddd_backend.preprocess_smiles", lambda s: s)
    backend.model.seq_to_emb = lambda _smiles: (_ for _ in ()).throw(LookupError("inference broke"))
    with pytest.raises(RuntimeError, match="inference failed") as exc:
        backend.embed(["CC"], 4)
    assert isinstance(exc.value.__cause__, LookupError)


def test_output_shape_and_finiteness_are_checked(monkeypatch):
    fake_modules(monkeypatch, ["CPUExecutionProvider"])
    backend = CDDDBackend("cpu")
    monkeypatch.setattr("mol_embed_service.cddd_backend.preprocess_smiles", lambda s: s)
    backend.model.seq_to_emb = lambda _: np.ones((2, 511), dtype=np.float32)
    with pytest.raises(ValueError, match=r"expected \(1, 512\)"):
        backend.embed(["CC"], 4)
    backend.model.seq_to_emb = lambda _: np.full((2, 512), np.nan, dtype=np.float32)
    with pytest.raises(ValueError, match="nonfinite"):
        backend.embed(["CC"], 4)


def test_rejected_inputs_remain_zero(monkeypatch):
    fake_modules(monkeypatch, ["CPUExecutionProvider"])
    backend = CDDDBackend("cpu")
    monkeypatch.setattr("mol_embed_service.cddd_backend.preprocess_smiles", lambda s: float("nan") if s == "bad" else s)
    backend.model.seq_to_emb = lambda _: np.ones((2, 512), dtype=np.float32)
    out = backend.embed(["CC", "bad"], 2)
    assert np.all(out[0] == 1) and np.all(out[1] == 0)


def test_interleaved_rejections_preserve_order_across_batches(monkeypatch):
    fake_modules(monkeypatch, ["CPUExecutionProvider"])
    backend = CDDDBackend("cpu")

    def preprocess(smiles):
        return float("nan") if smiles in {"bad", ""} else smiles

    monkeypatch.setattr("mol_embed_service.cddd_backend.preprocess_smiles", preprocess)
    calls = []

    def encode(smiles):
        calls.append(smiles[1:])
        return np.asarray([[0] * 512] + [[int(value[-1])] * 512 for value in smiles[1:]], dtype=np.float32)

    backend.model.seq_to_emb = encode
    output = backend.embed(["mol1", "bad", "mol2", "", "mol3"], 2)
    assert calls == [["mol1", "mol2"], ["mol3"]]
    for index, expected in enumerate([1, 0, 2, 0, 3]):
        assert np.all(output[index] == expected)
