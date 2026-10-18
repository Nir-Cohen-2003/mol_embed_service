"""Download-free contract tests for optional embedding adapters."""

import importlib.abc
import os
import subprocess
import sys
import threading
import types

import numpy as np
import pytest
import torch

from mol_embed_service import embedder as public
from mol_embed_service import optional_models as optional


class _BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"monroe", "minimol", "grasp"}:
            raise ImportError("blocked optional dependency")


def test_package_import_does_not_import_optional_upstreams():
    source_root = os.path.dirname(os.path.dirname(__file__))
    script = "import importlib.abc,sys; exec('class Block(importlib.abc.MetaPathFinder):\\n def find_spec(self, fullname, path=None, target=None):\\n  if fullname.split(\".\")[0] in {\"monroe\",\"minimol\",\"grasp\"}: raise ImportError(\"blocked\")\\n  return None'); sys.meta_path.insert(0,Block()); import mol_embed_service; from mol_embed_service import get_embedding_size; assert get_embedding_size('grasp') == 768"
    # Use the active test module's blocker in a fresh process without model downloads.
    subprocess.run([sys.executable, "-c", script], cwd=source_root, text=True, capture_output=True, check=True)


def test_registry_dimensions_and_unknown_model():
    assert [public.get_embedding_size(name) for name in ("monroe", "minimol", "grasp")] == [720, 512, 768]
    with pytest.raises(ValueError, match="monroe.*grasp"):
        public.get_embedding_size("unknown")


def _monroe(tmp_path, monkeypatch, output):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "weights.pt").write_bytes(b"trusted mock")
    calls = {}
    monroe = types.ModuleType("monroe")
    monroe.__path__ = []
    model = types.ModuleType("monroe.model")
    model.__path__ = []
    ckpt = types.ModuleType("monroe.model.ckpt")
    ckpt.load_ckpt = lambda path: calls.setdefault("load", path) or object()
    eval_pkg = types.ModuleType("monroe.eval")
    eval_pkg.__path__ = []
    embed = types.ModuleType("monroe.eval.embed")
    def helper(smiles, encoder, **kwargs):
        calls["helper"] = (smiles, encoder, kwargs)
        return output
    embed.embed_smiles = helper
    for name, module in (("monroe", monroe), ("monroe.model", model), ("monroe.model.ckpt", ckpt), ("monroe.eval", eval_pkg), ("monroe.eval.embed", embed)):
        monkeypatch.setitem(sys.modules, name, module)
    return calls


def test_monroe_configuration_and_reconstructs_duplicates_in_input_order(tmp_path, monkeypatch):
    rows = ["CCO", "CC", "CCO"]
    vectors = {"CC": np.full(720, 2), "CCO": np.full(720, 1)}
    calls = _monroe(tmp_path, monkeypatch, vectors)
    monkeypatch.setenv("MONROE_CHECKPOINT_DIR", str(tmp_path))
    model = optional.MonroeEmbedder(device="cpu")
    result = model.embed(rows, 2)
    assert calls["load"] == str(tmp_path)
    assert calls["helper"][0] == rows
    assert calls["helper"][2] == {"device": "cpu", "batch_size": 2, "n_workers": 1}
    assert result.shape == (3, 720) and result.dtype == np.float32
    assert np.array_equal(result[:, 0], [1, 2, 1])


def test_monroe_uses_cache_and_offline_miss_is_actionable(tmp_path, monkeypatch):
    monkeypatch.delenv("MONROE_CHECKPOINT_DIR", raising=False)
    monkeypatch.setenv("MOL_EMBED_OFFLINE", "1")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    with pytest.raises(RuntimeError, match="MOL_EMBED_OFFLINE=1"):
        optional.MonroeEmbedder()
    monkeypatch.setenv("MONROE_CHECKPOINT_DIR", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="config.json"):
        optional.MonroeEmbedder()


def test_monroe_checkpoint_override_missing_lfs_and_bad_rows(tmp_path, monkeypatch):
    monkeypatch.setenv("MONROE_CHECKPOINT_DIR", str(tmp_path / "wrong"))
    (tmp_path / "config.json").write_text("{}")
    weights = tmp_path / "weights.pt"
    _monroe(tmp_path, monkeypatch, {})
    weights.write_text("version https://git-lfs.github.com/spec/v1\n")
    with pytest.raises(ValueError, match="Git LFS pointer"):
        optional.MonroeEmbedder(checkpoint_dir=tmp_path)
    weights.write_bytes(b"actual mock")
    model = optional.MonroeEmbedder(checkpoint_dir=tmp_path)
    with pytest.raises(ValueError, match="missing embedding for row 0"):
        model.embed(["CCO"], 1)


def test_minimol_hidden_cuda_cpu_fallback_chunking_and_singleton(monkeypatch):
    optional._minimol_backend = None
    monkeypatch.setattr(optional.torch.cuda, "is_available", lambda: False)
    calls = []
    class Backend:
        def __init__(self, batch_size):
            calls.append(("init", batch_size))
        def __call__(self, batch):
            calls.append(("call", list(batch), torch.is_grad_enabled()))
            return [torch.ones(512) * (1 if value == "CCO" else 2) for value in batch]
    monkeypatch.setattr(optional, "create_minimol", lambda batch_size: Backend(batch_size))
    rows = ["CCO", "CC", "CCO"]
    result = optional.MiniMolEmbedder("cuda").embed(rows, 2)
    optional.MiniMolEmbedder("cpu").embed(rows, 1)
    assert calls.count(("init", 100)) == 1
    assert calls[1:3] == [("call", rows[:2], False), ("call", rows[2:], False)]
    assert result.shape == (3, 512) and result.dtype == np.float32
    assert result[0, 0] == result[2, 0]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_minimol_cuda_visible_rejects_even_cached_backend(monkeypatch, device):
    optional._minimol_backend = object()
    monkeypatch.setattr(optional.torch.cuda, "is_available", lambda: True)
    with pytest.raises(ValueError, match="CUDA is visible"):
        optional.MiniMolEmbedder(device)
    adapter = object.__new__(optional.MiniMolEmbedder)
    with pytest.raises(ValueError, match="CUDA is visible"):
        adapter.embed(["CCO"], 1)


def test_minimol_unsupported_device_and_concurrent_singleton(monkeypatch):
    optional._minimol_backend = None
    monkeypatch.setattr(optional.torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="unindexed"):
        optional.MiniMolEmbedder("cuda:0")
    initialized = []
    class Backend:
        def __init__(self, batch_size):
            initialized.append(batch_size)
        def __call__(self, smiles):
            return [torch.zeros(512) for _ in smiles]
    monkeypatch.setattr(optional, "create_minimol", lambda batch_size: Backend(batch_size))
    threads = [threading.Thread(target=lambda: optional.MiniMolEmbedder("cpu").embed(["CCO"], 1)) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert initialized == [100]


def test_missing_optional_dependencies_have_actionable_errors(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "weights.pt").write_bytes(b"weights")
    original_import = optional.importlib.import_module
    def without_optional(name, *args, **kwargs):
        if name.startswith(("monroe", "minimol", "grasp")):
            raise ModuleNotFoundError(name)
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(optional.importlib, "import_module", without_optional)
    with pytest.raises(ImportError, match="optional.*README"):
        optional.MonroeEmbedder(checkpoint_dir=tmp_path)
    monkeypatch.setattr(optional.torch.cuda, "is_available", lambda: False)
    optional._minimol_backend = None
    monkeypatch.setattr(optional, "create_minimol", lambda **kwargs: (_ for _ in ()).throw(ImportError("missing")))
    with pytest.raises(ImportError, match="optional.*README"):
        optional.MiniMolEmbedder("cpu").embed(["CCO"], 1)
    with pytest.raises(ImportError, match="optional.*README"):
        optional.GRASPEmbedder("cpu")


def test_grasp_pinned_loader_and_encode(monkeypatch, tmp_path):
    calls = {}
    monkeypatch.setenv("MOL_EMBED_OFFLINE", "1")
    hub = types.ModuleType("huggingface_hub")
    def download(**kwargs):
        calls.setdefault("downloads", []).append(kwargs)
        return str(tmp_path / kwargs["filename"])
    hub.hf_hub_download = download
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    class Encoder:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            calls["load"] = (args, kwargs)
            return cls()
        def encode(self, smiles, batch_size):
            calls["encode"] = (smiles, batch_size)
            return np.ones((len(smiles), 768), dtype=np.float64)
    package = types.ModuleType("grasp")
    package.GRASPEncoder = Encoder
    monkeypatch.setitem(sys.modules, "grasp", package)
    model = optional.GRASPEmbedder("cpu")
    result = model.embed(["CCO", "CCO", "CC"], 2)
    assert calls["load"] == ((str(tmp_path.resolve()),), {"device": "cpu"})
    assert [call["filename"] for call in calls["downloads"]] == ["model.safetensors", "config.json", "vocab.json"]
    assert all(call["repo_id"] == "caithmac/GRASP" and call["revision"] == optional._GRASP_REVISION for call in calls["downloads"])
    assert all(call["local_files_only"] is True for call in calls["downloads"])
    assert calls["encode"] == (["CCO", "CCO", "CC"], 2)
    assert result.shape == (3, 768) and result.dtype == np.float32


@pytest.mark.parametrize("bad", [[], ["CCO", "not smiles"], "CCO"])
def test_new_adapters_validate_inputs(bad):
    for name, width in (("Monroe", 720), ("GRASP", 768)):
        with pytest.raises((TypeError, ValueError)):
            optional._validate_inputs(name, bad, 1)
    with pytest.raises((TypeError, ValueError)):
        optional._validate_inputs("MiniMol", bad, 1)
    with pytest.raises(ValueError, match="positive integer"):
        optional._validate_inputs("GRASP", ["CCO"], 0)


@pytest.mark.parametrize("matrix", [np.ones((2, 4)), np.array([[np.nan]]), np.array([[np.inf]]), np.array([[1e100]])])
def test_output_validation_rejects_bad_shapes_and_nonfinite(matrix):
    with pytest.raises(ValueError):
        optional._validated_matrix("test", matrix, 1, 1)


@pytest.mark.parametrize("name,width", [("monroe", 720), ("minimol", 512), ("grasp", 768)])
def test_public_dispatch_saves_arrays_and_forwards_options(tmp_path, monkeypatch, name, width):
    calls = {}

    class Fake:
        def __init__(self, device):
            calls["device"] = device

        def embed(self, smiles, batch_size):
            calls["embed"] = (smiles, batch_size)
            return np.ones((len(smiles), width), dtype=np.float32)

    embedder_name = {"monroe": "MonroeEmbedder", "minimol": "MiniMolEmbedder", "grasp": "GRASPEmbedder"}[name]
    monkeypatch.setattr(public, embedder_name, Fake)
    output = tmp_path / f"{name}.npy"
    rows = ["CCO", "CC"]
    public.embed_smiles(rows, name, str(output), batch_size=7, device="cpu")
    saved = np.load(output)
    assert saved.shape == (2, width) and saved.dtype == np.float32
    assert calls == {"device": "cpu", "embed": (rows, 7)}


@pytest.mark.parametrize("name", ["monroe", "minimol", "grasp"])
@pytest.mark.parametrize("existing", [False, True])
def test_public_dispatch_failure_preserves_output(tmp_path, monkeypatch, name, existing):
    class Broken:
        def __init__(self, device):
            pass

        def embed(self, smiles, batch_size):
            raise RuntimeError("expected failure")

    embedder_name = {"monroe": "MonroeEmbedder", "minimol": "MiniMolEmbedder", "grasp": "GRASPEmbedder"}[name]
    monkeypatch.setattr(public, embedder_name, Broken)
    output = tmp_path / f"{name}.npy"
    original = b"existing output contents"
    if existing:
        output.write_bytes(original)
    with pytest.raises(RuntimeError, match="expected failure"):
        public.embed_smiles(["CCO"], name, str(output), batch_size=3, device="cpu")
    if existing:
        assert output.read_bytes() == original
    else:
        assert not output.exists()


def test_minimol_compensating_chunk_row_counts_fail_before_public_save(tmp_path, monkeypatch):
    optional._minimol_backend = None
    monkeypatch.setattr(optional.torch.cuda, "is_available", lambda: False)

    class Backend:
        def __call__(self, batch):
            count = 1 if batch[0] == "CCO" else 3
            return [torch.ones(512) for _ in range(count)]

    monkeypatch.setattr(optional, "_minimol_backend", Backend())
    output = tmp_path / "minimol.npy"
    original = b"existing output contents"
    output.write_bytes(original)
    with pytest.raises(ValueError, match=r"chunk starting at 0.*expected \(2, 512\)"):
        public.embed_smiles(["CCO", "CC", "CCC", "CCCC"], "minimol", str(output), batch_size=2, device="cpu")
    assert output.read_bytes() == original
