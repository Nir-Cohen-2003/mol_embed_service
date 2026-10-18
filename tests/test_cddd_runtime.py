import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from mol_embed_service.cddd_backend import CDDDBackend


def test_fresh_process_offline_embedding_never_imports_upstream_package():
    script = r'''\
import sys, types, numpy as np, importlib.machinery
ort = types.ModuleType("onnxruntime")
ort.__spec__ = importlib.machinery.ModuleSpec("onnxruntime", loader=None)
ort.get_available_providers = lambda: ["CPUExecutionProvider"]
ort.SessionOptions = type("Options", (), {})
class Session:
    def get_providers(self): return ["CPUExecutionProvider"]
    def run(self, _names, inputs): return [np.ones((len(inputs["Input/Placeholder:0"]), 512), dtype=np.float32)]
ort.InferenceSession = lambda *args, **kwargs: Session()
sys.modules["onnxruntime"] = ort
from mol_embed_service import cddd_checkpoint
cddd_checkpoint.get_model_path = lambda name="encoder": "/cache/encoder.onnx"
from mol_embed_service.cddd_backend import CDDDBackend
result = CDDDBackend("cpu").embed(["c1ccccc1"], 2)
assert result.shape == (1, 512) and np.all(result == 1)
assert not any(name == "cddd_onnx" or name.startswith("cddd_onnx.") for name in sys.modules)
'''
    env = dict(os.environ, MOL_EMBED_OFFLINE="1")
    subprocess.run([sys.executable, "-c", script], check=True, env=env)


@pytest.mark.skipif(
    not (Path.home() / ".cddd_onnx/models/encoder.onnx").exists(),
    reason="cached CDDD encoder is unavailable",
)
def test_cached_cddd_cpu_encoder_has_meaningful_output():
    backend = CDDDBackend("cpu")
    output = backend.embed(["c1ccccc1", "CN1C=NC2=C1C(=O)N(C)C(=O)N2C", "CCO"], 2)
    assert output.shape == (3, 512)
    assert np.isfinite(output).all()
    assert np.any(output[0] != 0)
    assert np.any(output[1] != 0)
    assert np.all(output[2] == 0)


@pytest.mark.skipif(
    not (Path.home() / ".cddd_onnx/models/encoder.onnx").exists(),
    reason="cached CDDD encoder is unavailable",
)
def test_cached_cddd_cuda_profile_and_cpu_agreement(tmp_path):
    import onnxruntime as ort

    if "CUDAExecutionProvider" not in ort.get_available_providers():
        pytest.skip("CUDAExecutionProvider is unavailable")
    profile_path = tmp_path / "cddd-profile.json"

    def profiled_options():
        options = ort.SessionOptions()
        options.enable_profiling = True
        options.profile_file_prefix = str(profile_path.with_suffix(""))
        return options

    smiles = ["c1ccccc1", "CN1C=NC2=C1C(=O)N(C)C(=O)N2C"]
    cpu = CDDDBackend("cpu").embed(smiles, 2)
    gpu_backend = CDDDBackend("cuda", session_options_factory=profiled_options)
    gpu = gpu_backend.embed(smiles, 2)
    trace = gpu_backend.model.encoder_session.end_profiling()
    with open(trace, encoding="utf-8") as profile_file:
        events = json.load(profile_file)
    compute = [
        event for event in events
        if event.get("cat") == "Node"
        and event.get("args", {}).get("provider") == "CUDAExecutionProvider"
        and "Memcpy" not in event.get("name", "")
    ]
    assert compute, f"no CUDA compute events in {trace}"
    assert np.isfinite(gpu).all() and np.any(gpu != 0)
    np.testing.assert_allclose(gpu, cpu, rtol=1e-3, atol=1e-4)
