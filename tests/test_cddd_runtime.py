import json
from pathlib import Path

import numpy as np
import pytest

from mol_embed_service.cddd_backend import CDDDBackend


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
