import numpy as np

from mol_embed_service import cddd_inference
from mol_embed_service.cddd_inference import VOCABULARY, make_inputs, preprocess_smiles, tokenize


def test_complete_model_vocabulary_and_special_ids():
    assert len(VOCABULARY) == 40
    assert VOCABULARY["<s>"] == 39 and VOCABULARY["</s>"] == 0
    tokens = tokenize("#%()+-0123456789:=@BCBrClFHI NOPS[]cionps")
    assert tokens == [1, 2, 4, 3, 5, 6, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15,
                      17, 18, 19, 21, 20, 38, 37, 22, 24, 23, 26, 25, 27, 28,
                      29, 30, 31, 32, 33, 34, 35, 36]
    assert tokenize("Cx?") == [20]


def test_ring_tokens_and_padding_have_exact_ids_and_dtypes():
    inputs, lengths = make_inputs(["c1ccccc1", "ClBr"])
    np.testing.assert_array_equal(inputs, [[39, 31, 7, 31, 31, 31, 31, 31, 7, 0],
                                          [39, 37, 38, 0, 0, 0, 0, 0, 0, 0]])
    np.testing.assert_array_equal(lengths, [10, 4])
    assert inputs.dtype == np.int32 and lengths.dtype == np.int32


def test_preprocess_salt_stereo_and_applicability_filter():
    assert preprocess_smiles("N[C@@H](C)C(=O)O.[Na+]") == "CC(N)C(=O)O"
    assert preprocess_smiles("N[C@@H](C)C(=O)O") == preprocess_smiles("N[C@H](C)C(=O)O")
    assert np.isnan(preprocess_smiles("[Na+]"))
    assert np.isnan(preprocess_smiles("C"))
    assert np.isnan(preprocess_smiles("not-a-smiles"))


def test_applicability_boundaries_are_strict(monkeypatch):
    cases = [
        ("MolLogP", -5), ("MolLogP", 7),
        ("MolWt", 12), ("MolWt", 600),
        ("HeavyAtomCount", 3), ("HeavyAtomCount", 50),
    ]
    for descriptor, boundary in cases:
        monkeypatch.setattr(cddd_inference.Descriptors, descriptor, lambda _mol, value=boundary: value)
        assert np.isnan(preprocess_smiles("CCCC"))
