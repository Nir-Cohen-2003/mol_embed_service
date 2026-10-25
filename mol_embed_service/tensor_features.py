"""Shared differentiable tensor feature paths for supported backbones."""


def chemberta_features(model, tokenizer, smiles, device):
    inputs = tokenizer(
        smiles, padding=True, truncation=True, max_length=512, return_tensors="pt"
    ).to(device)
    outputs = model(**inputs)
    token_embeddings = outputs.last_hidden_state
    mask = inputs["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
    return (token_embeddings * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


def chemeleon_features(model, featurizer, smiles, device):
    from chemprop.data import BatchMolGraph
    from rdkit import Chem

    mols = []
    for index, value in enumerate(smiles):
        mol = Chem.MolFromSmiles(value) if isinstance(value, str) else value
        if mol is None:
            raise ValueError(f"Invalid SMILES at row {index}: {value!r}")
        mols.append(mol)
    graph = BatchMolGraph([featurizer(mol) for mol in mols])
    graph.to(device=device)
    return model.fingerprint(graph)
