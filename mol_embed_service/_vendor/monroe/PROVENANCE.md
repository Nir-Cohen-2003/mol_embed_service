# Monroe inference vendor provenance

Derived from https://github.com/blazejba/Monroe at commit `57238edfffea03808abe761a00cd9a75fa41bb95`.

Copied inference implementation: model/constants.py, model/featurizer.py, model/grit.py, eval/embed.py, preprocessing graph construction functions, and assets/periodic_table.csv. Upstream LICENSE and NOTICE are retained in full. Training, benchmark, cache, and CLI code are excluded.

Local adaptations: imports and resource package names use the private `mol_embed_service._vendor.monroe` namespace; the graph module contains only `build_mol` and `_build_from_smiles_fallback`; featurization runs serially and rejects worker counts other than one, with no process pool, process-global tqdm override, or SIGALRM handler. Consequently this path has no hard wall-clock featurization timeout. The checkpoint loader uses safe CPU tensor deserialization and validates the complete encoder state while ignoring only the known `decoders.` and `dataset_heads.` checkpoint components.
