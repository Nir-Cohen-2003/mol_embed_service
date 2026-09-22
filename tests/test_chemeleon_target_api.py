from pathlib import Path

from mol_embed_service import get_chemeleon_provenance


def test_chemeleon_provenance_is_explicit_and_content_addressed(tmp_path: Path):
    checkpoint = tmp_path / "chemeleon.pt"
    checkpoint.write_bytes(b"checkpoint-v1")
    provenance = get_chemeleon_provenance(checkpoint)
    assert provenance["checkpoint_path"] == str(checkpoint.resolve())
    assert len(provenance["checkpoint_sha256"]) == 64
    assert provenance["semantic_version"] == "chemeleon-mean-aggregation-v1"
    assert provenance["service_version"] == "0.1.0"
