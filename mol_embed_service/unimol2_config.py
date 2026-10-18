"""Pinned Uni-Mol2 model-zoo metadata (no neural or network imports)."""

from dataclasses import dataclass
from types import MappingProxyType

HF_REVISION = "bf37d253c0f5ac61f468137eb00059978a104ff0"
SOURCE_REVISION = "90f52c41299a1a582da0f9765e9f87aa21faa16a"


@dataclass(frozen=True)
class UniMol2Config:
    name: str
    directory: str
    layers: int
    width: int
    heads: int
    bytes: int
    sha256: str
    pair_width: int = 512
    pair_hidden: int = 64
    kernels: int = 128

    @property
    def output_dim(self):
        return self.width

    @property
    def url(self):
        return f"https://huggingface.co/dptech/Uni-Mol2/resolve/{HF_REVISION}/modelzoo/{self.directory}/checkpoint.pt"


CONFIGS = MappingProxyType(
    {
        c.name: c
        for c in (
            UniMol2Config(
                "unimol2-84M",
                "84M",
                12,
                768,
                48,
                336632021,
                "5b9241630f1cf0b173fb06d1e76096e5daf5f91c3e51b066ba69524dafe60e35",
            ),
            UniMol2Config(
                "unimol2-164M",
                "164M",
                24,
                768,
                48,
                659412589,
                "26cd6b8d5381c30af40c66828d76d6b7a01f2b3ebaea00ec48e2e002ccfd28a6",
            ),
            UniMol2Config(
                "unimol2-310M",
                "310M",
                32,
                1024,
                64,
                1240761111,
                "695c6b1af0e006cc58bc2064c41f5ba7c2af7e3bb5d54074110584f77058e90b",
            ),
            UniMol2Config(
                "unimol2-570M",
                "570M",
                32,
                1536,
                96,
                2281364375,
                "bd487e399e9132dfa7adcc30c32fcbce96b1e20ee0373b80329b0aa98a99cf9d",
            ),
            UniMol2Config(
                "unimol2-1.1B",
                "1.1B",
                64,
                1536,
                96,
                4517690299,
                "48e1ba230119a5f286b81dd71bec4889aebebffa03e66d9241f3ceb7bede9a5e",
            ),
        )
    }
)
