"""Configuración de las particiones Dike."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MALPIX_CONFIGS = ("original", "signed", "mixed")
MALPIX_SEED = 20261110


@dataclass(frozen=True)
class MalPixData:
    """Rutas de una configuración binaria MalPix/Dike."""

    config: str
    data_root: Path
    models_dir: Path
    figures_dir: Path
    seed: int = MALPIX_SEED
    batch_size: int = 32
    epochs: int = 8
    freeze_epochs: int = 1
    lr: float = 1e-4

    @property
    def manifest_path(self) -> Path:
        return self.data_root / "manifests" / f"{self.config}.csv"

    @property
    def images_dir(self) -> Path:
        return self.data_root / "images"

    @property
    def external_manifest_path(self) -> Path:
        return self.data_root / "manifests" / "external_benign.csv"

    @classmethod
    def load(cls, config: str) -> "MalPixData":
        """Construye una configuración Dike: ``original``, ``signed`` o ``mixed``."""
        if config not in MALPIX_CONFIGS:
            raise ValueError(
                f"configuración MalPix desconocida: {config!r}; "
                f"usa una de {', '.join(MALPIX_CONFIGS)}"
            )
        return cls(
            config=config,
            data_root=ROOT / "datasets",
            models_dir=ROOT / "out" / "models" / "malpix" / config,
            figures_dir=ROOT / "out" / "figures" / "malpix" / config,
        )
