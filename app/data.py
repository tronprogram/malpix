"""Lectura de los manifiestos Dike y los lotes train, val y test."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from app.config import MalPixData
from app.dataset import read_external_rows, read_manifest_rows


class MalPixDataset(Dataset):
    """Dataset binario que lee un manifiesto y el store compartido de PNG."""

    def __init__(
        self,
        rows: list[dict[str, str]],
        data_root: Path,
        transform=None,
    ) -> None:
        self.rows = rows
        self.data_root = data_root
        self.transform = transform if transform is not None else malpix_transform()

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image_path = self.data_root / row["image"]
        with Image.open(image_path) as image:
            image = image.convert("L")
            image = self.transform(image)
        return image, int(row["label"])


MALPIX_MEAN = (0.5,)
MALPIX_STD = (0.5,)


def malpix_transform():
    """Preprocesado único de MalPix: gris, 224x224 y normalización de un canal."""
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=MALPIX_MEAN, std=MALPIX_STD),
        ]
    )


def load_malpix_loaders(system_data: MalPixData):
    """Carga train/val/test desde el manifiesto de una configuración."""
    rows = read_manifest_rows(system_data.config)
    transform = malpix_transform()
    loaders = []
    for split in ("train", "val", "test"):
        split_rows = [row for row in rows if row["split"] == split]
        if not split_rows:
            raise ValueError(f"{system_data.config}: split {split} vacío")
        dataset = MalPixDataset(split_rows, system_data.data_root, transform)
        generator = torch.Generator().manual_seed(system_data.seed)
        loaders.append(
            DataLoader(
                dataset,
                batch_size=system_data.batch_size,
                shuffle=split == "train",
                num_workers=0,
                generator=generator if split == "train" else None,
            )
        )
    return ["benign", "malware"], loaders[0], loaders[1], loaders[2]


def load_external_loader(system_data: MalPixData):
    """Carga el holdout benigno externo; nunca se mezcla con train/val/test."""
    rows = read_external_rows()
    dataset = MalPixDataset(rows, system_data.data_root, malpix_transform())
    return DataLoader(dataset, batch_size=system_data.batch_size, shuffle=False, num_workers=0)


def malpix_class_weights(system_data: MalPixData) -> list[float]:
    """Pesos balanceados calculados solo desde train: N / (2 * n_c)."""
    rows = [row for row in read_manifest_rows(system_data.config) if row["split"] == "train"]
    total = len(rows)
    weights: list[float] = []
    for label in ("0", "1"):
        count = sum(1 for row in rows if row["label"] == label)
        weights.append(total / (2 * count) if count else 0.0)
    return weights
