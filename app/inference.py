"""Predicción de una imagen con el checkpoint guardado."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image
from rich import print

from app.config import MalPixData
from app.data import malpix_transform
from app.model import build_model, device
from app.train import load_checkpoint


def run(system_data: MalPixData, image_path: Path) -> str:
    """Clasifica una imagen con el checkpoint binario de una configuración."""
    image_path = Path(image_path)
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    payload = load_checkpoint(system_data.models_dir / "best.pt")
    classes = [str(name) for name in payload["classes"]]
    if classes != ["benign", "malware"]:
        raise SystemExit(f"clases del checkpoint inesperadas: {classes}")

    dev = device()
    model = build_model(pretrained=False).to(dev)
    model.load_state_dict(payload["state"])
    model.eval()
    with Image.open(image_path) as image:
        tensor = malpix_transform()(image.convert("L"))
    with torch.no_grad():
        logits = model(tensor.unsqueeze(0).to(dev))
        probabilities = torch.softmax(logits, dim=1)[0]
    threshold = 0.5
    metrics_path = system_data.figures_dir / "metrics.json"
    if metrics_path.is_file():
        payload_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        threshold = float(
            payload_metrics.get("threshold_selection", {}).get(
                "selected_threshold", threshold
            )
        )
    malware_probability = float(probabilities[1].item())
    label = "malware" if malware_probability >= threshold else "benign"
    print(
        {
            "imagen": image_path.name,
            "clase": label,
            "probabilidad_malware": round(malware_probability, 6),
            "umbral": threshold,
            "config": system_data.config,
        }
    )
    return label
