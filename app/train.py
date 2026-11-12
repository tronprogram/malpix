"""Bucle de entrenamiento y lectura del checkpoint."""

from __future__ import annotations

from pathlib import Path

import torch
from rich import print
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader

from app.config import MalPixData
from app.data import load_malpix_loaders, malpix_class_weights
from app.model import build_model, device, freeze_trained_weights


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    dev: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    class_weights: torch.Tensor | None = None,
) -> tuple[float, list[int], list[int]]:
    """Recorre un loader una vez.

    Actualiza los pesos si recibe un optimizador. Si no, solo evalúa.
    """
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    total = 0.0
    seen = 0
    y_true: list[int] = []
    y_pred: list[int] = []

    if optimizer is not None:
        model.train()
        for images, labels in loader:
            images = images.to(dev)
            labels = labels.to(dev)
            optimizer.zero_grad()
            logits = model(images)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            total += float(loss.item()) * labels.size(0)
            seen += int(labels.size(0))
            y_true.extend(labels.detach().cpu().tolist())
            y_pred.extend(logits.argmax(dim=1).detach().cpu().tolist())
    else:
        model.eval()
        for images, labels in loader:
            images = images.to(dev)
            labels = labels.to(dev)
            with torch.no_grad():
                logits = model(images)
                loss = loss_fn(logits, labels)

            total += float(loss.item()) * labels.size(0)
            seen += int(labels.size(0))
            y_true.extend(labels.detach().cpu().tolist())
            y_pred.extend(logits.argmax(dim=1).detach().cpu().tolist())

    return (total / seen if seen else 0.0), y_true, y_pred


def save_if_better(
    path: Path,
    model: nn.Module,
    score: float,
    best: float,
    classes: list[str],
    benign: str,
):
    """Guarda el modelo cuando ``score`` supera a ``best``."""
    if score <= best:
        return best, False

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "score": score,
            "state": model.state_dict(),
            "classes": list(classes),
            "benign": benign,
        },
        path,
    )
    return score, True


def load_checkpoint(path: Path) -> dict:
    """Carga el diccionario escrito por ``save_if_better``."""
    if not path.is_file():
        raise FileNotFoundError(path)

    return torch.load(path, map_location="cpu", weights_only=False)


def predict_probs(
    model: nn.Module, loader: DataLoader, dev: torch.device
) -> tuple[list[int], list[float]]:
    """Devuelve etiquetas y probabilidad de malware para un loader."""
    model.eval()
    y_true: list[int] = []
    y_prob: list[float] = []
    for images, labels in loader:
        images = images.to(dev)
        with torch.no_grad():
            logits = model(images)
            probs = torch.softmax(logits, dim=1)[:, 1]
        y_true.extend(int(label) for label in labels.tolist())
        y_prob.extend(float(prob) for prob in probs.detach().cpu().tolist())
    return y_true, y_prob


def run(system_data: MalPixData) -> None:
    """Entrena una configuración Dike y conserva el mejor macro-F1 de validación."""
    classes, train_loader, val_loader, _test_loader = load_malpix_loaders(system_data)
    weights = malpix_class_weights(system_data)
    if any(weight <= 0 for weight in weights):
        raise ValueError("las clases de train deben tener muestras para calcular pesos")

    if len(classes) != 2:
        raise ValueError("MalPix debe tener exactamente dos clases")
    dev = device()
    model = build_model(pretrained=True, seed=system_data.seed).to(dev)
    weight_tensor = torch.tensor(weights, dtype=torch.float32, device=dev)
    freeze_trained_weights(model, True)
    trainable = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=system_data.lr)
    best = float("-inf")
    path = system_data.models_dir / "best.pt"

    print({"config": system_data.config, "class_weights": weights})
    for epoch in range(1, system_data.epochs + 1):
        if epoch == system_data.freeze_epochs + 1:
            freeze_trained_weights(model, False)
            trainable = [param for param in model.parameters() if param.requires_grad]
            optimizer = torch.optim.Adam(trainable, lr=system_data.lr)

        train_loss, _, _ = run_epoch(
            model, train_loader, dev, optimizer, class_weights=weight_tensor
        )
        val_loss, y_true, y_pred = run_epoch(model, val_loader, dev)
        score = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        best, saved = save_if_better(
            path, model, score, best, classes, "benign"
        )
        print(
            {
                "época": epoch,
                "train": round(train_loss, 4),
                "val": round(val_loss, 4),
                "macro-F1": round(score, 4),
                "guardado": saved,
            }
        )
