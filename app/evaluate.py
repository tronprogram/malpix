"""Métricas de test y equidad."""

from __future__ import annotations

import json

import numpy as np
from rich import print
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from app.config import MalPixData
from app.data import load_external_loader, load_malpix_loaders
from app.model import build_model, device
from app.train import load_checkpoint, predict_probs


def as_float(value: object) -> float:
    """Convierte un valor de métrica a ``float``."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError("se esperaba un número")
    return float(value)


def benign_false_positive_rate(y_true: list[int], y_pred: list[int], benign: int) -> float:
    """Devuelve la fracción de benignos predichos como malware."""
    benign_count = 0
    marked_malware = 0

    for actual, pred in zip(y_true, y_pred):
        if actual != benign:
            continue
        benign_count += 1
        if pred != benign:
            marked_malware += 1

    if benign_count == 0:
        return 0.0
    return marked_malware / benign_count


def malware_miss_rate(y_true: list[int], y_pred: list[int], benign: int) -> float:
    """Devuelve la fracción de malware predicho como benigno."""
    malware_count = 0
    marked_benign = 0

    for actual, pred in zip(y_true, y_pred):
        if actual == benign:
            continue
        malware_count += 1
        if pred == benign:
            marked_benign += 1

    if malware_count == 0:
        return 0.0
    return marked_benign / malware_count


def binary_metrics(
    y_true: list[int], y_prob: list[float], threshold: float
) -> dict[str, object]:
    """Métricas binarias idénticas para todas las configuraciones."""
    y_pred = [1 if prob >= threshold else 0 for prob in y_prob]
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1],
        average=None,
        zero_division=0,
    )
    precision_scores = np.asarray(precision, dtype=float)
    recall_scores = np.asarray(recall, dtype=float)
    f1_scores = np.asarray(f1, dtype=float)
    support_scores = np.asarray(support, dtype=float)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    result: dict[str, object] = {
        "threshold": float(threshold),
        "count": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class": [
            {
                "label": "benign" if index == 0 else "malware",
                "precision": float(precision_scores[index]),
                "recall": float(recall_scores[index]),
                "f1": float(f1_scores[index]),
                "support": int(support_scores[index]),
            }
            for index in range(2)
        ],
        "benign_false_positive_rate": benign_false_positive_rate(y_true, y_pred, 0),
        "malware_miss_rate": malware_miss_rate(y_true, y_pred, 0),
        "confusion_matrix": {
            "labels": ["benign", "malware"],
            "rows_actual_columns_predicted": matrix.tolist(),
        },
    }
    if len(set(y_true)) == 2:
        result["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    return result


def select_threshold(
    y_true: list[int], y_prob: list[float]
) -> tuple[float, dict[str, object]]:
    """Elige el umbral solo con validación.

    Criterio: máximo balanced accuracy; desempate por macro-F1, luego por
    cercanía a 0.5 y finalmente por el umbral más bajo.
    """
    if set(y_true) != {0, 1}:
        raise ValueError("la validación debe contener benignos y malware")
    best_key = None
    best_threshold = 0.5
    best_metrics: dict[str, object] = {}
    for threshold in np.round(np.arange(0.05, 0.951, 0.01), 2):
        metrics = binary_metrics(y_true, y_prob, float(threshold))
        key = (
            as_float(metrics["balanced_accuracy"]),
            as_float(metrics["macro_f1"]),
            -abs(float(threshold) - 0.5),
            -float(threshold),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = float(threshold)
            best_metrics = metrics
    return best_threshold, best_metrics


def run(system_data: MalPixData, include_external: bool = False) -> dict[str, object]:
    """Evalúa test y, opcionalmente, el holdout benigno externo."""
    payload = load_checkpoint(system_data.models_dir / "best.pt")
    classes = [str(name) for name in payload["classes"]]
    if classes != ["benign", "malware"]:
        raise SystemExit(f"clases del checkpoint inesperadas: {classes}")

    _classes, _train_loader, val_loader, test_loader = load_malpix_loaders(system_data)
    dev = device()
    model = build_model(pretrained=False).to(dev)
    model.load_state_dict(payload["state"])

    y_val, prob_val = predict_probs(model, val_loader, dev)
    threshold, val_metrics = select_threshold(y_val, prob_val)
    y_test, prob_test = predict_probs(model, test_loader, dev)
    test_metrics = binary_metrics(y_test, prob_test, threshold)

    summary: dict[str, object] = {
        "config": system_data.config,
        "classes": classes,
        "threshold_selection": {
            "method": "max balanced accuracy on validation; tie-break macro-F1, closeness to 0.5, lower threshold",
            "selected_threshold": threshold,
            "validation_metrics": val_metrics,
        },
        "test_metrics": test_metrics,
    }
    if include_external:
        external_loader = load_external_loader(system_data)
        y_external, prob_external = predict_probs(model, external_loader, dev)
        summary["external_benign"] = {
            "count": len(y_external),
            "threshold": threshold,
            "false_positive_rate": (
                sum(1 for prob in prob_external if prob >= threshold) / len(prob_external)
                if prob_external
                else 0.0
            ),
            "mean_malware_probability": (
                float(np.mean(prob_external)) if prob_external else 0.0
            ),
        }

    dest = system_data.figures_dir / "metrics.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        {
            "config": system_data.config,
            "threshold": round(threshold, 2),
            "test_macro-F1": round(as_float(test_metrics["macro_f1"]), 4),
            "test_balanced_accuracy": round(
                as_float(test_metrics["balanced_accuracy"]), 4
            ),
        }
    )
    return summary
