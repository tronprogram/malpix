"""Comandos para preparar, entrenar y evaluar el dataset Dike."""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

if __name__ == "__main__" and __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import typer

from app.config import MALPIX_CONFIGS, MalPixData

app = typer.Typer(no_args_is_help=True)
config_option = typer.Option(
    "original",
    "--config",
    help=f"Una de: {', '.join(MALPIX_CONFIGS)}.",
)


@app.command("prepare-dataset")
def prepare_dataset(
    collector_output: Path | None = typer.Option(
        None,
        "--collector-output",
        help="Salida del colector benigno; solo necesaria la primera vez.",
    ),
    seed: int = typer.Option(20261110, "--seed"),
) -> None:
    """Crea el store compartido, el split Dike y los manifiestos."""
    from app.dataset import prepare_dataset as migrate

    summary = migrate(collector_output=collector_output, seed=seed)
    typer.echo(json.dumps(summary, indent=2, ensure_ascii=False))


@app.command("validate-dataset")
def validate_dataset_command() -> None:
    """Valida paths, imágenes, hashes, leakage y conteos."""
    from app.dataset import validate_dataset

    result = validate_dataset()
    typer.echo(json.dumps(result, indent=2, ensure_ascii=False))
    if not result["ok"]:
        raise typer.Exit(code=1)


@app.command()
def train(
    epochs: int | None = typer.Option(None),
    batch_size: int | None = typer.Option(None),
    config: str = config_option,
) -> None:
    """Entrena una configuración Dike y guarda el mejor macro-F1 de validación."""
    system_data = MalPixData.load(config)
    if epochs is not None:
        system_data = replace(system_data, epochs=epochs)
    if batch_size is not None:
        system_data = replace(system_data, batch_size=batch_size)
    from app.train import run

    run(system_data)


@app.command()
def evaluate(
    config: str = config_option,
    external: bool = typer.Option(
        False, "--external", help="Evalúa también el holdout benigno externo."
    ),
) -> None:
    """Evalúa el checkpoint sobre el test Dike y, opcionalmente, el holdout externo."""
    from app.evaluate import run

    run(MalPixData.load(config), include_external=external)


@app.command()
def infer(
    image: Path,
    config: str = config_option,
) -> None:
    """Predice benigno o malware para una imagen en escala de grises."""
    from app.inference import run

    run(MalPixData.load(config), image)


if __name__ == "__main__":
    app()
