"""Migración, manifiestos, splits y validación del layout MalPix/Dike."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

from PIL import Image

from app.config import MALPIX_CONFIGS, MALPIX_SEED, MalPixData, ROOT

MANIFEST_COLUMNS = [
    "sha256",
    "label",
    "format",
    "size_bytes",
    "image",
    "split",
    "source",
    "family_key",
]
SPLIT_COLUMNS = ["sha256", "label", "format", "split"]
CLASS_BALANCE_COLUMNS = [
    "config",
    "split",
    "label",
    "class_name",
    "count",
    "class_weight",
]
APPROVAL_COLUMNS = ["sha256", "approved", "reason", "reviewed_by", "reviewed_at"]
UNSIGNED_COLUMNS = [
    "sha256",
    "label",
    "format",
    "size_bytes",
    "family_key",
    "original_path",
    "app_identity",
    "vendor",
    "category",
]
SUPPLEMENT_COLUMNS = [
    "sha256",
    "label",
    "format",
    "size_bytes",
    "family_key",
    "source_image",
]
CLASS_NAMES = {0: "benign", 1: "malware"}
SPLIT_ORDER = {"train": 0, "val": 1, "test": 2, "external_test": 3}
WIDTH = 256
HEIGHT = 224


def dataset_root() -> Path:
    return ROOT / "datasets"


def dike_manifest_path() -> Path:
    return dataset_root() / "dike" / "manifest.csv"


def dike_images_dir() -> Path:
    return dataset_root() / "dike" / "images"


def images_dir() -> Path:
    return dataset_root() / "images"


def manifests_dir() -> Path:
    return dataset_root() / "manifests"


def splits_dir() -> Path:
    return dataset_root() / "splits"


def supplemental_dir() -> Path:
    return dataset_root() / "supplemental"


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv_atomic(
    path: Path, rows: Iterable[Mapping[str, object]], columns: list[str]
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    count = 0
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key, "") for key in columns})
                count += 1
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return count


def _write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _stable_digest(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8", "replace")).hexdigest()


def _round_half_up(value: float) -> int:
    return int(math.floor(value + 0.5))


def _is_sha256(value: str) -> bool:
    value = (value or "").strip().lower()
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _validate_png(path: Path) -> tuple[bool, str]:
    try:
        with Image.open(path) as image:
            if image.format != "PNG":
                return False, f"format={image.format}"
            if image.mode != "L":
                return False, f"mode={image.mode}"
            if image.size != (HEIGHT, HEIGHT):
                return False, f"size={image.size}"
            image.load()
        return True, ""
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def _link_or_copy(source: Path, destination: Path) -> str:
    """Link a PNG without creating symlinks; copy only if linking is unavailable."""
    if not source.is_file():
        raise FileNotFoundError(source)
    if destination.exists():
        valid, reason = _validate_png(destination)
        if not valid:
            raise ValueError(f"imagen compartida inválida: {destination} ({reason})")
        return "existing"
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
        mode = "linked"
    except OSError:
        shutil.copy2(source, destination)
        mode = "copied"
    valid, reason = _validate_png(destination)
    if not valid:
        raise ValueError(f"imagen compartida inválida: {destination} ({reason})")
    return mode


def _convert_binary(source: Path, destination: Path) -> str:
    """Convierte un binario aprobado con el algoritmo MalPix exacto."""
    if destination.exists():
        valid, reason = _validate_png(destination)
        if not valid:
            raise ValueError(f"imagen aprobada inválida: {destination} ({reason})")
        return "existing"
    data = source.read_bytes()
    if not data:
        raise ValueError(f"binario vacío: {source}")
    height = max(1, (len(data) + WIDTH - 1) // WIDTH)
    padded = data + b"\x00" * (WIDTH * height - len(data))
    image = Image.frombytes("L", (WIDTH, height), padded)
    image = image.resize((HEIGHT, HEIGHT), resample=Image.Resampling.BILINEAR)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_name(f".{destination.name}.tmp")
    try:
        image.save(tmp, format="PNG", optimize=True)
        os.replace(tmp, destination)
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass
    valid, reason = _validate_png(destination)
    if not valid:
        raise ValueError(f"PNG aprobado inválido: {destination} ({reason})")
    return "converted"


def _split_rows(
    rows: list[dict[str, str]], seed: int
) -> tuple[dict[str, str], dict[str, object]]:
    """Divide Dike por (label, format) con orden estable y 70/15/15."""
    seen: set[str] = set()
    for row in rows:
        sha = row["sha256"].strip().lower()
        if not _is_sha256(sha) or sha in seen:
            raise ValueError(f"manifest Dike inválido o duplicado: {sha!r}")
        seen.add(sha)
        row["sha256"] = sha
        if row["label"] not in {"0", "1"}:
            raise ValueError(f"label Dike inválido: {row['label']!r}")
        if not row["format"]:
            raise ValueError(f"format Dike vacío: {sha}")

    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["label"], row["format"])].append(row)

    split_map: dict[str, str] = {}
    for (label, file_format), group in sorted(grouped.items()):
        ordered = sorted(
            group,
            key=lambda row: (
                _stable_digest(seed, f"{label}|{file_format}|{row['sha256']}"),
                row["sha256"],
            ),
        )
        test_count = _round_half_up(len(ordered) * 0.15)
        val_count = _round_half_up(len(ordered) * 0.15)
        if test_count + val_count >= len(ordered):
            raise ValueError(f"estrato demasiado pequeño para dividir: {label}/{file_format}")
        for index, row in enumerate(ordered):
            if index < test_count:
                split_map[row["sha256"]] = "test"
            elif index < test_count + val_count:
                split_map[row["sha256"]] = "val"
            else:
                split_map[row["sha256"]] = "train"

    counts = Counter((split_map[row["sha256"]], row["label"], row["format"]) for row in rows)
    details = {
        "seed": seed,
        "algorithm": "stable sha256 order within (label, format)",
        "val_fraction": 0.15,
        "test_fraction": 0.15,
        "source_manifest_sha256": hashlib.sha256(
            dike_manifest_path().read_bytes()
        ).hexdigest(),
        "counts": {
            split: {
                "total": sum(
                    count
                    for (split_name, _label, _format), count in counts.items()
                    if split_name == split
                ),
                "by_label": {
                    label: sum(
                        count
                        for (split_name, row_label, _format), count in counts.items()
                        if split_name == split and row_label == label
                    )
                    for label in ("0", "1")
                },
                "by_format": {
                    file_format: sum(
                        count
                        for (split_name, _label, row_format), count in counts.items()
                        if split_name == split and row_format == file_format
                    )
                    for file_format in sorted({row["format"] for row in rows})
                },
            }
            for split in ("train", "val", "test")
        },
        "generated_utc": _utc_now(),
    }
    return split_map, details


def ensure_dike_split(
    rows: list[dict[str, str]], seed: int = MALPIX_SEED
) -> tuple[dict[str, str], dict[str, object]]:
    """Crea el split una sola vez; nunca lo regenera silenciosamente."""
    csv_path = splits_dir() / "dike.csv"
    json_path = splits_dir() / "dike_split.json"
    if csv_path.exists():
        existing = _read_csv(csv_path)
        if list(existing[0].keys()) != SPLIT_COLUMNS:
            raise ValueError(f"columnas inesperadas en {csv_path}")
        split_map = {row["sha256"]: row["split"] for row in existing}
        if set(split_map) != {row["sha256"] for row in rows}:
            raise ValueError("dike.csv no cubre exactamente el manifest Dike")
        if any(split not in {"train", "val", "test"} for split in split_map.values()):
            raise ValueError("dike.csv contiene un split inválido")
        if not json_path.exists():
            _, details = _split_rows(rows, seed)
            _write_json_atomic(json_path, details)
        return split_map, json.loads(json_path.read_text(encoding="utf-8"))

    split_map, details = _split_rows(rows, seed)
    split_rows = [
        {
            "sha256": row["sha256"],
            "label": row["label"],
            "format": row["format"],
            "split": split_map[row["sha256"]],
        }
        for row in sorted(rows, key=lambda item: item["sha256"])
    ]
    _write_csv_atomic(csv_path, split_rows, SPLIT_COLUMNS)
    _write_json_atomic(json_path, details)
    return split_map, details


def _ensure_supplement(
    collector_output: Path | None, dike_hashes: set[str]
) -> None:
    """Copia metadatos del colector una sola vez; nunca pisa ediciones manuales."""
    supplemental_dir().mkdir(parents=True, exist_ok=True)
    signed_path = supplemental_dir() / "signed_train.csv"
    external_path = supplemental_dir() / "signed_external.csv"
    unsigned_path = supplemental_dir() / "unsigned_review.csv"
    approval_path = splits_dir() / "approved_unsigned.csv"

    if signed_path.exists() and external_path.exists() and unsigned_path.exists():
        if not approval_path.exists():
            unsigned_rows = _read_csv(unsigned_path)
            _write_csv_atomic(
                approval_path,
                [
                    {
                        "sha256": row["sha256"],
                        "approved": "0",
                        "reason": "",
                        "reviewed_by": "",
                        "reviewed_at": "",
                    }
                    for row in unsigned_rows
                ],
                APPROVAL_COLUMNS,
            )
        return

    if collector_output is None:
        raise ValueError(
            "faltan datasets/supplemental/*.csv; ejecuta prepare-dataset con "
            "--collector-output C:\\Users\\Tron\\benign-collector\\output"
        )
    if not collector_output.is_dir():
        raise FileNotFoundError(collector_output)

    selection_path = collector_output / "selection.csv"
    review_path = collector_output / "review.csv"
    if not selection_path.is_file() or not review_path.is_file():
        raise FileNotFoundError("faltan selection.csv/review.csv en el colector")

    selection = _read_csv(selection_path)
    train_rows = [
        row
        for row in selection
        if row["split"] == "train" and row["signature_status"] == "Valid"
    ]
    external_rows = [
        row for row in selection if row["split"] == "external_test"
    ]
    signed_train = [row for row in train_rows if row["sha256"] not in dike_hashes]
    excluded = len(train_rows) - len(signed_train)
    if excluded != 2:
        raise ValueError(f"se esperaban 2 solapamientos Dike/supplement, hay {excluded}")
    external_hashes = {row["sha256"] for row in external_rows}
    if external_hashes & dike_hashes:
        raise ValueError("el holdout externo se solapa con Dike")
    if external_hashes & {row["sha256"] for row in signed_train}:
        raise ValueError("el holdout externo se solapa con el supplement")

    _write_csv_atomic(
        signed_path,
        [
            {
                "sha256": row["sha256"],
                "label": "0",
                "format": row["format"],
                "size_bytes": row["size_bytes"],
                "family_key": row["family_key"],
                "source_image": str(
                    collector_output / "train" / "images" / f"{row['sha256']}.png"
                ),
            }
            for row in sorted(signed_train, key=lambda item: item["sha256"])
        ],
        SUPPLEMENT_COLUMNS,
    )
    _write_csv_atomic(
        external_path,
        [
            {
                "sha256": row["sha256"],
                "label": "0",
                "format": row["format"],
                "size_bytes": row["size_bytes"],
                "family_key": row["family_key"],
                "source_image": str(
                    collector_output / "external_test" / "images" / f"{row['sha256']}.png"
                ),
            }
            for row in sorted(external_rows, key=lambda item: item["sha256"])
        ],
        SUPPLEMENT_COLUMNS,
    )

    review = _read_csv(review_path)
    unsigned = [row for row in review if row["signature_status"] == "NotSigned"]
    _write_csv_atomic(
        unsigned_path,
        [
            {
                "sha256": row["sha256"],
                "label": "0",
                "format": row["format"],
                "size_bytes": row["size_bytes"],
                "family_key": row["family_key"],
                "original_path": row["original_path"],
                "app_identity": row["app_identity"],
                "vendor": row["vendor"],
                "category": row["source_category"],
            }
            for row in sorted(unsigned, key=lambda item: item["sha256"])
        ],
        UNSIGNED_COLUMNS,
    )
    _write_csv_atomic(
        approval_path,
        [
            {
                "sha256": row["sha256"],
                "approved": "0",
                "reason": "",
                "reviewed_by": "",
                "reviewed_at": "",
            }
            for row in sorted(unsigned, key=lambda item: item["sha256"])
        ],
        APPROVAL_COLUMNS,
    )


def _load_supplement() -> tuple[
    list[dict[str, str]], list[dict[str, str]], list[dict[str, str]], dict[str, str]
]:
    signed = _read_csv(supplemental_dir() / "signed_train.csv")
    external = _read_csv(supplemental_dir() / "signed_external.csv")
    unsigned = _read_csv(supplemental_dir() / "unsigned_review.csv")
    approvals = _read_csv(splits_dir() / "approved_unsigned.csv")
    approved_map = {row["sha256"]: row["approved"] for row in approvals}
    return signed, external, unsigned, approved_map


def _ensure_approved_images(
    unsigned: list[dict[str, str]], approved_map: dict[str, str]
) -> list[dict[str, str]]:
    unsigned_map = {row["sha256"]: row for row in unsigned}
    approved: list[dict[str, str]] = []
    for sha, flag in approved_map.items():
        if flag not in {"0", "1"}:
            raise ValueError(f"approved inválido para {sha}: {flag!r}")
        if flag != "1":
            continue
        if sha not in unsigned_map:
            raise ValueError(f"aprobado ausente de unsigned_review.csv: {sha}")
        row = unsigned_map[sha]
        source = Path(row["original_path"])
        destination = images_dir() / f"{sha}.png"
        if not destination.exists():
            _convert_binary(source, destination)
        else:
            valid, reason = _validate_png(destination)
            if not valid:
                raise ValueError(f"imagen aprobada inválida: {destination} ({reason})")
        approved.append(row)
    return approved


def _manifest_row(
    *,
    sha: str,
    label: str,
    file_format: str,
    size_bytes: str,
    split: str,
    source: str,
    family_key: str = "",
) -> dict[str, str]:
    return {
        "sha256": sha,
        "label": label,
        "format": file_format,
        "size_bytes": size_bytes,
        "image": f"images/{sha}.png",
        "split": split,
        "source": source,
        "family_key": family_key,
    }


def _sort_manifest(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return sorted(rows, key=lambda row: (SPLIT_ORDER[row["split"]], row["sha256"]))


def _class_balance_rows(
    config: str, rows: list[dict[str, str]]
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for split in ("train", "val", "test"):
        subset = [row for row in rows if row["split"] == split]
        total = len(subset)
        for label in ("0", "1"):
            count = sum(1 for row in subset if row["label"] == label)
            weight: object = ""
            if split == "train" and count:
                weight = f"{total / (2 * count):.8f}"
            result.append(
                {
                    "config": config,
                    "split": split,
                    "label": label,
                    "class_name": CLASS_NAMES[int(label)],
                    "count": count,
                    "class_weight": weight,
                }
            )
    return result


def _build_manifests(
    dike_rows: list[dict[str, str]],
    split_map: Mapping[str, str],
    signed: list[dict[str, str]],
    external: list[dict[str, str]],
    approved: list[dict[str, str]],
) -> dict[str, list[dict[str, str]]]:
    dike_hashes = {row["sha256"] for row in dike_rows}
    signed_hashes = {row["sha256"] for row in signed}
    external_hashes = {row["sha256"] for row in external}
    approved_hashes = {row["sha256"] for row in approved}
    if signed_hashes & dike_hashes:
        raise ValueError("el supplement firmado se solapa con Dike")
    if approved_hashes & (dike_hashes | signed_hashes):
        raise ValueError("un unsigned aprobado se solapa con Dike o el supplement")
    if external_hashes & (dike_hashes | signed_hashes | approved_hashes):
        raise ValueError("el holdout externo se solapa con datos de entrenamiento")

    dike_manifest = [
        _manifest_row(
            sha=row["sha256"],
            label=row["label"],
            file_format=row["format"],
            size_bytes=row["size_bytes"],
            split=split_map[row["sha256"]],
            source="dike",
        )
        for row in dike_rows
    ]
    signed_train = [
        _manifest_row(
            sha=row["sha256"],
            label=row["label"],
            file_format=row["format"],
            size_bytes=row["size_bytes"],
            split="train",
            source="signed_supplement",
            family_key=row["family_key"],
        )
        for row in signed
    ]
    approved_rows = [
        _manifest_row(
            sha=row["sha256"],
            label=row["label"],
            file_format=row["format"],
            size_bytes=row["size_bytes"],
            split="train",
            source="approved_unsigned",
            family_key=row["family_key"],
        )
        for row in approved
    ]
    external_manifest = [
        _manifest_row(
            sha=row["sha256"],
            label=row["label"],
            file_format=row["format"],
            size_bytes=row["size_bytes"],
            split="external_test",
            source="signed_supplement",
            family_key=row["family_key"],
        )
        for row in external
    ]

    configs = {
        "original": _sort_manifest(dike_manifest),
        "signed": _sort_manifest(dike_manifest + signed_train),
        "mixed": _sort_manifest(dike_manifest + signed_train + approved_rows),
    }
    for name, rows in configs.items():
        hashes = [row["sha256"] for row in rows]
        if len(hashes) != len(set(hashes)):
            raise ValueError(f"hash duplicado en manifiesto {name}")
    external_manifest = _sort_manifest(external_manifest)
    if len(external_manifest) != len({row["sha256"] for row in external_manifest}):
        raise ValueError("hash duplicado en external_benign")

    for name, rows in configs.items():
        _write_csv_atomic(manifests_dir() / f"{name}.csv", rows, MANIFEST_COLUMNS)
    _write_csv_atomic(
        manifests_dir() / "external_benign.csv", external_manifest, MANIFEST_COLUMNS
    )

    balance_rows: list[dict[str, object]] = []
    for name, rows in configs.items():
        balance_rows.extend(_class_balance_rows(name, rows))
    _write_csv_atomic(
        splits_dir() / "class_balance.csv", balance_rows, CLASS_BALANCE_COLUMNS
    )
    return {**configs, "external_benign": external_manifest}


def prepare_dataset(
    collector_output: Path | None = None, seed: int = MALPIX_SEED
) -> dict[str, object]:
    """Migra Dike + colector a un store compartido y manifiestos por configuración."""
    if not dike_manifest_path().is_file():
        raise FileNotFoundError(dike_manifest_path())
    dike_rows = _read_csv(dike_manifest_path())
    for row in dike_rows:
        source = dataset_root() / "dike" / row["image"]
        if not source.is_file():
            raise FileNotFoundError(source)
    dike_hashes = {row["sha256"] for row in dike_rows}
    split_map, split_details = ensure_dike_split(dike_rows, seed)
    _ensure_supplement(collector_output, dike_hashes)
    signed, external, unsigned, approved_map = _load_supplement()
    if len(signed) != 808:
        raise ValueError(f"se esperaban 808 filas signed_train, hay {len(signed)}")
    if len(external) != 221:
        raise ValueError(f"se esperaban 221 filas signed_external, hay {len(external)}")

    images_dir().mkdir(parents=True, exist_ok=True)
    modes = Counter()
    for row in dike_rows:
        source = dataset_root() / "dike" / row["image"]
        modes[_link_or_copy(source, images_dir() / f"{row['sha256']}.png")] += 1
    for row in signed + external:
        source = Path(row["source_image"])
        modes[_link_or_copy(source, images_dir() / f"{row['sha256']}.png")] += 1
    approved = _ensure_approved_images(unsigned, approved_map)
    manifests = _build_manifests(dike_rows, split_map, signed, external, approved)

    summary = {
        "generated_utc": _utc_now(),
        "seed": seed,
        "split": split_details,
        "images": {
            "total": len(list(images_dir().glob("*.png"))),
            "hardlinked": sum(
                1 for path in images_dir().glob("*.png") if path.stat().st_nlink > 1
            ),
            "link_modes": dict(modes),
        },
        "configs": {
            name: {
                "rows": len(rows),
                "counts": {
                    f"{split}/{label}": count
                    for (split, label), count in Counter(
                        (row["split"], row["label"]) for row in rows
                    ).items()
                },
            }
            for name, rows in manifests.items()
        },
        "supplement": {
            "signed_train": len(signed),
            "external_benign": len(external),
            "unsigned_review": len(unsigned),
            "approved_unsigned": len(approved),
            "excluded_dike_overlap": 2,
        },
    }
    _write_json_atomic(splits_dir() / "dataset_summary.json", summary)
    return summary


def _validate_manifest_rows(
    rows: list[dict[str, str]], name: str, errors: list[str]
) -> None:
    if not rows:
        errors.append(f"{name}: manifiesto vacío")
        return
    columns = list(rows[0].keys())
    if columns != MANIFEST_COLUMNS:
        errors.append(f"{name}: columnas {columns!r}, esperadas {MANIFEST_COLUMNS!r}")
        return
    seen: set[str] = set()
    for index, row in enumerate(rows, start=2):
        sha = (row.get("sha256") or "").strip().lower()
        if not _is_sha256(sha):
            errors.append(f"{name}:{index}: sha256 inválido")
        elif sha in seen:
            errors.append(f"{name}:{index}: sha256 duplicado")
        seen.add(sha)
        if row.get("label") not in {"0", "1"}:
            errors.append(f"{name}:{index}: label inválido")
        if not row.get("format"):
            errors.append(f"{name}:{index}: format vacío")
        if row.get("split") not in SPLIT_ORDER:
            errors.append(f"{name}:{index}: split inválido")
        if row.get("source") not in {
            "dike",
            "signed_supplement",
            "approved_unsigned",
        }:
            errors.append(f"{name}:{index}: source inválido")
        image = row.get("image") or ""
        if not image.startswith("images/") or ".." in Path(image).parts:
            errors.append(f"{name}:{index}: ruta de imagen insegura")
        elif Path(image).stem != sha:
            errors.append(f"{name}:{index}: imagen no coincide con sha")


def _manifest_config_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    return {
        f"{split}/{label}": sum(
            1 for row in rows if row["split"] == split and row["label"] == label
        )
        for split in ("train", "val", "test")
        for label in ("0", "1")
    }


def validate_dataset() -> dict[str, object]:
    """Valida paths, imágenes, duplicados, leakage y conteos."""
    errors: list[str] = []
    warnings: list[str] = []
    dike_rows = _read_csv(dike_manifest_path())
    dike_hashes = {row["sha256"] for row in dike_rows}
    split_path = splits_dir() / "dike.csv"
    if not split_path.exists():
        errors.append("falta datasets/splits/dike.csv")
        split_map: dict[str, str] = {}
    else:
        split_rows = _read_csv(split_path)
        if list(split_rows[0].keys()) != SPLIT_COLUMNS:
            errors.append("dike.csv tiene columnas incorrectas")
        split_map = {row["sha256"]: row["split"] for row in split_rows}
        if set(split_map) != dike_hashes:
            errors.append("dike.csv no cubre exactamente el manifest Dike")
        for row in split_rows:
            if row["split"] not in {"train", "val", "test"}:
                errors.append(f"dike.csv: split inválido {row['split']!r}")
    if {"0", "1"} - {row["label"] for row in dike_rows}:
        warnings.append("Dike no contiene ambas clases")

    external_path = manifests_dir() / "external_benign.csv"
    if not external_path.exists():
        errors.append("falta datasets/manifests/external_benign.csv")
        external_rows: list[dict[str, str]] = []
    else:
        external_rows = _read_csv(external_path)
        _validate_manifest_rows(external_rows, "external_benign", errors)
        if any(row["split"] != "external_test" for row in external_rows):
            errors.append("external_benign contiene un split distinto de external_test")
        if any(row["label"] != "0" for row in external_rows):
            errors.append("external_benign contiene una etiqueta distinta de benign")
        if len(external_rows) != 221:
            errors.append(f"external_benign debe tener 221 filas, tiene {len(external_rows)}")
    external_hashes = {row["sha256"] for row in external_rows}

    unsigned_dir = supplemental_dir()
    unsigned_rows = _read_csv(unsigned_dir / "unsigned_review.csv")
    unsigned_hashes = {row["sha256"] for row in unsigned_rows}
    if len(unsigned_hashes) != 694:
        errors.append(f"unsigned_review debe tener 694 hashes, tiene {len(unsigned_hashes)}")
    approvals = _read_csv(splits_dir() / "approved_unsigned.csv")
    approval_map = {row["sha256"]: row["approved"] for row in approvals}
    if set(approval_map) - unsigned_hashes:
        errors.append("approved_unsigned contiene hashes fuera de unsigned_review")
    for sha, flag in approval_map.items():
        if flag not in {"0", "1"}:
            errors.append(f"approved_unsigned: flag inválido para {sha}")
    approved_hashes = {sha for sha, flag in approval_map.items() if flag == "1"}

    config_rows: dict[str, list[dict[str, str]]] = {}
    dike_val: dict[str, set[str]] = {}
    dike_test: dict[str, set[str]] = {}
    for config in MALPIX_CONFIGS:
        path = manifests_dir() / f"{config}.csv"
        if not path.exists():
            errors.append(f"falta {path}")
            continue
        rows = _read_csv(path)
        config_rows[config] = rows
        _validate_manifest_rows(rows, config, errors)
        by_hash = {row["sha256"]: row for row in rows}
        if len(by_hash) != len(rows):
            errors.append(f"{config}: hashes duplicados")
        for split in ("train", "val", "test"):
            subset = [row for row in rows if row["split"] == split]
            if not subset:
                errors.append(f"{config}: split {split} vacío")
        actual_counts = Counter(row["split"] for row in rows)
        expected_train = 8345
        if config in {"signed", "mixed"}:
            expected_train += 808
        if config == "mixed":
            expected_train += len(approved_hashes)
        if actual_counts.get("train") != expected_train:
            errors.append(
                f"{config}: train esperado {expected_train}, tiene {actual_counts.get('train')}"
            )
        if actual_counts.get("val") != 1789 or actual_counts.get("test") != 1789:
            errors.append(f"{config}: validación/test Dike alterados")
        dike_val[config] = {
            row["sha256"] for row in rows if row["split"] == "val" and row["source"] == "dike"
        }
        dike_test[config] = {
            row["sha256"] for row in rows if row["split"] == "test" and row["source"] == "dike"
        }
        for row in rows:
            if row["source"] != "dike" and row["split"] != "train":
                errors.append(f"{config}: {row['source']} fuera de train")
            if row["source"] != "dike" and row["sha256"] in dike_hashes:
                errors.append(f"{config}: supplement solapa Dike: {row['sha256']}")
            if row["sha256"] in external_hashes:
                errors.append(f"{config}: holdout externo dentro de {row['split']}")
            if row["source"] == "approved_unsigned" and row["sha256"] not in approved_hashes:
                errors.append(f"{config}: unsigned no aprobado en mixed")
        known = {row["family_key"] for row in rows if row["family_key"]}
        train_families = {
            row["family_key"]
            for row in rows
            if row["split"] == "train" and row["family_key"]
        }
        eval_families = {
            row["family_key"]
            for row in rows
            if row["split"] in {"val", "test"} and row["family_key"]
        }
        leak = train_families & eval_families
        if leak:
            errors.append(f"{config}: fuga de familia train/eval: {sorted(leak)[:5]}")
        external_families = {row["family_key"] for row in external_rows if row["family_key"]}
        ext_leak = known & external_families
        if ext_leak:
            errors.append(f"{config}: fuga de familia con holdout: {sorted(ext_leak)[:5]}")
        missing_family = sum(1 for row in rows if row["source"] == "dike" and not row["family_key"])
        if missing_family:
            warnings.append(
                f"{config}: {missing_family} filas Dike sin family_key (metadata no disponible)"
            )

    split_val = {sha for sha, split in split_map.items() if split == "val"}
    split_test = {sha for sha, split in split_map.items() if split == "test"}
    split_counts = Counter(split_map.values())
    if split_counts.get("train") != 8345:
        errors.append(f"dike.csv: train esperado 8345, tiene {split_counts.get('train')}")
    if split_counts.get("val") != 1789:
        errors.append(f"dike.csv: val esperado 1789, tiene {split_counts.get('val')}")
    if split_counts.get("test") != 1789:
        errors.append(f"dike.csv: test esperado 1789, tiene {split_counts.get('test')}")
    if config_rows:
        baseline_test = dike_test.get("original", set())
        baseline_val = dike_val.get("original", set())
        for config in MALPIX_CONFIGS:
            if config not in config_rows:
                continue
            if dike_test.get(config) != split_test or dike_test.get(config) != baseline_test:
                errors.append(f"{config}: el test Dike no coincide con dike.csv")
            if dike_val.get(config) != split_val or dike_val.get(config) != baseline_val:
                errors.append(f"{config}: la validación Dike no coincide con dike.csv")
        if len(baseline_test) == 0:
            errors.append("el test Dike está vacío")
        if len(baseline_val) == 0:
            errors.append("la validación Dike está vacía")

    for config in MALPIX_CONFIGS:
        rows = config_rows.get(config, [])
        mixed_approved = {
            row["sha256"]
            for row in rows
            if row["source"] == "approved_unsigned"
        }
        if not mixed_approved <= approved_hashes:
            errors.append(f"{config}: approved_unsigned no aprobado")
        if config == "mixed" and mixed_approved != approved_hashes:
            errors.append("mixed no incluye exactamente todos los unsigned aprobados")
        for row in rows:
            image_path = dataset_root() / row["image"]
            if not image_path.is_file():
                errors.append(f"{config}: imagen faltante {row['image']}")
                continue
            valid, reason = _validate_png(image_path)
            if not valid:
                errors.append(f"{config}: PNG inválido {row['image']} ({reason})")
        balance = splits_dir() / "class_balance.csv"
        if balance.exists():
            balance_rows = [
                row for row in _read_csv(balance) if row["config"] == config
            ]
            train_rows = [row for row in rows if row["split"] == "train"]
            for label in ("0", "1"):
                expected = sum(1 for row in train_rows if row["label"] == label)
                actual_rows = [
                    row
                    for row in balance_rows
                    if row["split"] == "train" and row["label"] == label
                ]
                actual = int(actual_rows[0]["count"]) if actual_rows else -1
                if actual != expected:
                    errors.append(f"{config}: class_balance train/{label} incorrecto")
                if expected:
                    weight = len(train_rows) / (2 * expected)
                    stored = float(actual_rows[0]["class_weight"])
                    if abs(stored - weight) > 1e-6:
                        errors.append(f"{config}: class_weight train/{label} incorrecto")
        else:
            errors.append("falta datasets/splits/class_balance.csv")

    referenced: set[str] = set()
    for config in MALPIX_CONFIGS:
        for row in config_rows.get(config, []):
            referenced.add(row["image"])
    for row in external_rows:
        referenced.add(row["image"])
    store_files = {str(path.relative_to(dataset_root())).replace("\\", "/") for path in images_dir().glob("*.png")}
    if store_files - referenced:
        warnings.append(
            f"imágenes no referenciadas en el store: {len(store_files - referenced)}"
        )
    if not store_files:
        errors.append("datasets/images está vacío")
    for image in sorted(referenced):
        path = dataset_root() / image
        if not path.is_file():
            errors.append(f"imagen referenciada faltante: {image}")
        elif not _validate_png(path)[0]:
            errors.append(f"PNG inválido: {image}")

    if not approved_hashes:
        warnings.append("mixed coincide con signed: no hay unsigned aprobados")

    counts = {
        config: {
            "rows": len(rows),
            "splits": dict(Counter(row["split"] for row in rows)),
            "sources": dict(Counter(row["source"] for row in rows)),
            "class_counts": _manifest_config_counts(rows),
        }
        for config, rows in config_rows.items()
    }
    result = {
        "ok": not errors,
        "generated_utc": _utc_now(),
        "errors": errors,
        "warnings": sorted(set(warnings)),
        "dike_test_hash_count": len(dike_test.get("original", set())),
        "dike_val_hash_count": len(dike_val.get("original", set())),
        "external_benign_hash_count": len(external_hashes),
        "approved_unsigned_count": len(approved_hashes),
        "image_store_count": len(store_files),
        "configs": counts,
    }
    out_path = ROOT / "out" / "malpix" / "validation.json"
    _write_json_atomic(out_path, result)
    return result


def read_manifest_rows(config: str) -> list[dict[str, str]]:
    path = MalPixData.load(config).manifest_path
    rows = _read_csv(path)
    if not rows:
        raise ValueError(f"manifiesto vacío: {path}")
    if list(rows[0].keys()) != MANIFEST_COLUMNS:
        raise ValueError(f"columnas incorrectas en {path}")
    return rows


def read_external_rows() -> list[dict[str, str]]:
    path = MalPixData.load("original").external_manifest_path
    rows = _read_csv(path)
    if list(rows[0].keys()) != MANIFEST_COLUMNS:
        raise ValueError(f"columnas incorrectas en {path}")
    return rows
