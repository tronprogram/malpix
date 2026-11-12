"""Pruebas del split congelado, umbral de validación y loaders MalPix."""

from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch
from PIL import Image

from app.config import MalPixData, MALPIX_CONFIGS
from app.data import (
    MALPIX_MEAN,
    MALPIX_STD,
    MalPixDataset,
    load_external_loader,
    load_malpix_loaders,
    malpix_transform,
)
from app.dataset import (
    _read_csv,
    _split_rows,
    dataset_root,
    manifests_dir,
    splits_dir,
    validate_dataset,
)
from app.evaluate import select_threshold
from app.inference import run as infer_image
from app.model import build_model


class MalPixLayoutTests(unittest.TestCase):
    """Comprueba que el layout sea reproducible y no tenga fugas evidentes."""

    def test_dike_split_is_deterministic_and_stratified(self):
        rows = _read_csv(dataset_root() / "dike" / "manifest.csv")
        first_map, first_details = _split_rows(rows, 20261110)
        second_map, second_details = _split_rows(rows, 20261110)
        self.assertEqual(first_map, second_map)
        self.assertEqual(first_details["counts"], second_details["counts"])
        self.assertEqual(first_details["counts"]["train"]["total"], 8345)
        self.assertEqual(first_details["counts"]["val"]["total"], 1789)
        self.assertEqual(first_details["counts"]["test"]["total"], 1789)
        for split in ("train", "val", "test"):
            self.assertIn("0", first_details["counts"][split]["by_label"])
            self.assertIn("1", first_details["counts"][split]["by_label"])

    def test_all_configs_preserve_dike_val_and_test(self):
        by_config = {}
        for config in MALPIX_CONFIGS:
            rows = _read_csv(manifests_dir() / f"{config}.csv")
            by_config[config] = {
                split: {
                    row["sha256"]
                    for row in rows
                    if row["source"] == "dike" and row["split"] == split
                }
                for split in ("val", "test")
            }
        for config in MALPIX_CONFIGS:
            self.assertEqual(by_config[config], by_config["original"])

    def test_supplement_does_not_overlap_dike_or_external(self):
        dike = {
            row["sha256"]
            for row in _read_csv(dataset_root() / "dike" / "manifest.csv")
        }
        signed = {
            row["sha256"]
            for row in _read_csv(dataset_root() / "supplemental" / "signed_train.csv")
        }
        external = {
            row["sha256"]
            for row in _read_csv(dataset_root() / "supplemental" / "signed_external.csv")
        }
        self.assertEqual(len(signed), 808)
        self.assertEqual(len(external), 221)
        self.assertFalse(dike & signed)
        self.assertFalse(dike & external)
        self.assertFalse(signed & external)

    def test_class_balance_weights_match_train_counts(self):
        rows = _read_csv(splits_dir() / "class_balance.csv")
        for config in MALPIX_CONFIGS:
            manifest = _read_csv(manifests_dir() / f"{config}.csv")
            total = sum(1 for row in manifest if row["split"] == "train")
            for label in ("0", "1"):
                count = sum(
                    1
                    for row in manifest
                    if row["split"] == "train" and row["label"] == label
                )
                stored = next(
                    row
                    for row in rows
                    if row["config"] == config
                    and row["split"] == "train"
                    and row["label"] == label
                )
                self.assertEqual(int(stored["count"]), count)
                self.assertAlmostEqual(
                    float(stored["class_weight"]), total / (2 * count), places=7
                )

    def test_threshold_is_selected_on_validation_only(self):
        y_true = [0, 0, 0, 1, 1, 1]
        probabilities = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]
        threshold, metrics = select_threshold(y_true, probabilities)
        self.assertEqual(threshold, 0.5)
        self.assertEqual(metrics["balanced_accuracy"], 1.0)
        self.assertEqual(metrics["macro_f1"], 1.0)

    def test_threshold_requires_both_classes(self):
        with self.assertRaises(ValueError):
            select_threshold([0, 0, 0], [0.1, 0.2, 0.3])

    def test_loader_smoke_for_all_configs(self):
        for config in MALPIX_CONFIGS:
            data = MalPixData.load(config)
            classes, train_loader, val_loader, test_loader = load_malpix_loaders(data)
            self.assertEqual(classes, ["benign", "malware"])
            images, labels = next(iter(train_loader))
            self.assertEqual(tuple(images.shape), (32, 1, 224, 224))
            self.assertEqual(images.shape[0], labels.shape[0])
            self.assertTrue({int(value) for value in labels.tolist()} <= {0, 1})
            self.assertGreaterEqual(float(images.min()), -1.0)
            self.assertLessEqual(float(images.max()), 1.0)
            self.assertGreater(len(val_loader.dataset), 0)
            self.assertGreater(len(test_loader.dataset), 0)

    def test_malpix_model_is_single_channel_and_two_logits(self):
        data = MalPixData.load("original")
        _classes, train_loader, _val_loader, _test_loader = load_malpix_loaders(data)
        images, _labels = next(iter(train_loader))
        model = build_model(pretrained=False, seed=data.seed)
        model.eval()
        with torch.no_grad():
            logits = model(images)
        self.assertEqual(images.shape[1], 1)
        self.assertEqual(model.conv1.in_channels, 1)
        self.assertEqual(model.fc.out_features, 2)
        self.assertEqual(tuple(logits.shape), (images.shape[0], 2))

    def test_malpix_initialization_is_deterministic(self):
        first = build_model(pretrained=False, seed=20261110)
        second = build_model(pretrained=False, seed=20261110)
        self.assertTrue(torch.equal(first.conv1.weight, second.conv1.weight))
        self.assertTrue(torch.equal(first.fc.weight, second.fc.weight))
        self.assertTrue(torch.equal(first.fc.bias, second.fc.bias))

    def test_original_and_signed_share_training_policy(self):
        original = MalPixData.load("original")
        signed = MalPixData.load("signed")
        for field in ("seed", "batch_size", "epochs", "freeze_epochs", "lr"):
            self.assertEqual(getattr(original, field), getattr(signed, field))

    def test_preprocessing_is_single_channel_and_inference_compatible(self):
        data = MalPixData.load("original")
        row = next(row for row in _read_csv(manifests_dir() / "original.csv"))
        dataset = MalPixDataset([row], data.data_root)
        tensor, _label = dataset[0]
        self.assertEqual(tuple(tensor.shape), (1, 224, 224))
        self.assertEqual(MALPIX_MEAN, (0.5,))
        self.assertEqual(MALPIX_STD, (0.5,))
        with Image.open(data.data_root / row["image"]) as image:
            expected = malpix_transform()(image.convert("L"))
        self.assertTrue(torch.equal(tensor, expected))

    def test_inference_uses_the_same_single_channel_contract(self):
        data = MalPixData.load("original")
        row = next(row for row in _read_csv(manifests_dir() / "original.csv"))
        image_path = data.data_root / row["image"]
        with tempfile.TemporaryDirectory() as tmp:
            model = build_model(pretrained=False, seed=data.seed)
            inference_data = replace(
                data,
                models_dir=Path(tmp) / "models",
                figures_dir=Path(tmp) / "figures",
            )
            inference_data.models_dir.mkdir(parents=True)
            torch.save(
                {
                    "score": 0.0,
                    "state": model.state_dict(),
                    "classes": ["benign", "malware"],
                    "benign": "benign",
                },
                inference_data.models_dir / "best.pt",
            )
            with contextlib.redirect_stdout(io.StringIO()):
                label = infer_image(inference_data, image_path)
        self.assertIn(label, {"benign", "malware"})

    def test_external_loader_is_separate(self):
        data = MalPixData.load("original")
        loader = load_external_loader(data)
        self.assertEqual(len(loader.dataset), 221)
        images, labels = next(iter(loader))
        self.assertEqual(tuple(images.shape[1:]), (1, 224, 224))
        self.assertTrue(all(int(value) == 0 for value in labels.tolist()))

    def test_validation_command_passes(self):
        result = validate_dataset()
        self.assertTrue(result["ok"], result["errors"])
        self.assertEqual(result["dike_test_hash_count"], 1789)
        self.assertEqual(result["dike_val_hash_count"], 1789)
        self.assertEqual(result["external_benign_hash_count"], 221)
        self.assertEqual(result["approved_unsigned_count"], 0)
        self.assertEqual(result["image_store_count"], 12952)


if __name__ == "__main__":
    unittest.main()
