"""Boundary tests use only temporary synthetic images, never the project dataset."""

import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import pytest
import torch
from PIL import Image


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/inspect_hard_train.py"
spec = importlib.util.spec_from_file_location("inspect_hard_train", SCRIPT)
diagnostics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostics)


class ColorModel(torch.nn.Module):
    def forward(self, images):
        assert not self.training
        assert torch.is_inference_mode_enabled()
        assert not torch.is_grad_enabled()
        return torch.stack((images[:, 0].mean((1, 2)), images[:, 2].mean((1, 2))), dim=1)


@pytest.fixture
def synthetic_case(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostics, "build_model", lambda **kwargs: ColorModel())
    data = tmp_path / "data"
    data.mkdir()
    rows = []
    samples = [(0, (255, 0, 0), "g0"), (1, (0, 0, 255), "g1"),
               (0, (128, 128, 128), "g2"), (0, (0, 0, 255), "g2")]
    for index, (label, color, group) in enumerate(samples):
        name = f"synthetic-{index}.png"
        Image.new("RGB", (64, 64), color).save(data / name)
        rows.append({"relative_path": f"data/{name}", "label": label, "class_name": ["cat", "dog"][label],
                     "split": "train", "group_id": group, "width": 64, "height": 64})
    # These files must never be opened, not even for a thumbnail/header.
    for split, label in [("val", 0), ("test", 1)]:
        rows.append({"relative_path": f"data/MUST_NOT_OPEN_{split}.png", "label": label,
                     "class_name": ["cat", "dog"][label], "split": split,
                     "group_id": f"reserved_{split}", "width": 64, "height": 64})
    manifest = data / "splits.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    checkpoint = tmp_path / "own.pt"
    payload = {"model_config": {"name": "robust_resnet18", "width": 8}, "state_dict": {},
               "pretrained": False, "initialization": "random", "epoch": 20,
               "image_size": 64, "class_names": ["cat", "dog"], "resize_mode": "letterbox",
               "normalization": {"mean": [0.5] * 3, "std": [0.5] * 3},
               # A serving calibration must not influence train difficulty.
               "temperature": 0.25, "tta_horizontal_flip": True,
               "train_config": {"pretrained": False, "initialization": "random",
                                "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()}}
    torch.save(payload, checkpoint)
    return manifest, checkpoint, payload, tmp_path / "fresh_diagnostics"


def test_only_train_images_are_scored_and_outputs_are_finite(synthetic_case):
    manifest, checkpoint, _, output = synthetic_case
    summary = diagnostics.inspect_train(checkpoint, manifest, output, batch_size=3)
    assert summary["images"] == 4 and summary["groups"] == 3
    assert summary["correct"] == 3 and summary["errors"] == 1
    assert summary["scope"]["heldout_images_read"] is False
    assert summary["scope"]["temperature"] == 1.0 and summary["scope"]["tta"] is False
    assert summary["manifest"]["split_counts_metadata_only"] == {"train": 4, "val": 1, "test": 1}
    assert summary["checkpoint"]["sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    saved = json.loads((output / "summary.json").read_text())
    assert saved["checkpoint"]["epoch"] == 20
    with (output / "train_difficulty.csv").open(newline="", encoding="utf-8") as handle:
        scores = list(csv.DictReader(handle))
    assert len(scores) == 4
    assert all("MUST_NOT_OPEN" not in row["relative_path"] for row in scores)
    for row in scores:
        assert math.isfinite(float(row["loss"]))
        assert float(row["score"]) == float(row["loss"])
        assert float(row["cat_probability"]) + float(row["dog_probability"]) == pytest.approx(1.0)
    assert float(scores[0]["loss"]) == pytest.approx(math.log1p(math.exp(-2)), abs=1e-6)
    assert float(scores[3]["loss"]) == pytest.approx(math.log1p(math.exp(2)), abs=1e-6)
    with (output / "train_groups.csv").open(newline="", encoding="utf-8") as handle:
        groups = list(csv.DictReader(handle))
    assert len(groups) == 3
    assert next(row for row in groups if row["group_id"] == "g2")["images"] == "2"
    with Image.open(output / "hard_train_gallery.png") as gallery:
        assert gallery.width > 100 and gallery.height > 100


def test_manifest_mismatch_rejects_before_any_image_read(synthetic_case, monkeypatch):
    manifest, checkpoint, payload, output = synthetic_case
    payload["train_config"]["manifest_sha256"] = "0" * 64
    torch.save(payload, checkpoint)
    monkeypatch.setattr(diagnostics.Image, "open", lambda *args, **kwargs: pytest.fail("Image opened before hash check"))
    with pytest.raises(ValueError, match="Manifest hash mismatch"):
        diagnostics.inspect_train(checkpoint, manifest, output)
    assert not output.exists()


@pytest.mark.parametrize("flag", [True, None, 0])
def test_external_or_unconfirmed_pretraining_is_rejected(synthetic_case, monkeypatch, flag):
    manifest, checkpoint, payload, output = synthetic_case
    payload["pretrained"] = flag
    torch.save(payload, checkpoint)
    monkeypatch.setattr(diagnostics.Image, "open", lambda *args, **kwargs: pytest.fail("Unexpected image read"))
    with pytest.raises(ValueError, match="scratch-checkpoint"):
        diagnostics.inspect_train(checkpoint, manifest, output)


def test_existing_output_is_never_overwritten(synthetic_case, monkeypatch):
    manifest, checkpoint, _, output = synthetic_case
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    monkeypatch.setattr(diagnostics.Image, "open", lambda *args, **kwargs: pytest.fail("Unexpected image read"))
    with pytest.raises(FileExistsError, match="новый output"):
        diagnostics.inspect_train(checkpoint, manifest, output)
    assert marker.read_text() == "keep"


def test_checkpoint_is_read_once_and_hashes_the_loaded_snapshot(synthetic_case, monkeypatch):
    manifest, checkpoint, _, output = synthetic_case
    original_read = Path.read_bytes
    calls = 0
    def single_checkpoint_read(path):
        nonlocal calls
        if path == checkpoint:
            calls += 1
            assert calls == 1
        return original_read(path)
    monkeypatch.setattr(Path, "read_bytes", single_checkpoint_read)
    diagnostics.inspect_train(checkpoint, manifest, output)
    assert calls == 1
