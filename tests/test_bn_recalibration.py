"""Refresh tiny synthetic train images; real train/val/test data is never used."""

import csv
import hashlib
from pathlib import Path

import pytest
import torch
from PIL import Image
from torch import nn

import catsdogs.recalibrate_bn as recalibration
from catsdogs.model import build_model
from catsdogs.train import NORMALIZATION


class ProbeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor([1.25, 0.75, 2.0]))
        self.bn = nn.BatchNorm2d(3)
        self.dropout = nn.Dropout(0.9)
        self.classifier = nn.Linear(3, 2)
        self.register_buffer("unchanged_marker", torch.tensor(41))
        self.calls = []

    def forward(self, images):
        self.calls.append({"model_train": self.training, "bn_train": self.bn.training,
                           "dropout_train": self.dropout.training,
                           "gradient_mode": torch.is_grad_enabled(),
                           "inference_mode": torch.is_inference_mode_enabled()})
        features = self.bn(images * self.scale[None, :, None, None])
        features = self.dropout(features.mean(dim=(2, 3)))
        return self.classifier(features)


@pytest.fixture
def synthetic(tmp_path, monkeypatch):
    root = tmp_path / "project"
    directory = root / "data/train"
    directory.mkdir(parents=True)
    rows = []
    for number, (class_name, color) in enumerate([
        ("cat", (210, 25, 80)), ("cat", (80, 220, 30)), ("dog", (30, 45, 190)),
    ]):
        relative = f"data/train/{class_name}.{number}.png"
        Image.new("RGB", (16, 12), color).save(root / relative)
        rows.append({"relative_path": relative, "label": int(class_name == "dog"),
                     "class_name": class_name, "split": "train", "group_id": f"train-{number}",
                     "width": 16, "height": 12, "sha256": hashlib.sha256(relative.encode()).hexdigest()})
    for split in ("val", "test"):
        for class_name, label in (("cat", 0), ("dog", 1)):
            relative = f"data/{split}/{class_name}.png"
            rows.append({"relative_path": relative, "label": label, "class_name": class_name,
                         "split": split, "group_id": f"{split}-{class_name}",
                         "width": 16, "height": 12, "sha256": hashlib.sha256(relative.encode()).hexdigest()})
    manifest = root / "data/splits_v2.csv"
    write_manifest(manifest, rows)
    model = ProbeModel()
    with torch.no_grad():
        model.bn.running_mean.fill_(7)
        model.bn.running_var.fill_(5)
        model.bn.num_batches_tracked.fill_(123)
    source = root / "source.pt"
    checkpoint = {
        "model_config": {"name": "robust_resnet18", "width": 8, "num_classes": 2, "dropout": 0.15},
        "state_dict": model.state_dict(), "image_size": 64, "epoch": 17,
        "class_names": ["cat", "dog"], "normalization": NORMALIZATION, "resize_mode": "letterbox",
        "pretrained": False, "initialization": "random", "val_source": "ema",
        "train_config": {"pretrained": False, "initialization": "random",
                         "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()},
        "temperature": 0.37, "threshold": 0.71, "tta_horizontal_flip": True,
        "calibration": {"temperature": 0.37}, "validation": {"accuracy": 0.99},
        "selection_frozen_before_test": True, "selection_frozen": True,
    }
    torch.save(checkpoint, source)
    instances = []

    def fresh_model(**config):
        instance = ProbeModel()
        instances.append(instance)
        return instance

    monkeypatch.setattr(recalibration, "build_model", fresh_model)
    return {"root": root, "rows": rows, "manifest": manifest, "source": source,
            "output": root / "candidate.pt", "instances": instances}


def write_manifest(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(synthetic, **kwargs):
    return recalibration.recalibrate_checkpoint(synthetic["source"], synthetic["output"],
                                                synthetic["manifest"], batch_size=2, **kwargs)


def raw_bytes(tensor):
    return tensor.detach().cpu().numpy().tobytes()


def test_refresh_changes_only_bn_buffers_and_reads_source_once(synthetic, monkeypatch):
    original_read = Path.read_bytes
    source_bytes = original_read(synthetic["source"])
    reads, opened = [], []

    def count_read(path):
        if path.resolve() == synthetic["source"]:
            reads.append(path)
        return original_read(path)

    original_open = Image.open

    def train_only_open(path, *args, **kwargs):
        resolved = Path(path).resolve()
        assert resolved.parent == synthetic["root"] / "data/train"
        opened.append(resolved)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_bytes", count_read)
    monkeypatch.setattr(Image, "open", train_only_open)
    details = run(synthetic)
    assert len(reads) == 1
    assert original_read(synthetic["source"]) == source_bytes
    before = torch.load(synthetic["source"], map_location="cpu", weights_only=True)
    after = torch.load(synthetic["output"], map_location="cpu", weights_only=True)
    changed = {key for key in before["state_dict"]
               if raw_bytes(before["state_dict"][key]) != raw_bytes(after["state_dict"][key])}
    assert changed == {"bn.running_mean", "bn.running_var", "bn.num_batches_tracked"}
    assert set(details["changed_bn_buffers"]) == changed
    assert after["state_dict"]["bn.num_batches_tracked"].item() == 2
    assert len(opened) == len(set(opened)) == 3
    model = synthetic["instances"][0]
    assert all(not call["model_train"] and call["bn_train"] and not call["dropout_train"]
               and not call["gradient_mode"] and call["inference_mode"] for call in model.calls)
    assert not model.training and not model.bn.training
    assert model.bn.momentum == 0.1
    assert all(parameter.grad is None for parameter in model.parameters())
    assert details["source_checkpoint_sha256"] == hashlib.sha256(source_bytes).hexdigest()
    assert details["test_images_read"] is False and details["validation_images_read"] is False
    assert after["candidate_validation_pending"] is True
    assert after["selection_frozen_before_test"] is False and after["selection_frozen"] is False
    assert after["temperature"] == 1.0 and after["threshold"] == 0.5
    assert after["tta_horizontal_flip"] is False
    assert "calibration" not in after and "validation" not in after
    assert after["provenance"]["method"] == "train_only_batchnorm_recalibration"


def test_optional_val_is_eval_only_and_does_not_contribute_bn_statistics(synthetic, monkeypatch):
    for row in synthetic["rows"]:
        if row["split"] == "val":
            path = synthetic["root"] / row["relative_path"]
            path.parent.mkdir(exist_ok=True)
            Image.new("RGB", (16, 12), (250, 250, 250)).save(path)
    original_open = Image.open

    def no_test_open(path, *args, **kwargs):
        assert Path(path).parent.name != "test"
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Image, "open", no_test_open)
    details = run(synthetic, validate=True)
    candidate = torch.load(synthetic["output"], map_location="cpu", weights_only=True)
    assert details["before_validation"] is not None and details["after_validation"] is not None
    assert candidate["state_dict"]["bn.num_batches_tracked"].item() == 2
    calls = synthetic["instances"][0].calls
    assert [call["bn_train"] for call in calls] == [False, True, True, False]
    assert all(not call["dropout_train"] and not call["gradient_mode"] for call in calls)
    assert candidate["candidate_validation_pending"] is True


def test_confirmed_own_continuation_keeps_its_initialization_provenance(synthetic):
    checkpoint = torch.load(synthetic["source"], map_location="cpu", weights_only=True)
    checkpoint["initialization"] = "continuation_of_own_scratch_training"
    checkpoint["train_config"]["initialization"] = checkpoint["initialization"]
    checkpoint["train_config"]["initialization_checkpoint_sha256"] = "a" * 64
    torch.save(checkpoint, synthetic["source"])
    run(synthetic)
    candidate = torch.load(synthetic["output"], map_location="cpu", weights_only=True)
    assert candidate["initialization"] == "continuation_of_own_scratch_training"
    assert candidate["train_config"]["initialization_checkpoint_sha256"] == "a" * 64
    assert candidate["provenance"]["source_epoch"] == 17


@pytest.mark.parametrize("change", [
    "pretrained", "missing_pretrained", "train_pretrained", "initialization", "initialization_conflict",
    "continuation_sha", "manifest", "missing_manifest", "classes", "normalization", "resize",
    "ensemble", "epoch", "input_size", "dtype", "nan_parameter",
])
def test_source_guards_reject_before_opening_any_image(synthetic, monkeypatch, change):
    checkpoint = torch.load(synthetic["source"], map_location="cpu", weights_only=True)
    if change == "pretrained":
        checkpoint["pretrained"] = True
    elif change == "missing_pretrained":
        checkpoint.pop("pretrained")
    elif change == "train_pretrained":
        checkpoint["train_config"]["pretrained"] = True
    elif change == "initialization":
        checkpoint["initialization"] = "external_teacher"
    elif change == "initialization_conflict":
        checkpoint["train_config"]["initialization"] = "continuation_of_own_scratch_training"
    elif change == "continuation_sha":
        checkpoint["initialization"] = checkpoint["train_config"]["initialization"] = "continuation_of_own_scratch_training"
    elif change == "manifest":
        checkpoint["train_config"]["manifest_sha256"] = "0" * 64
    elif change == "missing_manifest":
        checkpoint["train_config"].pop("manifest_sha256")
    elif change == "classes":
        checkpoint["class_names"] = ["dog", "cat"]
    elif change == "normalization":
        checkpoint["normalization"] = {"mean": [0.1] * 3, "std": [0.5] * 3}
    elif change == "resize":
        checkpoint["resize_mode"] = "stretch"
    elif change == "ensemble":
        checkpoint["model_config"]["name"] = "logit_ensemble"
    elif change == "epoch":
        checkpoint["epoch"] = 0
    elif change == "input_size":
        checkpoint["image_size"] = 32
    elif change == "dtype":
        checkpoint["state_dict"]["scale"] = checkpoint["state_dict"]["scale"].double()
    else:
        checkpoint["state_dict"]["scale"][0] = float("nan")
    torch.save(checkpoint, synthetic["source"])

    def forbidden_open(*args, **kwargs):
        pytest.fail("A rejected source must not open training or held-out images")

    monkeypatch.setattr(Image, "open", forbidden_open)
    with pytest.raises((ValueError, RuntimeError)):
        run(synthetic)
    assert not synthetic["output"].exists()


@pytest.mark.parametrize("change", ["path", "group", "hash", "outside"])
def test_manifest_heldout_overlap_guards_precede_image_reads(synthetic, monkeypatch, change):
    rows = [dict(row) for row in synthetic["rows"]]
    if change == "path":
        rows[-1]["relative_path"] = rows[0]["relative_path"]
    elif change == "group":
        rows[-1]["group_id"] = rows[0]["group_id"]
    elif change == "hash":
        rows[-1]["sha256"] = rows[0]["sha256"]
    else:
        rows[0]["relative_path"] = "../../outside.png"
    write_manifest(synthetic["manifest"], rows)
    checkpoint = torch.load(synthetic["source"], map_location="cpu", weights_only=True)
    checkpoint["train_config"]["manifest_sha256"] = hashlib.sha256(synthetic["manifest"].read_bytes()).hexdigest()
    torch.save(checkpoint, synthetic["source"])
    monkeypatch.setattr(Image, "open", lambda *args, **kwargs: pytest.fail("Overlap must fail before image access"))
    with pytest.raises(ValueError):
        run(synthetic)
    assert not synthetic["output"].exists()


@pytest.mark.parametrize("existing", ["ordinary", "source", "dangling_symlink"])
def test_existing_output_is_never_overwritten_or_read_as_a_candidate(synthetic, monkeypatch, existing):
    if existing == "ordinary":
        synthetic["output"].write_bytes(b"existing candidate")
        expected = synthetic["output"].read_bytes()
    elif existing == "source":
        synthetic["output"] = synthetic["source"]
        expected = synthetic["source"].read_bytes()
    else:
        synthetic["output"].symlink_to(synthetic["root"] / "missing.pt")
        expected = None
    monkeypatch.setattr(Image, "open", lambda *a, **k: pytest.fail("Existing output must fail before images"))
    with pytest.raises(FileExistsError):
        run(synthetic)
    if expected is None:
        assert synthetic["output"].is_symlink()
        assert not (synthetic["root"] / "missing.pt").exists()
    else:
        assert synthetic["output"].read_bytes() == expected


def test_exclusive_output_create_preserves_a_concurrent_writers_file(synthetic, monkeypatch):
    original_open = Image.open

    def other_writer_wins(path, *args, **kwargs):
        synthetic["output"].write_bytes(b"other writer owns this candidate")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Image, "open", other_writer_wins)
    with pytest.raises(FileExistsError):
        run(synthetic)
    assert synthetic["output"].read_bytes() == b"other writer owns this candidate"


@pytest.mark.parametrize("mutation", ["parameter", "other_buffer"])
def test_unexpected_forward_mutations_cannot_be_saved(synthetic, monkeypatch, mutation):
    class BadForward(ProbeModel):
        def forward(self, images):
            if mutation == "parameter":
                self.scale.add_(0.01)
            else:
                self.unchanged_marker.add_(1)
            return super().forward(images)

    monkeypatch.setattr(recalibration, "build_model", lambda **config: BadForward())
    with pytest.raises(RuntimeError, match="Only BatchNorm running buffers"):
        run(synthetic)
    assert not synthetic["output"].exists()


def test_actual_tiny_cnn_roundtrip_preserves_all_learned_weights(synthetic, monkeypatch):
    config = {"name": "tiny_resnet18", "width": 8, "num_classes": 2, "dropout": 0.15}
    checkpoint = torch.load(synthetic["source"], map_location="cpu", weights_only=True)
    model = build_model(**config)
    checkpoint["model_config"], checkpoint["state_dict"] = config, model.state_dict()
    torch.save(checkpoint, synthetic["source"])
    monkeypatch.setattr(recalibration, "build_model", build_model)
    run(synthetic)
    candidate = torch.load(synthetic["output"], map_location="cpu", weights_only=True)
    restored = build_model(**candidate["model_config"]).eval()
    restored.load_state_dict(candidate["state_dict"], strict=True)
    for name, parameter in model.named_parameters():
        assert raw_bytes(parameter) == raw_bytes(candidate["state_dict"][name]), name
    with torch.inference_mode():
        assert torch.isfinite(restored(torch.zeros(1, 3, 64, 64))).all()
