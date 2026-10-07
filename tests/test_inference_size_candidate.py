"""Changing serving resolution must neither alter weights nor inherit old calibration."""

import hashlib
import importlib.util
import io
from pathlib import Path

import pytest
import torch
from PIL import Image

from catsdogs.model import build_model


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/set_inference_size.py"
spec = importlib.util.spec_from_file_location("inference_size_candidate", SCRIPT)
size_script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(size_script)


@pytest.fixture(autouse=True)
def synthetic_only(monkeypatch):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)

    def no_images(*args, **kwargs):
        pytest.fail("Candidate formation must never open images")

    monkeypatch.setattr(Image, "open", no_images)
    yield
    torch.set_num_threads(old_threads)


def source_checkpoint(path, architecture="robust_resnet18"):
    config = {"name": architecture, "width": 8, "num_classes": 2, "dropout": 0.15}
    if architecture in {"mixed", "robust_pair"}:
        config = {"name": "logit_ensemble", "members": [
            {"name": "robust_resnet18", "width": 8, "dropout": 0.15},
            {"name": "tiny_resnet18" if architecture == "mixed" else "robust_resnet18", "width": 12},
        ], "mixture_weights": [0.25, 0.75]}
    with torch.random.fork_rng():
        torch.manual_seed(804)
        model = build_model(**config).eval()
    with torch.no_grad():
        single = model.members[0] if hasattr(model, "members") else model
        single.classifier.bias[0] = -0.0
        single.classifier.bias[1] = 1.5
    initialization = "own_model_ensemble" if config["name"] == "logit_ensemble" else "random"
    checkpoint = {
        "model_config": config, "state_dict": model.state_dict(), "image_size": 192,
        "class_names": ["cat", "dog"], "normalization": {"mean": [0.5] * 3, "std": [0.5] * 3},
        "resize_mode": "letterbox", "pretrained": False, "epoch": 6, "val_source": "ema",
        "train_config": {"manifest_sha256": "a" * 64, "pretrained": False, "initialization": initialization},
        "temperature": 0.37, "threshold": 0.71, "tta_horizontal_flip": True,
        "selection_frozen_before_test": True, "selection_frozen": True,
        "candidate_validation_pending": False,
        "validation": {"accuracy": 0.99}, "calibration": {"temperature": 0.37},
        "selection": {"frozen_at_utc": "old"},
        "provenance": {"method": "old_synthetic_operation", "source_sha256": "c" * 64},
    }
    # Existing make_ensemble checkpoints record initialization only inside train_config.
    if config["name"] != "logit_ensemble":
        checkpoint["initialization"] = initialization
    torch.save(checkpoint, path)
    return checkpoint


def assert_same_tensor_bytes(first, second):
    assert first.keys() == second.keys()
    for key, expected in first.items():
        actual = second[key]
        assert actual.shape == expected.shape and actual.dtype == expected.dtype
        assert torch.equal(actual.contiguous().flatten().view(torch.uint8),
                           expected.contiguous().flatten().view(torch.uint8)), key


@pytest.mark.parametrize("architecture", ["tiny_resnet18", "robust_resnet18", "mixed", "robust_pair"])
def test_native_single_or_ensemble_candidate_preserves_every_tensor_and_resets_selection(tmp_path, architecture):
    source, output = tmp_path / "source.pt", tmp_path / "candidate.pt"
    original = source_checkpoint(source, architecture)
    original_bytes = source.read_bytes()
    report = size_script.set_inference_size(source, 256, output)
    candidate = torch.load(output, map_location="cpu", weights_only=True)
    assert_same_tensor_bytes(original["state_dict"], candidate["state_dict"])
    assert candidate["model_config"] == original["model_config"]
    assert candidate["train_config"] == original["train_config"]
    assert candidate["epoch"] == original["epoch"]
    assert candidate["image_size"] == 256
    assert candidate["temperature"] == 1.0 and candidate["threshold"] == 0.5
    assert candidate["tta_horizontal_flip"] is False
    assert candidate["candidate_validation_pending"] is True
    assert candidate["selection_frozen_before_test"] is False and candidate["selection_frozen"] is False
    assert not {"validation", "calibration", "selection"} & candidate.keys()
    provenance = candidate["provenance"]
    assert provenance["source_provenance"] == original["provenance"]
    assert provenance["source_sha256"] == hashlib.sha256(original_bytes).hexdigest()
    assert provenance["source_checkpoint_sha256"] == provenance["source_sha256"]
    assert provenance["original_image_size"] == 192 and provenance["candidate_image_size"] == 256
    assert provenance["manifest_sha256"] == "a" * 64
    assert provenance["validation_required"] is True and report["validation_required"] is True
    assert source.read_bytes() == original_bytes


def test_manifest_verification_and_v1_default_name_remain_optional(tmp_path):
    source, output, manifest = (tmp_path / name for name in ["legacy_v1.pt", "candidate.pt", "splits.csv"])
    checkpoint = source_checkpoint(source, "tiny_resnet18")
    checkpoint["model_config"].pop("name")
    manifest.write_text("synthetic manifest; no image paths are consulted\n")
    checkpoint["train_config"]["manifest_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    torch.save(checkpoint, source)
    size_script.set_inference_size(source, 224, output, manifest)
    candidate = torch.load(output, map_location="cpu", weights_only=True)
    assert candidate["model_config"] == checkpoint["model_config"]
    assert_same_tensor_bytes(checkpoint["state_dict"], candidate["state_dict"])


def test_source_snapshot_is_read_once_and_never_loaded_again_from_the_path(tmp_path, monkeypatch):
    source, output = tmp_path / "source.pt", tmp_path / "candidate.pt"
    original = source_checkpoint(source)
    payload = source.read_bytes()
    original_read, original_load = Path.read_bytes, torch.load
    reads = []

    def replace_after_snapshot(path):
        reads.append(path)
        if path == source:
            source.write_bytes(b"another writer replaced the source")
            return payload
        return original_read(path)

    def load_snapshot(handle, *args, **kwargs):
        assert isinstance(handle, io.BytesIO)
        return original_load(handle, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "read_bytes", replace_after_snapshot)
        patch.setattr(torch, "load", load_snapshot)
        size_script.set_inference_size(source, 256, output)
    assert reads == [source]
    candidate = torch.load(output, map_location="cpu", weights_only=True)
    assert_same_tensor_bytes(original["state_dict"], candidate["state_dict"])
    assert candidate["provenance"]["source_sha256"] == hashlib.sha256(payload).hexdigest()


def test_confirmed_own_continuation_retains_its_training_provenance(tmp_path):
    source, output = tmp_path / "source.pt", tmp_path / "candidate.pt"
    checkpoint = source_checkpoint(source)
    checkpoint["initialization"] = "continuation_of_own_scratch_training"
    checkpoint["train_config"]["initialization"] = checkpoint["initialization"]
    checkpoint["train_config"]["initialization_checkpoint_sha256"] = "d" * 64
    torch.save(checkpoint, source)
    size_script.set_inference_size(source, 256, output)
    candidate = torch.load(output, map_location="cpu", weights_only=True)
    assert candidate["initialization"] == checkpoint["initialization"]
    assert candidate["train_config"] == checkpoint["train_config"]
    assert_same_tensor_bytes(checkpoint["state_dict"], candidate["state_dict"])


@pytest.mark.parametrize("size", [63, 1025, 0, -1, 256.0, True, None])
def test_invalid_requested_size_is_rejected_without_reading_source(tmp_path, monkeypatch, size):
    def no_read(path):
        pytest.fail("Invalid sizes must fail before reading any checkpoint")

    monkeypatch.setattr(Path, "read_bytes", no_read)
    with pytest.raises(ValueError):
        size_script.set_inference_size(tmp_path / "unused.pt", size, tmp_path / "candidate.pt")


@pytest.mark.parametrize("change", [
    "pretrained", "missing_pretrained", "train_pretrained", "train_missing_pretrained",
    "manifest_missing", "manifest_invalid", "classes", "normalization", "resize_mode",
    "model_unknown", "nested", "ensemble_weights", "model_classes", "model_width", "model_dropout",
    "initialization", "train_initialization", "conflicting_initialization", "continuation_missing_sha", "source_size", "state_dict",
])
def test_invalid_source_metadata_never_creates_output(tmp_path, change):
    source, output = tmp_path / "source.pt", tmp_path / "new_dir/candidate.pt"
    checkpoint = source_checkpoint(source)
    if change == "pretrained":
        checkpoint["pretrained"] = True
    elif change == "missing_pretrained":
        checkpoint.pop("pretrained")
    elif change == "train_pretrained":
        checkpoint["train_config"]["pretrained"] = True
    elif change == "train_missing_pretrained":
        checkpoint["train_config"].pop("pretrained")
    elif change == "manifest_missing":
        checkpoint["train_config"].pop("manifest_sha256")
    elif change == "manifest_invalid":
        checkpoint["train_config"]["manifest_sha256"] = "not a SHA"
    elif change == "classes":
        checkpoint["class_names"] = ["dog", "cat"]
    elif change == "normalization":
        checkpoint["normalization"]["mean"] = [0.1] * 3
    elif change == "resize_mode":
        checkpoint["resize_mode"] = "center_crop"
    elif change == "model_unknown":
        checkpoint["model_config"]["name"] = "other_network"
    elif change == "nested":
        checkpoint["model_config"] = {"name": "logit_ensemble", "members": [
            {"name": "logit_ensemble", "members": [{"width": 8}, {"width": 8}]}, {"width": 8}
        ]}
    elif change == "ensemble_weights":
        checkpoint["model_config"] = {"name": "logit_ensemble", "members": [{"width": 8}, {"width": 8}],
                                      "mixture_weights": [1, float("nan")]}
    elif change == "model_classes":
        checkpoint["model_config"]["num_classes"] = 3
    elif change == "model_width":
        checkpoint["model_config"]["width"] = 7
    elif change == "model_dropout":
        checkpoint["model_config"]["dropout"] = float("nan")
    elif change == "initialization":
        checkpoint["initialization"] = "external_teacher"
    elif change == "train_initialization":
        checkpoint["initialization"] = None
        checkpoint["train_config"]["initialization"] = "external_teacher"
    elif change == "conflicting_initialization":
        checkpoint["train_config"]["initialization"] = "own_model_ensemble"
    elif change == "continuation_missing_sha":
        checkpoint["initialization"] = checkpoint["train_config"]["initialization"] = "continuation_of_own_scratch_training"
    elif change == "source_size":
        checkpoint["image_size"] = 1025
    else:
        checkpoint["state_dict"] = {"weight": "not a tensor"}
    torch.save(checkpoint, source)
    with pytest.raises(ValueError):
        size_script.set_inference_size(source, 256, output)
    assert not output.exists() and not output.parent.exists()


def test_wrong_optional_manifest_is_rejected_before_creating_output(tmp_path):
    source, output, manifest = (tmp_path / name for name in ["source.pt", "candidate.pt", "splits.csv"])
    source_checkpoint(source)
    manifest.write_text("different split\n")
    with pytest.raises(ValueError, match="manifest SHA"):
        size_script.set_inference_size(source, 256, output, manifest)
    assert not output.exists()


@pytest.mark.parametrize("existing", ["file", "source", "symlink", "dangling_symlink"])
def test_existing_file_source_or_symlink_is_refused_before_source_read(tmp_path, monkeypatch, existing):
    source, output = tmp_path / "source.pt", tmp_path / "candidate.pt"
    source_checkpoint(source)
    if existing == "file":
        output.write_bytes(b"keep me")
    elif existing == "source":
        output = source
    elif existing == "symlink":
        output.symlink_to(source)
    else:
        output.symlink_to(tmp_path / "missing.pt")

    def no_read(path):
        pytest.fail("Existing destination must be refused before reading any source")

    with monkeypatch.context() as patch:
        patch.setattr(Path, "read_bytes", no_read)
        with pytest.raises(FileExistsError):
            size_script.set_inference_size(source, 256, output)
    if existing == "file":
        assert output.read_bytes() == b"keep me"
    elif "symlink" in existing:
        assert output.is_symlink()
    assert source.exists()


def test_exclusive_output_preserves_a_concurrent_writer(tmp_path, monkeypatch):
    source, output = tmp_path / "source.pt", tmp_path / "candidate.pt"
    source_checkpoint(source)
    original_read = Path.read_bytes

    def competing_writer(path):
        payload = original_read(path)
        output.write_bytes(b"another writer won")
        return payload

    with monkeypatch.context() as patch:
        patch.setattr(Path, "read_bytes", competing_writer)
        with pytest.raises(FileExistsError):
            size_script.set_inference_size(source, 256, output)
    assert output.read_bytes() == b"another writer won"


def test_failed_save_removes_only_its_own_partial_output(tmp_path, monkeypatch):
    source, output = tmp_path / "source.pt", tmp_path / "candidate.pt"
    source_checkpoint(source)
    payload = source.read_bytes()

    def interrupted_save(checkpoint, handle):
        handle.write(b"partial candidate")
        raise OSError("synthetic write failure")

    monkeypatch.setattr(torch, "save", interrupted_save)
    with pytest.raises(OSError, match="write failure"):
        size_script.set_inference_size(source, 256, output)
    assert not output.exists() and source.read_bytes() == payload


def test_existing_cli_arguments_and_optional_manifest_call_the_same_function(tmp_path, monkeypatch, capsys):
    source, output = tmp_path / "source.pt", tmp_path / "candidate.pt"
    source_checkpoint(source, "tiny_resnet18")
    monkeypatch.setattr("sys.argv", [str(SCRIPT), "--checkpoint", str(source),
                                    "--image-size", "224", "--output", str(output)])
    size_script.main()
    assert capsys.readouterr().out.strip() == str(output)
    assert torch.load(output, map_location="cpu", weights_only=True)["image_size"] == 224
