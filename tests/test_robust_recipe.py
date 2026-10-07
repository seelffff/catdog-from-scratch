"""Exercise the V2 recipe without training on or inspecting the held-out data."""

import csv
import hashlib
import json
import sys

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision import transforms

from catsdogs.model import RobustResNet, TinyResNet, build_model
from catsdogs.preprocessing import preprocess_image
from catsdogs.train import ImageDataset, NORMALIZATION, main, seed_everything, training_augmentation


torch.set_num_threads(1)


def test_original_default_model_keeps_legacy_checkpoint_shapes():
    model = build_model()
    assert isinstance(model, TinyResNet)
    assert sum(parameter.numel() for parameter in model.parameters()) == 2_795_554
    state = model.state_dict()
    assert state["stem.0.weight"].shape == (32, 3, 3, 3)
    assert state["stages.2.skip.0.weight"].shape == (64, 32, 1, 1)
    assert state["classifier.weight"].shape == (2, 256)
    assert not any("attention" in name for name in state)


def test_robust_recipe_backpropagates_through_attention_and_updates_parameters():
    seed_everything(2027)
    model = build_model(name="robust_resnet18", width=8, dropout=0.1)
    assert isinstance(model, RobustResNet)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    before = model.stem[0].weight.detach().clone()
    images = torch.randn(4, 3, 65, 73)
    labels = torch.tensor([0, 1, 0, 1])
    logits = model(images)
    assert logits.shape == (4, 2)
    loss = torch.nn.functional.cross_entropy(logits, labels, label_smoothing=0.02)
    assert torch.isfinite(loss)
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    attention_gradients = [parameter.grad for name, parameter in model.named_parameters()
                           if "attention" in name]
    assert sum(float(gradient.abs().sum()) for gradient in attention_gradients) > 0
    optimizer.step()
    assert not torch.equal(model.stem[0].weight, before)


@pytest.mark.parametrize("ensemble", [False, True])
def test_robust_checkpoint_roundtrip_preserves_exact_inference(tmp_path, ensemble):
    robust = {"name": "robust_resnet18", "width": 8, "dropout": 0.0}
    config = ({"name": "logit_ensemble", "members": [robust, {"width": 8}],
               "mixture_weights": [3, 1]} if ensemble else robust)
    seed_everything(19)
    model = build_model(**config).eval()
    inputs = torch.randn(2, 3, 67, 79)
    with torch.inference_mode():
        expected = model(inputs)
    checkpoint = tmp_path / "robust.pt"
    torch.save({"model_config": config, "state_dict": model.state_dict(),
                "pretrained": False}, checkpoint)
    loaded = torch.load(checkpoint, map_location="cpu", weights_only=True)
    restored = build_model(**loaded["model_config"]).eval()
    restored.load_state_dict(loaded["state_dict"], strict=True)
    with torch.inference_mode():
        assert torch.equal(restored(inputs), expected)


@pytest.fixture
def small_manifest(tmp_path):
    directory = tmp_path / "data"
    directory.mkdir()
    y, x = np.mgrid[:37, :83]
    pixels = np.stack((x * 3 % 256, y * 7 % 256, (x + y) * 2 % 256), axis=-1).astype(np.uint8)
    Image.fromarray(pixels).save(directory / "cat.png")
    manifest = directory / "splits_v2.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["relative_path", "label", "class_name", "split",
                                                    "width", "height"])
        writer.writeheader()
        for split in ("train", "val"):
            writer.writerow({"relative_path": "data/cat.png", "label": 0, "class_name": "cat",
                             "split": split, "width": 83, "height": 37})
    return manifest


@pytest.mark.parametrize("recipe", ["baseline", "robust", "weak"])
def test_augmentation_is_seeded_finite_and_varies_training_images(small_manifest, recipe):
    dataset = ImageDataset(small_manifest, "train", 64, augment=True,
                           augmentation_recipe=recipe)
    seed_everything(41)
    first, label = dataset[0]
    seed_everything(41)
    repeated, repeated_label = dataset[0]
    assert label == repeated_label == 0
    assert torch.equal(first, repeated)
    variants = []
    for seed in range(12):
        seed_everything(seed)
        tensor, _ = dataset[0]
        assert tensor.shape == (3, 64, 64)
        assert tensor.dtype == torch.float32
        assert torch.isfinite(tensor).all()
        assert tensor.min() >= -1 and tensor.max() <= 1
        variants.append(tensor)
    assert any(not torch.equal(variants[0], tensor) for tensor in variants[1:])


@pytest.mark.parametrize("recipe", ["baseline", "robust", "weak"])
def test_training_choice_keeps_validation_identical_to_serving(small_manifest, recipe):
    dataset = ImageDataset(small_manifest, "val", 64, augmentation_recipe=recipe)
    actual, label = dataset[0]
    with Image.open(small_manifest.parent / "cat.png") as image:
        expected = preprocess_image(image, 64, NORMALIZATION)
    assert label == 0
    assert torch.equal(actual, expected)
    assert torch.equal(dataset[0][0], expected)


def test_default_training_recipe_still_matches_explicit_baseline(small_manifest):
    default = ImageDataset(small_manifest, "train", 64, augment=True)
    baseline = ImageDataset(small_manifest, "train", 64, augment=True, augmentation_recipe="baseline")
    for seed in (0, 5, 17):
        seed_everything(seed)
        expected = baseline[0][0]
        seed_everything(seed)
        assert torch.equal(default[0][0], expected)
    # The optional full-frame branch is confined to the new weak recipe.
    assert not any(isinstance(transform, transforms.RandomApply)
                   for recipe in ("baseline", "robust")
                   for transform in training_augmentation(64, recipe).transforms)


def test_weak_recipe_full_frame_branch_preserves_every_pixel_before_photo_jitter(small_manifest, monkeypatch):
    dataset = ImageDataset(small_manifest, "train", 64, augment=True, augmentation_recipe="weak")
    pipeline = dataset.augmentation.transforms

    def unexpected_crop(image):
        pytest.fail("The seeded full-frame branch should skip RandomResizedCrop")

    monkeypatch.setattr(pipeline[0].transforms[0], "forward", unexpected_crop)
    # Isolate framing from the separately tested flip and colour augmentation.
    monkeypatch.setattr(pipeline[1], "forward", lambda image: image)
    monkeypatch.setattr(pipeline[2], "forward", lambda image: image)
    seed_everything(1)  # First torch random value is above the crop probability 0.5.
    actual, _ = dataset[0]
    with Image.open(small_manifest.parent / "cat.png") as image:
        expected = preprocess_image(image, 64, NORMALIZATION)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("changed", ["manifest", "architecture", "missing_manifest"])
def test_continuation_requires_same_audited_split_and_architecture(tmp_path, monkeypatch, changed):
    manifest = tmp_path / "splits_v2.csv"
    manifest.write_text("synthetic manifest marker", encoding="utf-8")
    config = {"name": "robust_resnet18", "num_classes": 2, "width": 8, "dropout": 0.15}
    model = build_model(**config)
    checkpoint = {"model_config": config, "state_dict": model.state_dict(), "pretrained": False,
                  "train_config": {"manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()}}
    if changed == "manifest":
        checkpoint["train_config"]["manifest_sha256"] = "b" * 64
    elif changed == "missing_manifest":
        checkpoint["train_config"] = {}
    else:
        checkpoint["model_config"] = {**config, "name": "tiny_resnet18"}
    source = tmp_path / "source.pt"
    torch.save(checkpoint, source)
    monkeypatch.setattr(sys, "argv", ["train", "--device", "cpu", "--manifest", str(manifest),
                                     "--output", str(tmp_path / "continuation"), "--width", "8",
                                     "--architecture", "robust_resnet18", "--augmentation", "weak",
                                     "--init-checkpoint", str(source)])
    message = "compatible scratch-trained" if changed == "architecture" else "original audited"
    with pytest.raises(ValueError, match=message):
        main()


def test_weak_recipe_and_own_continuation_are_recorded_before_training(small_manifest, tmp_path, monkeypatch):
    import catsdogs.train as train_module

    class NoTrainingRequested(Exception):
        pass

    def refuse_batches(loader):
        raise NoTrainingRequested

    config = {"name": "robust_resnet18", "num_classes": 2, "width": 8, "dropout": 0.15}
    source = tmp_path / "own_v2.pt"
    torch.save({"model_config": config, "state_dict": build_model(**config).state_dict(),
                "pretrained": False, "train_config": {
                    "manifest_sha256": hashlib.sha256(small_manifest.read_bytes()).hexdigest()}}, source)
    output = tmp_path / "recorded_recipe"
    monkeypatch.setattr(train_module.DataLoader, "__iter__", refuse_batches)
    monkeypatch.setattr(sys, "argv", ["train", "--device", "cpu", "--manifest", str(small_manifest),
                                     "--output", str(output), "--workers", "0", "--width", "8",
                                     "--architecture", "robust_resnet18", "--augmentation", "weak",
                                     "--init-checkpoint", str(source)])
    with pytest.raises(NoTrainingRequested):
        main()
    recorded = json.loads((output / "config.json").read_text(encoding="utf-8"))
    assert recorded["augmentation"] == "weak"
    assert recorded["architecture"] == "robust_resnet18"
    assert recorded["initialization"] == "continuation_of_own_scratch_training"
    assert recorded["initialization_checkpoint_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert recorded["manifest_sha256"] == hashlib.sha256(small_manifest.read_bytes()).hexdigest()
    assert not (output / "history.json").exists()


@pytest.mark.parametrize("value", ["-0.01", "1", "nan", "inf"])
def test_invalid_label_smoothing_rejected_before_creating_experiment(tmp_path, monkeypatch, value):
    output = tmp_path / "experiment"
    monkeypatch.setattr(sys, "argv", ["train", "--device", "cpu", "--output", str(output),
                                     "--label-smoothing", value])
    with pytest.raises(ValueError, match="Label smoothing"):
        main()
    assert not output.exists()
