"""Synthetic frozen checkpoints verify exact inference without real dataset images."""

from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import math

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from app.config import Settings
from app.model.loader import load_predictor
from app.model._scratch_architecture import build_model
from app.model._scratch_preprocessing import preprocess_image
from app.model.scratch_v2 import ScratchPredictor, V2_MANIFEST_SHA256


def config(name="tiny_resnet18"):
    return {"name": name, "num_classes": 2, "width": 8, "dropout": 0.15}


def checkpoint_for(model_config, tta=False):
    torch.manual_seed(73)
    torch.set_num_threads(1)
    model = build_model(**model_config).eval()
    initialization = "own_model_ensemble" if model_config["name"] == "logit_ensemble" else "random"
    checkpoint = {
        "state_dict": model.state_dict(), "model_config": model_config,
        "train_config": {"pretrained": False, "initialization": initialization,
                         "manifest_sha256": V2_MANIFEST_SHA256},
        "initialization": initialization, "pretrained": False,
        "selection_frozen_before_test": True, "candidate_validation_pending": False,
        "class_names": ["cat", "dog"], "resize_mode": "letterbox", "image_size": 64,
        "normalization": {"mean": [0.5] * 3, "std": [0.5] * 3},
        "temperature": 0.73, "threshold": 0.5, "tta_horizontal_flip": tta,
    }
    return checkpoint, model


def save(tmp_path, checkpoint):
    path = tmp_path / "synthetic_final.pt"
    torch.save(checkpoint, path)
    return path


def image(seed=12):
    pixels = np.random.default_rng(seed).integers(0, 256, (57, 93, 3), dtype=np.uint8)
    return Image.fromarray(pixels)


@pytest.mark.parametrize("model_config,tta", [
    (config(), False), (config("robust_resnet18"), True),
    ({"name": "logit_ensemble", "members": [config(), config("robust_resnet18")],
      "mixture_weights": [0.3, 0.7]}, True),
])
def test_probability_matches_saved_architecture_temperature_and_probability_tta(tmp_path, model_config, tta):
    checkpoint, reference = checkpoint_for(model_config, tta)
    path = save(tmp_path, checkpoint)
    predictor = ScratchPredictor(path, expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    source = image()
    tensor = preprocess_image(source, 64, checkpoint["normalization"]).unsqueeze(0)
    with torch.inference_mode():
        expected = (reference(tensor) / checkpoint["temperature"]).softmax(1)
        if tta:
            expected = (expected + (reference(torch.flip(tensor, (3,))) / checkpoint["temperature"]).softmax(1)) / 2
    result = predictor.predict(source)
    assert result.p_cat == pytest.approx(expected[0, 0].item(), abs=1e-7)
    assert result.p_dog == expected[0, 1].item()
    assert predictor.name == "scratch-v2" and next(predictor.model.parameters()).device.type == "cpu"
    assert predictor.model.training is False and torch.get_num_threads() == 1


def test_unflipped_first_member_maps_are_bounded_and_do_not_change_prediction(tmp_path):
    ensemble = {"name": "logit_ensemble", "members": [config(), config("robust_resnet18")],
                "mixture_weights": [0.6, 0.4]}
    checkpoint, reference = checkpoint_for(ensemble, tta=True)
    checkpoint["image_size"] = 160
    path = save(tmp_path, checkpoint)
    predictor = ScratchPredictor(path)
    without_maps = ScratchPredictor(path, collect_activations=False)
    source = image()
    tensor = preprocess_image(source, 160, checkpoint["normalization"]).unsqueeze(0)
    with torch.inference_mode():
        raw_first = reference.members[0].stem[0](tensor)[:, :4].clone()
        expected = torch.nn.functional.adaptive_avg_pool2d(raw_first, (32, 32))[0].numpy()
    result = predictor.predict(source)
    assert result.p_cat == pytest.approx(without_maps.predict(source).p_cat, abs=1e-7)
    assert set(result.activations) == {"conv1", "relu1", "pool1"}
    assert np.asarray(result.activations["conv1"]) == pytest.approx(expected, abs=1e-6)
    assert "first member" in predictor.activation_source and "unflipped" in predictor.activation_source
    for maps in result.activations.values():
        array = np.asarray(maps)
        assert array.ndim == 3 and array.shape[0] <= 4 and max(array.shape[1:]) <= 32
        assert np.isfinite(array).all()


def test_hook_snapshot_keeps_negative_pre_relu_values(tmp_path):
    checkpoint, _ = checkpoint_for(config())
    predictor = ScratchPredictor(save(tmp_path, checkpoint))
    # Identity BN makes conv output share storage with the following inplace ReLU.
    predictor.model.stem[1] = nn.Identity()
    result = predictor.predict(image())
    assert np.min(result.activations["conv1"]) < 0
    assert np.min(result.activations["relu1"]) >= 0


def test_concurrent_predict_calls_do_not_mix_hook_snapshots(tmp_path):
    checkpoint, _ = checkpoint_for(config(), tta=True)
    predictor = ScratchPredictor(save(tmp_path, checkpoint))
    sources = [image(1), image(2)]
    expected = [predictor.predict(source) for source in sources]
    with ThreadPoolExecutor(max_workers=2) as executor:
        actual = list(executor.map(predictor.predict, sources))
    for first, second in zip(expected, actual, strict=True):
        assert first.p_cat == pytest.approx(second.p_cat, abs=1e-7)
        assert first.activations == second.activations


@pytest.mark.parametrize("mutation", [
    "v1_manifest", "pretrained", "unfrozen", "pending", "class_order", "temperature", "threshold", "provenance",
])
def test_non_final_or_wrong_provenance_checkpoint_is_refused(tmp_path, mutation):
    checkpoint, _ = checkpoint_for(config())
    checkpoint = copy.deepcopy(checkpoint)
    if mutation == "v1_manifest": checkpoint["train_config"]["manifest_sha256"] = "0" * 64
    elif mutation == "pretrained": checkpoint["pretrained"] = True
    elif mutation == "unfrozen": checkpoint["selection_frozen_before_test"] = False
    elif mutation == "pending": checkpoint["candidate_validation_pending"] = True
    elif mutation == "class_order": checkpoint["class_names"] = ["dog", "cat"]
    elif mutation == "temperature": checkpoint["temperature"] = math.nan
    elif mutation == "threshold": checkpoint["threshold"] = 0.6
    elif mutation == "provenance": checkpoint["train_config"]["initialization"] = "unknown"
    with pytest.raises(ValueError):
        ScratchPredictor(save(tmp_path, checkpoint))


def test_expected_hash_rejected_before_checkpoint_loading(tmp_path, monkeypatch):
    checkpoint, _ = checkpoint_for(config())
    path = save(tmp_path, checkpoint)
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: pytest.fail("Wrong SHA must reject first"))
    with pytest.raises(ValueError, match="MODEL_SHA256"):
        ScratchPredictor(path, expected_sha256="0" * 64)


def test_nonfinite_logits_are_not_reported_as_a_class(tmp_path):
    checkpoint, _ = checkpoint_for(config())
    predictor = ScratchPredictor(save(tmp_path, checkpoint), collect_activations=False)
    class Broken(nn.Module):
        def forward(self, _tensor):
            return torch.tensor([[math.nan, 1.0]])
    predictor.model = Broken()
    with pytest.raises(ValueError, match="finite"):
        predictor.predict(image())


def test_loader_selects_scratch_predictor_and_optional_sha(tmp_path, monkeypatch):
    checkpoint, _ = checkpoint_for(config())
    path = save(tmp_path, checkpoint)
    monkeypatch.setenv("MODEL_SHA256", hashlib.sha256(path.read_bytes()).hexdigest())
    predictor = load_predictor(Settings(model_path=path))
    assert predictor.name == "scratch-v2"
    assert math.isfinite(predictor.predict(image()).p_cat)
    assert load_predictor(Settings()).name == "dummy"
