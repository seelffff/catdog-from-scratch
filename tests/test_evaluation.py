"""Synthetic checks of reported metrics and the inference contract; no holdout access."""

import numpy as np
import pytest
import torch
from PIL import Image

from catsdogs.evaluate import metrics, probabilities, save_predictions, wilson
from catsdogs.inference import Predictor


def predictor_with_model(tmp_path, monkeypatch, model, temperature=1.0, horizontal_flip=False):
    import catsdogs.model as model_module

    monkeypatch.setattr(model_module, "build_model", lambda **config: model)
    path = tmp_path / "synthetic.pt"
    torch.save({
        "model_config": {}, "state_dict": {}, "image_size": 32,
        "class_names": ["cat", "dog"],
        "normalization": {"mean": [0.5] * 3, "std": [0.5] * 3},
        "resize_mode": "letterbox", "temperature": temperature,
        "threshold": 0.5, "tta_horizontal_flip": horizontal_flip,
    }, path)
    return Predictor(path)


def test_tta_temperature_probabilities_match_serving(tmp_path, monkeypatch):
    class OrientationModel(torch.nn.Module):
        def forward(self, batch):
            left = batch[:, 0, :, :16].mean(dim=(1, 2))
            right = batch[:, 0, :, 16:].mean(dim=(1, 2))
            canonical = torch.tensor([5.0, 0.0], device=batch.device)
            flipped = torch.tensor([0.0, 1.0], device=batch.device)
            return torch.where((left > right)[:, None], canonical, flipped)

    predictor = predictor_with_model(tmp_path, monkeypatch, OrientationModel(),
                                     temperature=2.0, horizontal_flip=True)
    image = Image.new("RGB", (32, 32), (0, 0, 0))
    image.paste((255, 0, 0), (0, 0, 16, 32))
    served = predictor.predict_image(image)
    expected = probabilities(np.array([[5.0, 0.0]]), np.array([[0.0, 1.0]]), temperature=2.0)[0]
    assert served["probabilities"]["cat"] == pytest.approx(expected[0], abs=1e-6)
    assert served["probabilities"]["dog"] == pytest.approx(expected[1], abs=1e-6)
    assert served["class_index"] == int(expected.argmax())


def test_exact_tie_uses_same_class_in_service_and_metrics(tmp_path, monkeypatch):
    class TieModel(torch.nn.Module):
        def forward(self, batch):
            return torch.zeros((len(batch), 2), device=batch.device)

    predictor = predictor_with_model(tmp_path, monkeypatch, TieModel())
    served = predictor.predict_image(Image.new("RGB", (32, 32)))
    probs, labels = np.full((2, 2), 0.5), np.array([0, 1])
    reported = metrics(probs, labels)
    # Row zero contains the metric's decision for a known-cat example.
    assert reported["confusion_matrix"][0][served["class_index"]] == 1
    saved = save_predictions(tmp_path / "synthetic_predictions.csv", [
        {"relative_path": "cat.synthetic.png", "group_id": "cat_synthetic"},
        {"relative_path": "dog.synthetic.png", "group_id": "dog_synthetic"},
    ], labels, probs)
    assert all(row["predicted_class"] == served["class_name"] for row in saved)


def test_modified_manifest_is_rejected_before_loading_images(tmp_path, monkeypatch):
    import catsdogs.evaluate as evaluation

    checkpoint = tmp_path / "synthetic.pt"
    manifest = tmp_path / "synthetic.csv"
    torch.save({"pretrained": False, "train_config": {"manifest_sha256": "0" * 64}}, checkpoint)
    manifest.write_text("changed synthetic manifest", encoding="utf-8")
    monkeypatch.setattr("sys.argv", [
        "evaluate", "--checkpoint", str(checkpoint), "--manifest", str(manifest),
        "--device", "cpu", "--output", str(tmp_path / "reports"),
    ])

    def forbidden_image_collection(*args, **kwargs):
        raise AssertionError("Images must not be loaded with a changed split")

    monkeypatch.setattr(evaluation, "collect", forbidden_image_collection)
    with pytest.raises(ValueError, match="Manifest SHA-256 differs"):
        evaluation.main()


def test_binary_metrics_and_wilson_interval_have_expected_values():
    values = np.array([[0.9, 0.1], [0.4, 0.6], [0.2, 0.8], [0.7, 0.3]])
    result = metrics(values, np.array([0, 0, 1, 1]))
    assert result["confusion_matrix"] == [[1, 1], [1, 1]]
    assert result["accuracy"] == 0.5
    assert result["balanced_accuracy"] == 0.5
    assert result["macro_f1"] == 0.5
    assert result["roc_auc"] == 0.75
    assert wilson(0, 10) == pytest.approx([0.0, 0.2775327998628892], abs=1e-12)
    assert wilson(10, 10) == pytest.approx([0.7224672001371107, 1.0], abs=1e-12)
