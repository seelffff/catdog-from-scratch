import hashlib
import importlib.util
from pathlib import Path

import pytest
import torch
from PIL import Image

from catsdogs.inference import Predictor
from catsdogs.model import LogitEnsemble, build_model

torch.set_num_threads(1)
SCRIPT = Path(__file__).resolve().parents[1] / "scripts/make_ensemble.py"
spec = importlib.util.spec_from_file_location("make_ensemble", SCRIPT)
ensemble_script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ensemble_script)


class ConstantModel(torch.nn.Module):
    def __init__(self, values):
        super().__init__()
        self.register_buffer("values", torch.tensor(values, dtype=torch.float32))

    def forward(self, batch):
        return self.values.unsqueeze(0).expand(len(batch), -1)


def test_ensemble_computes_weighted_mean_of_logits_and_registers_weight_buffer():
    model = LogitEnsemble([ConstantModel([1, -2]), ConstantModel([-3, 4])], [1, 3])
    assert "mixture_weights" in model.state_dict()
    assert model.mixture_weights.tolist() == [0.25, 0.75]
    assert torch.equal(model(torch.zeros(2, 3, 8, 8)), torch.tensor([[-2.0, 2.5], [-2.0, 2.5]]))
    # Averaging class probabilities would be a different operation.
    averaged_probabilities = (torch.softmax(torch.tensor([1.0, -2.0]), 0)
                              + 3 * torch.softmax(torch.tensor([-3.0, 4.0]), 0)) / 4
    assert not torch.allclose(torch.softmax(model(torch.zeros(1, 3, 8, 8))[0], 0), averaged_probabilities)


@pytest.mark.parametrize("weights", [[0, 1], [-1, 2], [float("nan"), 1], [float("inf"), 1], [1]])
def test_ensemble_rejects_invalid_mixture_weights(weights):
    with pytest.raises(ValueError):
        LogitEnsemble([ConstantModel([0, 1]), ConstantModel([1, 0])], weights)


def test_ensemble_limits_members_and_rejects_nested_ensembles():
    member = ConstantModel([0, 1])
    with pytest.raises(ValueError):
        LogitEnsemble([member])
    with pytest.raises(ValueError):
        LogitEnsemble([member] * 4)
    nested = LogitEnsemble([member, member])
    with pytest.raises(ValueError):
        LogitEnsemble([nested, member])
    with pytest.raises(ValueError, match="Nested"):
        build_model(name="logit_ensemble", members=[{"name": "logit_ensemble"}, {"width": 8}])


def scratch_checkpoint(path, width, logits, image_size=160):
    config = {"name": "tiny_resnet18", "num_classes": 2, "width": width, "dropout": 0.0}
    model = build_model(**config).eval()
    with torch.no_grad():
        model.classifier.weight.zero_()
        model.classifier.bias.copy_(torch.tensor(logits))
    checkpoint = {"model_config": config, "state_dict": model.state_dict(), "image_size": image_size,
                  "class_names": ["cat", "dog"], "normalization": {"mean": [0.5] * 3, "std": [0.5] * 3},
                  "resize_mode": "letterbox", "epoch": 7, "val_source": "raw", "pretrained": False,
                  "train_config": {"manifest_sha256": "a" * 64}}
    torch.save(checkpoint, path)
    return checkpoint


def test_candidate_restores_members_weights_provenance_and_generic_prediction(tmp_path):
    first, second, output = tmp_path / "first.pt", tmp_path / "second.pt", tmp_path / "ensemble.pt"
    scratch_checkpoint(first, 8, [1.0, -2.0], image_size=160)
    scratch_checkpoint(second, 12, [-3.0, 4.0], image_size=192)
    source_digests = [hashlib.sha256(path.read_bytes()).hexdigest() for path in [first, second]]
    report = ensemble_script.make_candidate([first, second], output, 224, [1, 3])
    checkpoint = torch.load(output, map_location="cpu", weights_only=True)
    assert checkpoint["epoch"] == 0
    assert checkpoint["val_source"] == "own_model_ensemble"
    assert checkpoint["image_size"] == 224
    assert checkpoint["train_config"]["manifest_sha256"] == "a" * 64
    assert [member["sha256"] for member in report["provenance"]["members"]] == source_digests
    assert [member["image_size"] for member in report["provenance"]["members"]] == [160, 192]
    assert source_digests == [hashlib.sha256(path.read_bytes()).hexdigest() for path in [first, second]]
    restored = build_model(**checkpoint["model_config"]).eval()
    restored.load_state_dict(checkpoint["state_dict"], strict=True)
    assert restored.mixture_weights.tolist() == [0.25, 0.75]
    with torch.inference_mode():
        assert torch.allclose(restored(torch.zeros(1, 3, 80, 64)), torch.tensor([[-2.0, 2.5]]))
    prediction = Predictor(output).predict_image(Image.new("RGB", (32, 24)))
    assert prediction["probabilities"]["dog"] == pytest.approx(torch.softmax(torch.tensor([-2.0, 2.5]), 0)[1].item())


@pytest.mark.parametrize("change", ["pretrained", "normalization", "class_names", "resize_mode", "manifest"])
def test_candidate_rejects_incompatible_sources_without_creating_output(tmp_path, change):
    first, second, output = tmp_path / "first.pt", tmp_path / "second.pt", tmp_path / "ensemble.pt"
    scratch_checkpoint(first, 8, [0.0, 1.0])
    modified = scratch_checkpoint(second, 8, [1.0, 0.0])
    if change == "manifest":
        modified["train_config"]["manifest_sha256"] = "b" * 64
    elif change == "normalization":
        modified[change]["mean"] = [0.1] * 3
    elif change == "class_names":
        modified[change] = ["dog", "cat"]
    elif change == "pretrained":
        modified[change] = True
    else:
        modified[change] = "center_crop"
    torch.save(modified, second)
    with pytest.raises(ValueError):
        ensemble_script.make_candidate([first, second], output, 224)
    assert not output.exists()


def test_candidate_never_overwrites_an_existing_checkpoint(tmp_path):
    first, second, output = tmp_path / "first.pt", tmp_path / "second.pt", tmp_path / "ensemble.pt"
    scratch_checkpoint(first, 8, [0.0, 1.0])
    scratch_checkpoint(second, 8, [1.0, 0.0])
    output.write_bytes(b"keep this candidate")
    with pytest.raises(FileExistsError):
        ensemble_script.make_candidate([first, second], output, 224)
    assert output.read_bytes() == b"keep this candidate"
