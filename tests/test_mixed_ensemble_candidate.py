"""Release candidates must preserve their own single-network raw logits."""

import hashlib
import importlib.util
import io
from pathlib import Path

import pytest
import torch

from catsdogs.model import build_model


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/make_ensemble.py"
spec = importlib.util.spec_from_file_location("mixed_ensemble_candidate", SCRIPT)
ensemble_script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ensemble_script)


@pytest.fixture(autouse=True)
def small_cpu_threads():
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old_threads)


def own_checkpoint(path, architecture, width, seed, image_size=224):
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        config = {"name": architecture, "num_classes": 2, "width": width, "dropout": 0.15}
        model = build_model(**config).eval()
    checkpoint = {
        "model_config": config, "state_dict": model.state_dict(), "image_size": image_size,
        "class_names": ["cat", "dog"],
        "normalization": {"mean": [0.5] * 3, "std": [0.5] * 3},
        "resize_mode": "letterbox", "pretrained": False, "initialization": "random",
        "epoch": 9, "val_source": "ema",
        "train_config": {"manifest_sha256": "a" * 64, "pretrained": False, "initialization": "random"},
        # Source calibration must not leak into a new unvalidated ensemble.
        "temperature": 0.3, "threshold": 0.7, "tta_horizontal_flip": True,
    }
    torch.save(checkpoint, path)
    return checkpoint, model


@pytest.mark.parametrize("second_architecture", ["tiny_resnet18", "robust_resnet18"])
def test_candidate_roundtrip_matches_raw_weighted_network_logits(tmp_path, second_architecture):
    first, second, output = (tmp_path / name for name in ["robust.pt", "other.pt", "candidate.pt"])
    first_checkpoint, first_model = own_checkpoint(first, "robust_resnet18", 8, 701)
    second_checkpoint, second_model = own_checkpoint(second, second_architecture, 12, 702, image_size=256)
    source_bytes = [path.read_bytes() for path in [first, second]]
    report = ensemble_script.make_candidate([first, second], output, 96, [2, 3])
    candidate = torch.load(output, map_location="cpu", weights_only=True)
    restored = build_model(**candidate["model_config"]).eval()
    restored.load_state_dict(candidate["state_dict"], strict=True)
    generator = torch.Generator().manual_seed(703)
    with torch.inference_mode():
        for shape in [(2, 3, 64, 64), (1, 3, 65, 79)]:
            batch = torch.rand(shape, generator=generator) * 2 - 1
            expected = first_model(batch) * 0.4 + second_model(batch) * 0.6
            torch.testing.assert_close(restored(batch), expected, rtol=1e-6, atol=1e-7)
    for restored_member, original in zip(restored.members, [first_checkpoint, second_checkpoint], strict=True):
        for key, value in restored_member.state_dict().items():
            assert torch.equal(value, original["state_dict"][key])
    assert candidate["candidate_validation_pending"] is True
    assert candidate["selection_frozen_before_test"] is False
    assert (candidate["temperature"], candidate["threshold"], candidate["tta_horizontal_flip"]) == (1.0, 0.5, False)
    assert candidate["image_size"] == 96
    assert candidate["train_config"]["manifest_sha256"] == "a" * 64
    assert [record["image_size"] for record in report["provenance"]["members"]] == [224, 256]
    assert [record["sha256"] for record in report["provenance"]["members"]] == [
        hashlib.sha256(payload).hexdigest() for payload in source_bytes
    ]
    assert source_bytes == [path.read_bytes() for path in [first, second]]


@pytest.mark.parametrize("change", [
    "nested", "unknown_architecture", "manifest_mismatch", "manifest_missing", "manifest_invalid",
    "classes", "normalization", "resize_mode", "pretrained", "missing_pretrained",
])
def test_mixed_candidate_guards_reject_before_model_build(tmp_path, monkeypatch, change):
    first, second, output = (tmp_path / name for name in ["first.pt", "second.pt", "candidate.pt"])
    own_checkpoint(first, "robust_resnet18", 8, 704)
    checkpoint, _ = own_checkpoint(second, "tiny_resnet18", 8, 705)
    if change == "nested":
        checkpoint["model_config"] = {"name": "logit_ensemble", "members": [
            {"name": "tiny_resnet18", "width": 8}, {"name": "robust_resnet18", "width": 8}
        ]}
    elif change == "unknown_architecture":
        checkpoint["model_config"]["name"] = "external_model"
    elif change == "manifest_mismatch":
        checkpoint["train_config"]["manifest_sha256"] = "b" * 64
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
    elif change == "pretrained":
        checkpoint["pretrained"] = True
    else:
        checkpoint.pop("pretrained")
    torch.save(checkpoint, second)

    def no_build(**kwargs):
        pytest.fail("Incompatible sources must be rejected before constructing candidate networks")

    monkeypatch.setattr(ensemble_script, "build_model", no_build)
    with pytest.raises(ValueError):
        ensemble_script.make_candidate([first, second], output, 96)
    assert not output.exists()


def test_source_sha_and_model_use_the_same_single_immutable_snapshot(tmp_path, monkeypatch):
    first, second, output = (tmp_path / name for name in ["first.pt", "second.pt", "candidate.pt"])
    _, first_model = own_checkpoint(first, "robust_resnet18", 8, 706)
    _, second_model = own_checkpoint(second, "tiny_resnet18", 8, 707)
    original_bytes = {path.resolve(): path.read_bytes() for path in [first, second]}
    counts = {path: 0 for path in original_bytes}
    original_read = Path.read_bytes
    original_load = torch.load

    def read_snapshot(path):
        if path in counts:
            counts[path] += 1
            # Simulate a competing producer replacing the source immediately after the read.
            path.write_bytes(b"replacement should never be loaded or hashed")
            return original_bytes[path]
        return original_read(path)

    def require_immutable_load(source, *args, **kwargs):
        assert isinstance(source, io.BytesIO)
        return original_load(source, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "read_bytes", read_snapshot)
        patch.setattr(torch, "load", require_immutable_load)
        report = ensemble_script.make_candidate([first, second], output, 96, [1, 3])
    assert list(counts.values()) == [1, 1]
    candidate = torch.load(output, map_location="cpu", weights_only=True)
    restored = build_model(**candidate["model_config"]).eval()
    restored.load_state_dict(candidate["state_dict"], strict=True)
    with torch.inference_mode():
        batch = torch.zeros(1, 3, 64, 64)
        torch.testing.assert_close(restored(batch), first_model(batch) * 0.25 + second_model(batch) * 0.75)
    assert [record["sha256"] for record in report["provenance"]["members"]] == [
        hashlib.sha256(original_bytes[path.resolve()]).hexdigest() for path in [first, second]
    ]
