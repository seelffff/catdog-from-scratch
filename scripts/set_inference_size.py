"""Create an untested resolution candidate without changing any learned tensor."""

import argparse
import hashlib
import io
import math
import re
from pathlib import Path

import torch


NORMALIZATION = {"mean": [0.5] * 3, "std": [0.5] * 3}
SHA_PATTERN = re.compile(r"[0-9a-f]{64}")


def _image_size(value: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and 64 <= value <= 1024


def _validate_model_config(config: dict, *, member: bool = False) -> None:
    """Validate the configurations supported by build_model without initializing weights."""
    if not isinstance(config, dict):
        raise ValueError("Missing model configuration")
    name = config.get("name", "tiny_resnet18")
    if not isinstance(name, str) or name not in {"tiny_resnet18", "robust_resnet18", "logit_ensemble"}:
        raise ValueError("Unsupported scratch architecture")
    num_classes = config.get("num_classes", 2)
    if not isinstance(num_classes, int) or isinstance(num_classes, bool) or num_classes != 2:
        raise ValueError("Expected exactly two classes")
    width, dropout = config.get("width", 32), config.get("dropout", 0.15)
    if not isinstance(width, int) or isinstance(width, bool) or width < 8:
        raise ValueError("Expected integer model width >=8")
    if (not isinstance(dropout, (int, float)) or isinstance(dropout, bool)
            or not math.isfinite(dropout) or not 0 <= dropout < 1):
        raise ValueError("Invalid dropout")
    allowed = {"name", "num_classes", "width", "dropout", "members", "mixture_weights"}
    if set(config) - allowed:
        raise ValueError("Unsupported model configuration fields")
    if name != "logit_ensemble":
        if config.get("members") is not None or config.get("mixture_weights") is not None:
            raise ValueError("Ensemble settings require an ensemble architecture")
        return
    if member:
        raise ValueError("Nested ensembles are not supported")
    members = config.get("members")
    if not isinstance(members, (list, tuple)) or not 2 <= len(members) <= 3:
        raise ValueError("Expected two or three single-network ensemble members")
    for child in members:
        _validate_model_config(child, member=True)
    weights = config.get("mixture_weights")
    if weights is None:
        return
    if (not isinstance(weights, (list, tuple)) or len(weights) != len(members)
            or any(not isinstance(value, (int, float)) or isinstance(value, bool)
                   or not math.isfinite(value) or value <= 0 for value in weights)):
        raise ValueError("Every member requires a finite positive mixture weight")
    largest = max(weights)
    scaled = [value / largest for value in weights]
    normalized = torch.tensor([value / sum(scaled) for value in scaled], dtype=torch.float32)
    if not torch.all(normalized > 0):
        raise ValueError("Mixture weights differ too much to represent in float32")


def _validate_checkpoint(checkpoint: dict) -> str:
    if not isinstance(checkpoint, dict):
        raise ValueError("Expected a checkpoint dictionary")
    train_config = checkpoint.get("train_config")
    if (checkpoint.get("pretrained") is not False or not isinstance(train_config, dict)
            or train_config.get("pretrained") is not False):
        raise ValueError("Only our own scratch checkpoints with confirmed pretrained=False are supported")
    manifest_sha256 = train_config.get("manifest_sha256")
    if not isinstance(manifest_sha256, str) or not SHA_PATTERN.fullmatch(manifest_sha256):
        raise ValueError("Checkpoint must contain the training manifest SHA-256")
    _validate_model_config(checkpoint.get("model_config"))
    if (checkpoint.get("class_names") != ["cat", "dog"]
            or checkpoint.get("resize_mode") != "letterbox"
            or checkpoint.get("normalization") != NORMALIZATION):
        raise ValueError("Expected cat/dog classes, letterbox and the training normalization")
    if not _image_size(checkpoint.get("image_size")):
        raise ValueError("Source image size must be an integer between 64 and 1024")
    initialization = checkpoint.get("initialization")
    if initialization is None:
        initialization = train_config.get("initialization")
    if initialization is not None:
        if initialization not in {"random", "continuation_of_own_scratch_training", "own_model_ensemble"}:
            raise ValueError("Unsupported initialization provenance")
        if train_config.get("initialization", initialization) != initialization:
            raise ValueError("Conflicting initialization provenance")
        if (initialization == "continuation_of_own_scratch_training"
                and not SHA_PATTERN.fullmatch(str(train_config.get("initialization_checkpoint_sha256", "")))):
            raise ValueError("Own continuation must include its source checkpoint SHA-256")
    state = checkpoint.get("state_dict")
    if (not isinstance(state, dict) or not state
            or any(not isinstance(key, str) or not isinstance(value, torch.Tensor) for key, value in state.items())):
        raise ValueError("Expected a nonempty tensor state_dict")
    return manifest_sha256


def set_inference_size(checkpoint_path: Path, image_size: int, output: Path,
                       manifest_path: Path | None = None) -> dict:
    """Write a pending candidate; tensor values and source training metadata are untouched."""
    if not _image_size(image_size):
        raise ValueError("Image size must be an integer between 64 and 1024")
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    # Check before resolving: even a dangling destination symlink is an existing output.
    output = Path(output).expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError("Use a new output path; existing files and symlinks are not replaced")
    output = output.resolve()
    if output == checkpoint_path:
        raise ValueError("Source checkpoint and output must be different")
    payload = checkpoint_path.read_bytes()
    source_sha256 = hashlib.sha256(payload).hexdigest()
    checkpoint = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    manifest_sha256 = _validate_checkpoint(checkpoint)
    if manifest_path is not None:
        actual_manifest_sha256 = hashlib.sha256(Path(manifest_path).expanduser().resolve().read_bytes()).hexdigest()
        if actual_manifest_sha256 != manifest_sha256:
            raise ValueError("Checkpoint and requested manifest SHA-256 differ")
    original_size = checkpoint["image_size"]
    checkpoint["image_size"] = image_size
    checkpoint["candidate_validation_pending"] = True
    checkpoint["selection_frozen_before_test"] = False
    checkpoint["selection_frozen"] = False
    checkpoint["temperature"] = 1.0
    checkpoint["threshold"] = 0.5
    checkpoint["tta_horizontal_flip"] = False
    checkpoint["provenance"] = {
        "method": "inference_resolution_candidate", "source_checkpoint": str(checkpoint_path),
        "source_sha256": source_sha256, "source_checkpoint_sha256": source_sha256,
        "manifest_sha256": manifest_sha256, "original_image_size": original_size,
        "candidate_image_size": image_size, "source_provenance": checkpoint.get("provenance"),
        "learned_tensors_unchanged": True, "validation_required": True,
        "data_access": "No train, validation or test images were read",
    }
    for stale_field in ("validation", "calibration", "selection"):
        checkpoint.pop(stale_field, None)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        try:
            torch.save(checkpoint, handle)
        except Exception:
            output.unlink(missing_ok=True)
            raise
    return {"checkpoint": str(output), "image_size": image_size,
            "validation_required": True, "provenance": checkpoint["provenance"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image-size", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, help="Optionally verify the intended split before writing")
    args = parser.parse_args()
    result = set_inference_size(args.checkpoint, args.image_size, args.output, args.manifest)
    print(result["checkpoint"])


if __name__ == "__main__":
    main()
