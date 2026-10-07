"""Create a pending candidate by refreshing BatchNorm on clean TRAIN only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
from collections import defaultdict
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from .model import build_model
from .paths import PROJECT_ROOT
from .train import ImageDataset, NORMALIZATION, cpu_state, evaluate

CLASSES = ["cat", "dog"]
SHA_PATTERN = re.compile(r"[0-9a-f]{64}")


def _validated_source(payload: bytes, manifest_sha256: str) -> dict:
    checkpoint = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("Expected a scratch checkpoint dictionary")
    config = checkpoint.get("train_config", {})
    if not isinstance(config, dict):
        raise ValueError("Expected a confirmed training configuration")
    if checkpoint.get("pretrained") is not False or config.get("pretrained") is not False:
        raise ValueError("BatchNorm recalibration requires our scratch checkpoint with pretrained=False")
    initialization = checkpoint.get("initialization")
    if initialization not in {"random", "continuation_of_own_scratch_training"}:
        raise ValueError("Expected confirmed random or own scratch continuation initialization")
    if config.get("initialization") != initialization:
        raise ValueError("Initialization conflicts with the source training provenance")
    if initialization == "continuation_of_own_scratch_training" and not SHA_PATTERN.fullmatch(
        str(config.get("initialization_checkpoint_sha256", ""))
    ):
        raise ValueError("Own continuation requires its initialization checkpoint SHA-256")
    original_hash = config.get("manifest_sha256")
    if not isinstance(original_hash, str) or not SHA_PATTERN.fullmatch(original_hash):
        raise ValueError("A confirmed training manifest SHA-256 is mandatory")
    if original_hash != manifest_sha256:
        raise ValueError("Manifest SHA-256 differs from the original audited training split")
    if checkpoint.get("class_names") != CLASSES:
        raise ValueError("Expected class_names=['cat', 'dog']")
    if checkpoint.get("normalization") != NORMALIZATION:
        raise ValueError("Checkpoint normalization must match the clean training preprocessing")
    if checkpoint.get("resize_mode") != "letterbox":
        raise ValueError("Checkpoint resize_mode must be letterbox")
    model_config = checkpoint.get("model_config")
    if not isinstance(model_config, dict) or model_config.get("name") not in {"tiny_resnet18", "robust_resnet18"}:
        raise ValueError("Recalibration supports a single own scratch CNN, not an ensemble")
    size, epoch = checkpoint.get("image_size"), checkpoint.get("epoch")
    if type(size) is not int or not 64 <= size <= 1024:
        raise ValueError("Expected an input size between 64 and 1024")
    if type(epoch) is not int or epoch < 1:
        raise ValueError("Expected the number of a completed training epoch")
    if not isinstance(checkpoint.get("state_dict"), dict):
        raise ValueError("Checkpoint must contain a state_dict")
    return checkpoint


def _manifest_snapshot(payload: bytes, root: Path) -> dict[str, list[dict]]:
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8")))
    required = {"relative_path", "label", "class_name", "split", "group_id", "width", "height"}
    if not reader.fieldnames or not required.issubset(reader.fieldnames):
        raise ValueError("Expected the audited manifest columns")
    if len(reader.fieldnames) != len(set(reader.fieldnames)):
        raise ValueError("Manifest columns must not repeat")
    selected = {split: [] for split in ("train", "val", "test")}
    group_splits, group_labels, hash_splits = defaultdict(set), defaultdict(set), defaultdict(set)
    seen_paths = set()
    for row in reader:
        if any(not isinstance(value, str) for value in row.values()):
            raise ValueError("Malformed manifest row")
        split, group = row["split"], row["group_id"]
        if split not in selected or not group:
            raise ValueError("Every manifest row needs a known split and group_id")
        relative = Path(row["relative_path"])
        resolved = (root / relative).resolve()
        if not row["relative_path"] or relative.is_absolute() or not resolved.is_relative_to(root):
            raise ValueError("Image paths must remain inside the project")
        if resolved in seen_paths:
            raise ValueError("A train/heldout image path cannot repeat in the manifest")
        seen_paths.add(resolved)
        for field in ("label", "width", "height"):
            row[field] = int(row[field])
        if row["label"] not in (0, 1) or row["class_name"] != CLASSES[row["label"]]:
            raise ValueError("Manifest class labels must match cat/dog")
        if row["width"] < 1 or row["height"] < 1:
            raise ValueError("Manifest image dimensions must be positive")
        group_splits[group].add(split)
        group_labels[group].add(row["label"])
        if row.get("sha256"):
            hash_splits[row["sha256"]].add(split)
        selected[split].append(row)
    if any("train" in parts and len(parts) > 1 for parts in group_splits.values()):
        raise ValueError("A train duplicate group cannot intersect val/test")
    if any("train" in parts and len(parts) > 1 for parts in hash_splits.values()):
        raise ValueError("An exact train image hash cannot intersect val/test")
    if any(len(labels) != 1 for labels in group_labels.values()):
        raise ValueError("Duplicate groups cannot contain conflicting class labels")
    if not selected["train"]:
        raise ValueError("A non-empty train split is required")
    return selected


def _bitwise_equal(first: torch.Tensor, second: torch.Tensor) -> bool:
    if first.shape != second.shape or first.dtype != second.dtype:
        return False
    first_bytes = first.detach().cpu().reshape(-1).contiguous().view(torch.uint8)
    second_bytes = second.detach().cpu().reshape(-1).contiguous().view(torch.uint8)
    return torch.equal(first_bytes, second_bytes)


def _verify_changes(before: dict, after: dict, allowed: set[str]) -> list[str]:
    if before.keys() != after.keys():
        raise RuntimeError("The checkpoint state structure must remain unchanged")
    changed = []
    for name, previous in before.items():
        current = after[name]
        if not isinstance(previous, torch.Tensor) or not isinstance(current, torch.Tensor):
            raise ValueError("Every state_dict value must be a tensor")
        if previous.shape != current.shape or previous.dtype != current.dtype:
            raise RuntimeError(f"Checkpoint tensor shape and dtype must remain unchanged: {name}")
        if not _bitwise_equal(previous, current):
            if name not in allowed:
                raise RuntimeError(f"Only BatchNorm running buffers may change; changed {name}")
            changed.append(name)
        if current.is_floating_point() and not torch.isfinite(current).all():
            raise ValueError(f"Non-finite checkpoint tensor: {name}")
    return sorted(changed)


def recalibrate_checkpoint(checkpoint_path: Path, output_path: Path, manifest_path: Path,
                           device: str = "cpu", batch_size: int = 64, seed: int = 0,
                           validate: bool = False) -> dict:
    """Refresh only BN buffers; return provenance without selecting the candidate.

    The immutable checkpoint snapshot supplies both loading and source SHA.
    Validation is optional and always in eval mode. Test pixels are never read.
    Cumulative averaging weights batches equally; it is not an exact pooled
    population-variance estimate when the last batch is shorter.
    """
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().absolute()
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError("Use a new output path to preserve every existing checkpoint")
    output_path = output_path.resolve()
    manifest_path = Path(manifest_path).expanduser().resolve()
    if output_path == checkpoint_path:
        raise FileExistsError("Use a new output path to preserve the original checkpoint")
    if device not in {"cpu", "mps"} or batch_size < 1 or type(seed) is not int or seed < 0:
        raise ValueError("Expected cpu/mps, positive batch size and a non-negative integer seed")
    source_bytes = checkpoint_path.read_bytes()
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    manifest_bytes = manifest_path.read_bytes()
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    checkpoint = _validated_source(source_bytes, manifest_sha256)
    snapshot = _manifest_snapshot(manifest_bytes, manifest_path.parent.parent)
    if validate and not snapshot["val"]:
        raise ValueError("Optional validation requires a non-empty val split")
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("GPU access is needed for MPS")
    if device == "cpu":
        torch.set_num_threads(2)
    selected_device = torch.device(device)
    model = build_model(**checkpoint["model_config"]).cpu().eval()
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    _verify_changes(checkpoint["state_dict"], cpu_state(model), set())
    batchnorms = [(name, module) for name, module in model.named_modules()
                  if isinstance(module, nn.modules.batchnorm._BatchNorm) and module.track_running_stats]
    if not batchnorms:
        raise ValueError("The scratch model must contain tracked BatchNorm statistics")
    allowed_buffers = {f"{name}.{field}" if name else field for name, _ in batchnorms
                       for field in ("running_mean", "running_var", "num_batches_tracked")}
    train_data = ImageDataset(manifest_path, "train", checkpoint["image_size"], augment=False)
    if train_data.rows != snapshot["train"] or train_data.augmentation is not None:
        raise ValueError("Manifest changed after its snapshot, or training augmentation is active")
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0,
                              generator=torch.Generator().manual_seed(seed))
    val_loader = None
    if validate:
        val_data = ImageDataset(manifest_path, "val", checkpoint["image_size"], augment=False)
        if val_data.rows != snapshot["val"] or val_data.augmentation is not None:
            raise ValueError("Manifest changed after snapshot, or validation augmentation is active")
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=0)
    model.to(selected_device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    before_validation = evaluate(model, val_loader, selected_device) if val_loader is not None else None
    _verify_changes(checkpoint["state_dict"], cpu_state(model), set())
    momenta = {module: module.momentum for _, module in batchnorms}
    try:
        for _, module in batchnorms:
            module.reset_running_stats()
            module.momentum = None
            module.train()
        with torch.inference_mode():
            for images, _ in train_loader:
                logits = model(images.to(selected_device))
                if logits.shape != (len(images), 2) or not torch.isfinite(logits).all():
                    raise RuntimeError("Invalid logits during train-only BN refresh")
    finally:
        for module, momentum in momenta.items():
            module.momentum = momentum
        model.eval()
    refreshed_state = cpu_state(model)
    changed_buffers = _verify_changes(checkpoint["state_dict"], refreshed_state, allowed_buffers)
    after_validation = evaluate(model, val_loader, selected_device) if val_loader is not None else None
    _verify_changes(refreshed_state, cpu_state(model), set())
    details = {
        "source_checkpoint": str(checkpoint_path), "source_checkpoint_sha256": source_sha256,
        "source_epoch": checkpoint["epoch"], "source_validation_weights": checkpoint.get("val_source", "raw"),
        "manifest_sha256": manifest_sha256, "data": "clean_train_only_without_augmentation",
        "images": len(train_data), "batch_size": batch_size, "batches": len(train_loader),
        "shuffle": True, "seed": seed, "method": "reset BN buffers, cumulative equal-batch averaging",
        "learned_parameters_bitwise_unchanged": True, "changed_bn_buffers": changed_buffers,
        "dropout_and_other_modules": "eval", "gradients": False, "test_images_read": False,
        "validation_images_read": validate, "before_validation": before_validation,
        "after_validation": after_validation,
    }
    source_provenance = checkpoint.get("provenance")
    checkpoint["state_dict"] = refreshed_state
    checkpoint["batchnorm_recalibration"] = details
    checkpoint["provenance"] = {**details, "method": "train_only_batchnorm_recalibration",
                                "statistics_method": details["method"],
                                "source_training_provenance": source_provenance}
    checkpoint["val_source"] = f"{checkpoint.get('val_source', 'raw')}_bn_recalibrated"
    checkpoint["temperature"], checkpoint["threshold"] = 1.0, 0.5
    checkpoint["tta_horizontal_flip"] = False
    checkpoint["selection_frozen_before_test"] = False
    checkpoint["selection_frozen"] = False
    checkpoint["candidate_validation_pending"] = True
    checkpoint.pop("validation", None)
    checkpoint.pop("calibration", None)
    checkpoint.pop("selection", None)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("xb") as handle:
        try:
            torch.save(checkpoint, handle)
        except BaseException:
            output_path.unlink(missing_ok=True)
            raise
    return details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "data/splits_v2.csv")
    parser.add_argument("--device", default="cpu", choices=["mps", "cpu"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--validate", action="store_true", help="Record before/after val diagnostics; never select or test")
    args = parser.parse_args()
    report = recalibrate_checkpoint(args.checkpoint, args.output, args.manifest,
                                   args.device, args.batch_size, args.seed, args.validate)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
