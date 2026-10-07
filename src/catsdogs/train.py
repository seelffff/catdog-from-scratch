"""Reproducible training on audited train/validation splits, with no pretrained weights."""

import argparse
import copy
import csv
import hashlib
import io
import json
import math
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from .data import read_manifest
from .model import build_model
from .paths import PROJECT_ROOT
from .preprocessing import letterbox_image, preprocess_image

NORMALIZATION = {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]}


def training_augmentation(image_size: int, recipe: str = "baseline"):
    """Select an explicit recipe without changing the original training default."""
    if recipe == "weak":
        return transforms.Compose([
            transforms.RandomApply([
                transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0), ratio=(0.9, 1.1),
                                             interpolation=InterpolationMode.BILINEAR),
            ], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.01),
            transforms.ToTensor(),
            transforms.Normalize(NORMALIZATION["mean"], NORMALIZATION["std"]),
        ])
    if recipe == "robust":
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.6, 1.0), ratio=(0.85, 1.15),
                                         interpolation=InterpolationMode.BILINEAR),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=5,
                                   interpolation=InterpolationMode.BILINEAR,
                                   fill=(128, 128, 128)),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.2, hue=0.015),
            transforms.RandomGrayscale(p=0.12),
            transforms.ToTensor(),
            transforms.Normalize(NORMALIZATION["mean"], NORMALIZATION["std"]),
            transforms.RandomErasing(p=0.12, scale=(0.01, 0.08), value=0),
        ])
    if recipe != "baseline":
        raise ValueError(f"Unknown augmentation recipe: {recipe}")
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0), ratio=(0.9, 1.1),
                                     interpolation=InterpolationMode.BILINEAR),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=10, translate=(0.04, 0.04),
                                interpolation=InterpolationMode.BILINEAR, fill=(128, 128, 128)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.025),
        transforms.ToTensor(),
        transforms.Normalize(NORMALIZATION["mean"], NORMALIZATION["std"]),
        transforms.RandomErasing(p=0.1, scale=(0.01, 0.07), value=0),
    ])


class ImageDataset(Dataset):
    def __init__(self, manifest: Path, split: str, image_size: int, augment: bool = False,
                 augmentation_recipe: str = "baseline"):
        self.rows = read_manifest(manifest, split=split)
        self.root = manifest.resolve().parent.parent
        self.image_size = image_size
        self.augmentation = training_augmentation(image_size, augmentation_recipe) if augment else None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(self.root / row["relative_path"]) as image:
            if self.augmentation is None:
                tensor = preprocess_image(image, self.image_size, NORMALIZATION)
            else:
                tensor = self.augmentation(letterbox_image(image, self.image_size))
        return tensor, int(row["label"])


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_sampling_weights(path: Path, train_rows: list[dict]):
    """Validate train-only priorities, discount duplicates, then balance classes.

    These weights affect sampling, not the loss. The CSV cannot introduce an
    image or select only a subset of the current audited training split.
    """
    paths = [row["relative_path"] for row in train_rows]
    if not train_rows or len(set(paths)) != len(paths):
        raise ValueError("Sampling requires a non-empty train split with unique paths")
    if any(row.get("split") != "train" for row in train_rows):
        raise ValueError("Sampling rows must belong exclusively to train")
    if any(row.get("label") not in (0, 1) or not row.get("group_id") for row in train_rows):
        raise ValueError("Sampling requires audited class labels and duplicate group IDs")
    group_classes = {}
    for row in train_rows:
        previous = group_classes.setdefault(row["group_id"], row["label"])
        if previous != row["label"]:
            raise ValueError("A duplicate training group cannot have conflicting class labels")
    if {row["label"] for row in train_rows} != {0, 1}:
        raise ValueError("Class-balanced sampling requires both training classes")

    source_bytes = Path(path).read_bytes()
    reader = csv.DictReader(io.StringIO(source_bytes.decode("utf-8-sig")))
    fields = reader.fieldnames
    if (fields is None or not {"relative_path", "weight"}.issubset(fields)
            or len(fields) != len(set(fields))):
        raise ValueError("Sampling CSV requires unique relative_path and weight columns")
    allowed_paths = set(paths)
    raw_weights = {}
    for number, row in enumerate(reader, start=2):
        relative_path = row.get("relative_path")
        if relative_path not in allowed_paths:
            raise ValueError(f"Sampling CSV line {number} contains a path outside the current train split")
        if relative_path in raw_weights:
            raise ValueError(f"Sampling CSV repeats training path: {relative_path}")
        try:
            weight = float(row["weight"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"Sampling CSV line {number} has an invalid weight") from error
        if not math.isfinite(weight) or not 0 < weight <= 2:
            raise ValueError(f"Sampling CSV line {number}: weight must be finite and in (0, 2]")
        raw_weights[relative_path] = weight
    missing_paths = allowed_paths - raw_weights.keys()
    if missing_paths:
        raise ValueError(f"Sampling CSV must cover every training image; missing {len(missing_paths)} paths")

    group_sizes = Counter(row["group_id"] for row in train_rows)
    adjusted = [raw_weights[row["relative_path"]] / group_sizes[row["group_id"]]
                for row in train_rows]
    class_masses = [math.fsum(weight for weight, row in zip(adjusted, train_rows, strict=True)
                             if row["label"] == label) for label in (0, 1)]
    if any(not math.isfinite(mass) or mass <= 0 for mass in class_masses):
        raise ValueError("Sampling weights are too extreme to normalize safely")
    weights = torch.tensor([0.5 * weight / class_masses[row["label"]]
                            for weight, row in zip(adjusted, train_rows, strict=True)],
                           dtype=torch.float64)
    # Very small positive CSV values must remain representable after normalization.
    if not torch.isfinite(weights).all() or not torch.all(weights > 0):
        raise ValueError("Sampling weights are too extreme to normalize safely")
    policy = {
        "name": "WeightedRandomSampler",
        "replacement": True,
        "num_samples": len(train_rows),
        "input_weight_range": "(0, 2]",
        "duplicate_group_adjustment": "divide each input weight by its train group size",
        "normalization_order": "duplicate groups first, class masses second",
        "class_mass_target": {"cat": 0.5, "dog": 0.5},
        "normalized_class_mass": {
            class_name: math.fsum(float(weights[index]) for index, row in enumerate(train_rows)
                                 if row["label"] == label)
            for label, class_name in enumerate(("cat", "dog"))
        },
        "train_groups": len(group_sizes),
        "duplicate_groups_discounted": sum(size > 1 for size in group_sizes.values()),
        "input_weight_min": min(raw_weights.values()),
        "input_weight_max": max(raw_weights.values()),
    }
    return weights, {"sample_weights_source": str(Path(path).resolve()),
                     "sample_weights_source_sha256": hashlib.sha256(source_bytes).hexdigest(),
                     "sampler_policy": policy}


def json_write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def cpu_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def save_checkpoint(path, checkpoint):
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(checkpoint, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def evaluate(model, loader, device):
    model.eval()
    totals = torch.zeros(2, device=device)
    confusion = torch.zeros((2, 2), dtype=torch.int64, device=device)
    count = 0
    with torch.inference_mode():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            logits = model(images)
            loss = nn.functional.cross_entropy(logits, targets, reduction="sum")
            predictions = logits.argmax(1)
            totals[0] += loss
            totals[1] += (predictions == targets).float().sum()
            for actual in range(2):
                for predicted in range(2):
                    confusion[actual, predicted] += ((targets == actual) & (predictions == predicted)).sum()
            count += len(targets)
    totals = totals.cpu().tolist()
    matrix = confusion.cpu().numpy()
    recall = np.diag(matrix) / np.maximum(matrix.sum(axis=1), 1)
    return {"loss": totals[0] / count, "accuracy": totals[1] / count,
            "balanced_accuracy": float(recall.mean()), "confusion_matrix": matrix.tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "data/splits.csv")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "artifacts/runs/scratch160")
    parser.add_argument("--device", choices=["mps", "cpu"], default="mps")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--architecture", choices=["tiny_resnet18", "robust_resnet18"],
                        default="tiny_resnet18")
    parser.add_argument("--augmentation", choices=["baseline", "robust", "weak"], default="baseline")
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--label-smoothing", type=float, default=0.03)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--min-lr", type=float, default=0.00001)
    parser.add_argument("--weight-decay", type=float, default=0.0003)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--patience", type=int, default=18)
    parser.add_argument("--min-epochs", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixup", type=float, default=0.0)
    parser.add_argument("--ema-decay", type=float, default=0.0)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--sample-weights", type=Path,
                        help="CSV covering train only: relative_path,weight; finite weights in (0, 2]")
    parser.add_argument("--max-minutes", type=float, default=240)
    args = parser.parse_args()
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable in this process. Run with permission to use the Mac GPU.")
    if args.epochs < 1 or args.image_size < 64 or args.batch_size < 1 or args.workers < 0:
        raise ValueError("Invalid training dimensions or epoch count")
    if not 0 <= args.label_smoothing < 1:
        raise ValueError("Label smoothing must be in [0, 1)")
    if args.output.exists() and (args.output / "history.json").exists():
        raise FileExistsError("Choose a new output directory rather than overwrite an existing experiment")
    args.output.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    torch.set_num_threads(4)
    device = torch.device(args.device)
    manifest_hash = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    model_config = {"name": args.architecture, "num_classes": 2, "width": args.width, "dropout": args.dropout}
    model = build_model(**model_config)
    initialization = "random"
    initialization_sha256 = None
    if args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location="cpu", weights_only=True)
        if checkpoint.get("pretrained", False) or checkpoint["model_config"] != model_config:
            raise ValueError("Initialization must be from our compatible scratch-trained model")
        original_manifest_hash = checkpoint.get("train_config", {}).get("manifest_sha256")
        if original_manifest_hash != manifest_hash:
            raise ValueError("Continuation must use the original audited train/validation split")
        model.load_state_dict(checkpoint["state_dict"])
        initialization = "continuation_of_own_scratch_training"
        initialization_sha256 = hashlib.sha256(args.init_checkpoint.read_bytes()).hexdigest()
    model.to(device=device)
    if not 0 <= args.ema_decay < 1:
        raise ValueError("EMA decay must be in [0, 1)")
    ema_model = copy.deepcopy(model).eval() if args.ema_decay > 0 else None
    if ema_model is not None:
        for parameter in ema_model.parameters():
            parameter.requires_grad_(False)
    global_step = 0
    train_data = ImageDataset(args.manifest, "train", args.image_size, augment=True,
                              augmentation_recipe=args.augmentation)
    val_data = ImageDataset(args.manifest, "val", args.image_size)
    generator = torch.Generator().manual_seed(args.seed)
    common = {"batch_size": args.batch_size, "num_workers": args.workers,
              "persistent_workers": args.workers > 0, "pin_memory": False,
              "worker_init_fn": seed_worker}
    if args.workers > 0:
        common["prefetch_factor"] = 2
    sampling_config = {"sample_weights_source": None, "sample_weights_source_sha256": None,
                       "sampler_policy": {"name": "uniform_shuffle", "replacement": False,
                                          "num_samples": len(train_data)}}
    if args.sample_weights is not None:
        sample_weights, sampling_config = load_sampling_weights(args.sample_weights, train_data.rows)
        sampling_config["sampler_policy"]["seed"] = args.seed
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_data),
                                        replacement=True, generator=generator)
        train_loader = DataLoader(train_data, sampler=sampler, generator=generator, **common)
    else:
        train_loader = DataLoader(train_data, shuffle=True, generator=generator, **common)
    val_loader = DataLoader(val_data, shuffle=False, **common)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    config.update({"model_config": model_config, "initialization": initialization, "pretrained": False,
                   "initialization_checkpoint_sha256": initialization_sha256,
                   "parameter_count": parameter_count, "manifest_sha256": manifest_hash,
                   "train_images": len(train_data), "val_images": len(val_data), "torch_version": str(torch.__version__)})
    config.update(sampling_config)
    json_write(args.output / "config.json", config)
    print(json.dumps({"event": "start", **config}), flush=True)
    started = time.monotonic()
    history = []
    best_score = -1.0
    best_loss = math.inf
    best_epoch = 0
    stale_epochs = 0
    stop_reason = "epochs_completed"
    try:
        for epoch in range(1, args.epochs + 1):
            epoch_started = time.monotonic()
            if epoch <= args.warmup:
                lr = args.lr * epoch / max(args.warmup, 1)
            else:
                progress = (epoch - args.warmup - 1) / max(args.epochs - args.warmup - 1, 1)
                lr = args.min_lr + (args.lr - args.min_lr) * (1 + math.cos(math.pi * progress)) / 2
            for group in optimizer.param_groups:
                group["lr"] = lr
            model.train()
            totals = torch.zeros(2, device=device)
            count = 0
            last_progress = time.monotonic()
            for step, (images, targets) in enumerate(train_loader, start=1):
                images = images.to(device)
                targets = targets.to(device)
                optimizer.zero_grad(set_to_none=True)
                lam = 1.0
                other_targets = targets
                if args.mixup > 0 and random.random() < 0.5:
                    lam = float(np.random.beta(args.mixup, args.mixup))
                    indices = torch.randperm(len(targets), device=device)
                    images = lam * images + (1 - lam) * images[indices]
                    other_targets = targets[indices]
                logits = model(images)
                loss = lam * criterion(logits, targets) + (1 - lam) * criterion(logits, other_targets)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                global_step += 1
                if ema_model is not None:
                    decay = min(args.ema_decay, (1 + global_step) / (10 + global_step))
                    with torch.no_grad():
                        for average, parameter in zip(ema_model.parameters(), model.parameters(), strict=True):
                            average.lerp_(parameter, 1 - decay)
                        for average, buffer in zip(ema_model.buffers(), model.buffers(), strict=True):
                            average.copy_(buffer)
                predictions = logits.detach().argmax(1)
                totals[0] += loss.detach() * len(targets)
                totals[1] += (lam * (predictions == targets).float()
                              + (1 - lam) * (predictions == other_targets).float()).sum()
                count += len(targets)
                if time.monotonic() - last_progress >= 30:
                    print(json.dumps({"event": "batch", "epoch": epoch, "step": step,
                                      "steps": len(train_loader)}), flush=True)
                    last_progress = time.monotonic()
            values = totals.cpu().tolist()
            if not math.isfinite(values[0]):
                raise RuntimeError("Training loss became non-finite")
            raw_val = evaluate(model, val_loader, device)
            ema_val = evaluate(ema_model, val_loader, device) if ema_model is not None else None
            chosen_model = model
            val = raw_val
            val_source = "raw"
            if ema_val is not None and (ema_val["balanced_accuracy"], -ema_val["loss"]) > (raw_val["balanced_accuracy"], -raw_val["loss"]):
                val, chosen_model, val_source = ema_val, ema_model, "ema"
            row = {"epoch": epoch, "lr": lr, "train_loss": values[0] / count,
                   "train_accuracy": values[1] / count, "val_loss": val["loss"],
                   "val_accuracy": val["accuracy"], "val_balanced_accuracy": val["balanced_accuracy"],
                   "seconds": time.monotonic() - epoch_started,
                   "raw_val_accuracy": raw_val["accuracy"],
                   "ema_val_accuracy": ema_val["accuracy"] if ema_val is not None else None,
                   "val_source": val_source}
            history.append(row)
            score = val["balanced_accuracy"]
            improved = score > best_score + 1e-9 or (abs(score - best_score) < 1e-9 and val["loss"] < best_loss)
            if improved:
                best_score, best_loss, best_epoch = score, val["loss"], epoch
                stale_epochs = 0
            else:
                stale_epochs += 1
            checkpoint = {"model_config": model_config, "state_dict": cpu_state(chosen_model),
                          "image_size": args.image_size, "class_names": ["cat", "dog"],
                          "normalization": NORMALIZATION, "resize_mode": "letterbox",
                          "temperature": 1.0, "threshold": 0.5, "pretrained": False,
                          "initialization": initialization, "epoch": epoch,
                          "validation": val, "train_config": config, "val_source": val_source}
            save_checkpoint(args.output / "last.pt", checkpoint)
            if improved:
                save_checkpoint(args.output / "best.pt", checkpoint)
            json_write(args.output / "history.json", history)
            with (args.output / "history.csv").open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=list(row))
                writer.writeheader()
                writer.writerows(history)
            print(json.dumps({"event": "epoch", **row, "best_epoch": best_epoch,
                              "best_val_balanced_accuracy": best_score}), flush=True)
            if epoch >= args.min_epochs and stale_epochs >= args.patience:
                stop_reason = "early_stopping"
                break
            if time.monotonic() - started > args.max_minutes * 60:
                stop_reason = "time_budget"
                break
    except KeyboardInterrupt:
        stop_reason = "interrupted_after_last_completed_epoch"
    result = {"epochs_completed": len(history), "best_epoch": best_epoch,
              "best_val_balanced_accuracy": best_score, "best_val_loss": best_loss,
              "elapsed_seconds": time.monotonic() - started, "stop_reason": stop_reason,
              "best_checkpoint": str(args.output / "best.pt")}
    json_write(args.output / "result.json", result)
    print(json.dumps({"event": "finished", **result}), flush=True)


if __name__ == "__main__":
    main()
