"""Inspect difficult TRAIN images without training, sampling, or changing labels.

Example: python scripts/inspect_hard_train.py --checkpoint RUN/best.pt
         --manifest data/splits_v2.csv --output artifacts/v2/diagnostics/epoch020
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import sys
import tempfile
import os

import numpy as np
from PIL import Image, ImageOps
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from catsdogs.model import build_model
from catsdogs.train import ImageDataset, NORMALIZATION

CLASSES = ["cat", "dog"]
SCORE_COLUMNS = (
    "relative_path", "group_id", "label", "class_name", "epoch",
    "predicted_label", "predicted_class", "correct", "cat_probability",
    "dog_probability", "confidence", "p_true", "loss", "score",
)
GROUP_COLUMNS = (
    "group_id", "label", "class_name", "images", "errors", "mean_loss",
    "max_loss", "mean_p_true", "min_p_true", "relative_paths",
)


def validated_checkpoint(payload: bytes, manifest_sha256: str) -> dict:
    checkpoint = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    config = checkpoint.get("train_config", {})
    if checkpoint.get("pretrained") is not False or config.get("pretrained") is not False:
        raise ValueError("Требуется собственный scratch-checkpoint с pretrained=False")
    initialization = checkpoint.get("initialization")
    if initialization not in {"random", "continuation_of_own_scratch_training"}:
        raise ValueError("Не подтверждено происхождение собственного scratch-обучения")
    if config.get("initialization") != initialization:
        raise ValueError("Происхождение инициализации противоречит train_config")
    if initialization == "continuation_of_own_scratch_training" and not re.fullmatch(
        r"[0-9a-f]{64}", str(config.get("initialization_checkpoint_sha256", ""))
    ):
        raise ValueError("Продолжение обучения должно содержать SHA-256 исходного checkpoint")
    original_manifest = config.get("manifest_sha256")
    if not isinstance(original_manifest, str) or not re.fullmatch(r"[0-9a-f]{64}", original_manifest):
        raise ValueError("Checkpoint не содержит корректный manifest SHA-256")
    if original_manifest != manifest_sha256:
        raise ValueError("Manifest hash mismatch: нельзя диагностировать чужое разделение")
    if checkpoint.get("model_config", {}).get("name") not in {"tiny_resnet18", "robust_resnet18"}:
        raise ValueError("Диагностика предназначена для одной собственной обучаемой CNN")
    if checkpoint.get("class_names") != CLASSES or checkpoint.get("resize_mode") != "letterbox":
        raise ValueError("Ожидаются классы cat/dog и предобработка letterbox")
    if checkpoint.get("normalization") != NORMALIZATION:
        raise ValueError("Нормализация checkpoint должна совпадать с clean ImageDataset")
    if not isinstance(checkpoint.get("epoch"), int) or checkpoint["epoch"] < 1:
        raise ValueError("Checkpoint должен содержать номер завершённой эпохи обучения")
    if not isinstance(checkpoint.get("image_size"), int) or not 64 <= checkpoint["image_size"] <= 1024:
        raise ValueError("Некорректный размер входа checkpoint")
    return checkpoint


def train_rows_from_snapshot(payload: bytes, root: Path) -> tuple[list[dict], Counter]:
    all_rows = list(csv.DictReader(io.StringIO(payload.decode("utf-8"))))
    rows, counts = [], Counter()
    group_splits = defaultdict(set)
    path_splits = defaultdict(set)
    for row in all_rows:
        split = row.get("split")
        if split not in {"train", "val", "test"}:
            raise ValueError("Manifest содержит неизвестный split")
        counts[split] += 1
        group = row.get("group_id")
        if not group:
            raise ValueError("Manifest должен содержать непустые group_id")
        relative = Path(row["relative_path"])
        resolved = (root / relative).resolve()
        if relative.is_absolute() or not resolved.is_relative_to(root):
            raise ValueError("Путь фотографии должен находиться внутри проекта")
        group_splits[group].add(split)
        path_splits[str(resolved)].add(split)
        if split == "train":
            row = dict(row)
            for field in ("label", "width", "height"):
                row[field] = int(row[field])
            if row["label"] not in (0, 1) or row.get("class_name") != CLASSES[row["label"]]:
                raise ValueError("Некорректная train-метка")
            rows.append(row)
    if any(len(splits) > 1 and "train" in splits for splits in group_splits.values()):
        raise ValueError("Train-группа пересекает val/test")
    if any(len(splits) > 1 and "train" in splits for splits in path_splits.values()):
        raise ValueError("Train-путь пересекает val/test")
    if not rows or len({row["relative_path"] for row in rows}) != len(rows):
        raise ValueError("Ожидается непустой train без повторяющихся путей")
    labels = defaultdict(set)
    for row in rows:
        labels[row["group_id"]].add(row["label"])
    if any(len(values) != 1 for values in labels.values()):
        raise ValueError("Train-группа содержит противоречивые метки")
    return rows, counts


def aggregate_groups(scores: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in scores:
        grouped[row["group_id"]].append(row)
    groups = []
    for group_id, members in grouped.items():
        groups.append({
            "group_id": group_id, "label": members[0]["label"], "class_name": members[0]["class_name"],
            "images": len(members), "errors": sum(not row["correct"] for row in members),
            "mean_loss": float(np.mean([row["loss"] for row in members])),
            "max_loss": max(row["loss"] for row in members),
            "mean_p_true": float(np.mean([row["p_true"] for row in members])),
            "min_p_true": min(row["p_true"] for row in members),
            "relative_paths": json.dumps([row["relative_path"] for row in members], ensure_ascii=False),
        })
    return sorted(groups, key=lambda row: (-row["mean_loss"], row["group_id"]))


def save_csv(path: Path, rows: list[dict], columns: tuple[str, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def save_gallery(path: Path, scores: list[dict], root: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="hard-train-mpl-") as cache:
        os.environ.setdefault("MPLCONFIGDIR", cache)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        selected = sorted(scores, key=lambda row: (-row["loss"], row["relative_path"]))[:40]
        figure, axes = plt.subplots(math.ceil(len(selected) / 4), 4,
                                   figsize=(14, 3.7 * math.ceil(len(selected) / 4)), squeeze=False)
        for axis in axes.ravel():
            axis.axis("off")
        for axis, row in zip(axes.ravel(), selected):
            with Image.open(root / row["relative_path"]) as image:
                axis.imshow(ImageOps.exif_transpose(image).convert("RGB"))
            axis.set_title(f"{Path(row['relative_path']).name}\nTrain label: {row['class_name']}; guess: {row['predicted_class']}\n"
                           f"CE={row['loss']:.3f}; P(true)={row['p_true']:.1%}; confidence={row['confidence']:.1%}", fontsize=9)
        figure.suptitle("TRAIN only — difficulty review, not a test result", fontsize=13)
        figure.tight_layout(rect=(0, 0, 1, 0.98))
        figure.savefig(path, dpi=130)
        plt.close(figure)


def inspect_train(checkpoint_path: Path, manifest_path: Path, output: Path,
                  device: str = "cpu", batch_size: int = 32) -> dict:
    checkpoint_path, manifest_path, output = map(lambda path: Path(path).expanduser().resolve(),
                                                (checkpoint_path, manifest_path, output))
    if output.exists():
        raise FileExistsError("Укажите новый output: существующая диагностика не перезаписывается")
    if batch_size < 1 or device not in {"cpu", "mps"}:
        raise ValueError("Ожидаются batch_size > 0 и device cpu/mps")
    manifest_bytes = manifest_path.read_bytes()
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    # Exactly one read: loading and provenance hashing use the same immutable bytes.
    checkpoint_bytes = checkpoint_path.read_bytes()
    checkpoint_sha256 = hashlib.sha256(checkpoint_bytes).hexdigest()
    checkpoint = validated_checkpoint(checkpoint_bytes, manifest_sha256)
    root = manifest_path.parent.parent
    expected_rows, split_counts = train_rows_from_snapshot(manifest_bytes, root)
    dataset = ImageDataset(manifest_path, "train", checkpoint["image_size"], augment=False)
    if dataset.rows != expected_rows:
        raise ValueError("Manifest изменился после чтения snapshot; изображения не открывались")
    if dataset.augmentation is not None:
        raise RuntimeError("Для диагностики запрещена аугментация")
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS недоступен этому процессу")
    if device == "cpu":
        torch.set_num_threads(1)
    model = build_model(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    scores, offset = [], 0
    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            if logits.shape != (len(labels), 2) or not torch.isfinite(logits).all().item():
                raise RuntimeError("Модель вернула некорректные или неконечные логиты")
            losses = torch.nn.functional.cross_entropy(logits, labels, reduction="none", label_smoothing=0.0)
            probabilities = torch.softmax(logits, dim=1)
            guesses = logits.argmax(dim=1)
            if not torch.isfinite(losses).all().item():
                raise RuntimeError("Получен неконечный loss")
            losses, probabilities, guesses = losses.cpu().tolist(), probabilities.cpu().tolist(), guesses.cpu().tolist()
            for index, (loss, probability, guess) in enumerate(zip(losses, probabilities, guesses, strict=True)):
                row = expected_rows[offset + index]
                label = row["label"]
                scores.append({"relative_path": row["relative_path"], "group_id": row["group_id"],
                               "label": label, "class_name": CLASSES[label], "epoch": checkpoint["epoch"],
                               "predicted_label": guess, "predicted_class": CLASSES[guess], "correct": guess == label,
                               "cat_probability": probability[0], "dog_probability": probability[1],
                               "confidence": max(probability), "p_true": probability[label], "loss": loss, "score": loss})
            offset += len(labels)
    if offset != len(expected_rows):
        raise RuntimeError("Не все train-строки получили score")
    groups = aggregate_groups(scores)
    losses = np.asarray([row["loss"] for row in scores])
    summary = {
        "checkpoint": {"path": str(checkpoint_path), "sha256": checkpoint_sha256, "epoch": checkpoint["epoch"],
                       "model_config": checkpoint["model_config"], "initialization": checkpoint["initialization"],
                       "pretrained": False, "val_source": checkpoint.get("val_source", "raw")},
        "manifest": {"path": str(manifest_path), "sha256": manifest_sha256,
                     "image_root": str(root), "split_counts_metadata_only": dict(split_counts)},
        "scope": {"split": "train", "augmentation": False, "mixup": False, "label_smoothing": 0.0,
                  "temperature": 1.0, "tta": False, "criterion": "per-image cross-entropy on original class labels",
                  "score": "loss; review aid only, not an automatic noise label or sampling weight",
                  "evaluation": "model.eval and torch.inference_mode; no weights or labels changed",
                  "heldout_images_read": False},
        "images": len(scores), "groups": len(groups),
        "correct": sum(row["correct"] for row in scores), "errors": sum(not row["correct"] for row in scores),
        "high_confidence_errors_ge_0_9": sum(not row["correct"] and row["confidence"] >= 0.9 for row in scores),
        "per_class": {name: {"images": sum(row["class_name"] == name for row in scores),
                            "errors": sum(row["class_name"] == name and not row["correct"] for row in scores)} for name in CLASSES},
        "loss": {"mean": float(losses.mean()), "median": float(np.median(losses)),
                 "p90": float(np.percentile(losses, 90)), "max": float(losses.max())},
        "top_groups": groups[:40], "files": {"scores": "train_difficulty.csv", "groups": "train_groups.csv",
                                                 "gallery": "hard_train_gallery.png"},
        "review_note": "Hard images can be rare valid examples or label noise; review before any bounded upweighting.",
    }
    output.mkdir(parents=True, exist_ok=False)
    save_csv(output / "train_difficulty.csv", scores, SCORE_COLUMNS)
    save_csv(output / "train_groups.csv", groups, GROUP_COLUMNS)
    save_gallery(output / "hard_train_gallery.png", scores, root)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                                         encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    try:
        summary = inspect_train(args.checkpoint, args.manifest, args.output, args.device, args.batch_size)
    except (OSError, ValueError, KeyError, RuntimeError, TypeError) as error:
        parser.exit(1, f"Диагностика не выполнена: {error}\n")
    print(json.dumps({"output": str(args.output.resolve()), "epoch": summary["checkpoint"]["epoch"],
                      "train_images": summary["images"], "train_groups": summary["groups"],
                      "train_errors": summary["errors"], "checkpoint_sha256": summary["checkpoint"]["sha256"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
