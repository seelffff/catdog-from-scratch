"""Validation-only selection followed by one frozen final test evaluation."""

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader

from .model import build_model
from .train import ImageDataset, PROJECT_ROOT, json_write


def softmax(logits, temperature=1.0):
    values = logits / temperature
    values = values - values.max(axis=1, keepdims=True)
    exponentials = np.exp(values)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def probabilities(logits, flipped_logits=None, temperature=1.0):
    result = softmax(logits, temperature)
    if flipped_logits is not None:
        result = (result + softmax(flipped_logits, temperature)) / 2
    return result


def nll(probs, labels):
    return float(-np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-12, 1)).mean())


def wilson(correct, total, z=1.959963984540054):
    fraction = correct / total
    denominator = 1 + z * z / total
    center = (fraction + z * z / (2 * total)) / denominator
    distance = z * math.sqrt(fraction * (1 - fraction) / total + z * z / (4 * total * total)) / denominator
    return [center - distance, center + distance]


def predicted_classes(probs, threshold=0.5):
    """Apply the same inclusive dog-probability threshold as Predictor."""
    return (probs[:, 1] >= threshold).astype(np.int64)


def metrics(probs, labels):
    predicted = predicted_classes(probs)
    matrix = np.zeros((2, 2), dtype=np.int64)
    for actual, guess in zip(labels, predicted, strict=True):
        matrix[actual, guess] += 1
    correct = int(np.trace(matrix))
    per_class = {}
    for index, name in enumerate(["cat", "dog"]):
        tp = int(matrix[index, index])
        precision = tp / max(int(matrix[:, index].sum()), 1)
        recall = tp / max(int(matrix[index, :].sum()), 1)
        per_class[name] = {"precision": precision, "recall": recall,
                           "f1": 2 * precision * recall / max(precision + recall, 1e-12),
                           "support": int(matrix[index, :].sum())}
    confidence = probs.max(axis=1)
    calibration_error = 0.0
    for lower, upper in zip(np.linspace(0, 1, 16)[:-1], np.linspace(0, 1, 16)[1:]):
        selected = (confidence > lower) & (confidence <= upper)
        if selected.any():
            calibration_error += selected.mean() * abs(float((predicted[selected] == labels[selected]).mean())
                                                       - float(confidence[selected].mean()))
    from sklearn.metrics import roc_auc_score
    return {"images": len(labels), "correct": correct, "errors": len(labels) - correct,
            "accuracy": correct / len(labels), "balanced_accuracy": float(np.mean([v["recall"] for v in per_class.values()])),
            "accuracy_wilson_95_ci": wilson(correct, len(labels)),
            "macro_f1": float(np.mean([v["f1"] for v in per_class.values()])),
            "per_class": per_class, "confusion_matrix": matrix.tolist(),
            "confusion_matrix_order": ["cat", "dog"], "roc_auc": float(roc_auc_score(labels, probs[:, 1])),
            "negative_log_likelihood": nll(probs, labels), "ece_15_bins": float(calibration_error)}


@torch.inference_mode()
def collect(model, manifest, split, image_size, device, horizontal_flip=False):
    dataset = ImageDataset(manifest, split, image_size)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
    logits, flipped, labels = [], [], []
    model.eval()
    for batch, targets in loader:
        batch = batch.to(device)
        logits.append(model(batch).cpu().numpy())
        if horizontal_flip:
            flipped.append(model(torch.flip(batch, dims=[3])).cpu().numpy())
        labels.append(targets.numpy())
    return np.concatenate(logits), np.concatenate(flipped) if flipped else None, np.concatenate(labels), dataset.rows


def save_predictions(path, rows, labels, probs):
    predictions = []
    for row, label, values, predicted in zip(rows, labels, probs, predicted_classes(probs), strict=True):
        predicted = int(predicted)
        predictions.append({"relative_path": row["relative_path"], "group_id": row["group_id"],
                            "true_class": ["cat", "dog"][label], "predicted_class": ["cat", "dog"][predicted],
                            "cat_probability": float(values[0]), "dog_probability": float(values[1]),
                            "correct": bool(predicted == label)})
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    return predictions


def gallery(path, predictions, root, mistakes=True):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    selected = [row for row in predictions if row["correct"] != mistakes]
    selected.sort(key=lambda row: max(row["cat_probability"], row["dog_probability"]), reverse=True)
    selected = selected[:16]
    if not selected:
        return
    figure, axes = plt.subplots(math.ceil(len(selected) / 4), 4, figsize=(12, 3 * math.ceil(len(selected) / 4)), squeeze=False)
    for axis in axes.ravel():
        axis.axis("off")
    for axis, row in zip(axes.ravel(), selected):
        with Image.open(root / row["relative_path"]) as image:
            axis.imshow(ImageOps.exif_transpose(image).convert("RGB"))
        axis.set_title(f'{Path(row["relative_path"]).name}\nTrue: {row["true_class"]}; predicted: {row["predicted_class"]}\n'
                       f'Confidence: {max(row["cat_probability"], row["dog_probability"]):.1%}', fontsize=9)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "data/splits.csv")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "reports")
    parser.add_argument("--device", choices=["cpu", "mps"], default="mps")
    parser.add_argument("--finalize", action="store_true", help="Freeze validation choices, save models/best.pt, then evaluate test")
    parser.add_argument("--model-output", type=Path, default=PROJECT_ROOT / "models/best.pt")
    args = parser.parse_args()
    if args.finalize and (args.output / "evaluation.json").exists():
        raise FileExistsError("A frozen final evaluation already exists in this folder; do not overwrite it during model selection")
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("GPU access is needed to use MPS")
    torch.set_num_threads(1)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("pretrained") is not False:
        raise ValueError("This project evaluates only our models trained without external pretrained weights")
    manifest_hash = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    training_manifest_hash = checkpoint.get("train_config", {}).get("manifest_sha256")
    if training_manifest_hash is not None and manifest_hash != training_manifest_hash:
        raise ValueError("Manifest SHA-256 differs from the training checkpoint; use the original audited split to prevent data leakage")
    model = build_model(**checkpoint["model_config"]).to(device).eval()
    model.load_state_dict(checkpoint["state_dict"])
    logits, flipped, labels, rows = collect(model, args.manifest, "val", checkpoint["image_size"], device, True)
    plain = probabilities(logits)
    with_flip = probabilities(logits, flipped)
    plain_metrics, flip_metrics = metrics(plain, labels), metrics(with_flip, labels)
    chosen_flip = flip_metrics["balanced_accuracy"] > plain_metrics["balanced_accuracy"] + 0.001
    chosen = with_flip if chosen_flip else plain
    temperatures = np.geomspace(0.25, 4.0, 101)
    scores = [nll(probabilities(logits, flipped if chosen_flip else None, float(t)), labels) for t in temperatures]
    temperature = float(temperatures[int(np.argmin(scores))])
    calibrated = probabilities(logits, flipped if chosen_flip else None, temperature)
    report = {"model": {"architecture": checkpoint["model_config"], "image_size": checkpoint["image_size"],
                        "parameter_count": sum(p.numel() for p in model.parameters()), "pretrained": False},
              "dataset": {"manifest_sha256": manifest_hash},
              "selection": {"source_checkpoint": str(args.checkpoint), "source_epoch": checkpoint["epoch"],
                            "source_checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                            "validation_weights": checkpoint.get("val_source", "raw"),
                            "own_training_provenance": checkpoint.get("provenance"),
                            "plain_validation": plain_metrics, "flip_validation": flip_metrics,
                            "tta_horizontal_flip": chosen_flip, "selection_split": "val"},
              "calibration": {"method": "temperature scaling, validation-only log-grid search",
                              "temperature": temperature, "threshold": 0.5,
                              "val_nll_before": nll(chosen, labels), "val_nll_after": nll(calibrated, labels)},
              "validation": metrics(calibrated, labels)}
    val_predictions = save_predictions(args.output / "validation_predictions.csv", rows, labels, calibrated)
    gallery(args.output / "validation_errors.png", val_predictions, args.manifest.resolve().parent.parent)
    if args.finalize:
        checkpoint["temperature"] = temperature
        checkpoint["threshold"] = 0.5
        checkpoint["tta_horizontal_flip"] = chosen_flip
        checkpoint["validation"] = report["validation"]
        checkpoint["calibration"] = report["calibration"]
        checkpoint["selection_frozen_before_test"] = True
        checkpoint["candidate_validation_pending"] = False
        report["selection"]["frozen_at_utc"] = datetime.now(timezone.utc).isoformat()
        args.model_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, args.model_output)
        report["model"]["checkpoint_bytes"] = args.model_output.stat().st_size
        report["model"]["checkpoint_sha256"] = hashlib.sha256(args.model_output.read_bytes()).hexdigest()
        test_logits, test_flip, test_labels, test_rows = collect(model, args.manifest, "test", checkpoint["image_size"], device, chosen_flip)
        test_probs = probabilities(test_logits, test_flip, temperature)
        report["test"] = metrics(test_probs, test_labels)
        seen = set()
        representatives = []
        for index, row in enumerate(test_rows):
            if row["group_id"] not in seen:
                seen.add(row["group_id"])
                representatives.append(index)
        report["test_unique_groups"] = metrics(test_probs[representatives], test_labels[representatives])
        report["confidence_interval_note"] = "Wilson 95% interval; image-level approximate. Unique-group result removes detected duplicate pairs."
        predictions = save_predictions(args.output / "test_predictions.csv", test_rows, test_labels, test_probs)
        gallery(args.output / "test_errors.png", predictions, args.manifest.resolve().parent.parent)
        gallery(args.output / "test_correct_examples.png", predictions, args.manifest.resolve().parent.parent, mistakes=False)
        json_write(args.output / "evaluation.json", report)
    else:
        json_write(args.output / "validation.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
