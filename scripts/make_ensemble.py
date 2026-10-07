"""Package own scratch checkpoints into a validation-only ensemble candidate."""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import math
import os
import re
import sys
import tempfile
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from catsdogs.model import build_model


def make_candidate(checkpoint_paths: list[Path], output: Path, image_size: int,
                   weights: list[float] | None = None) -> dict:
    if not 2 <= len(checkpoint_paths) <= 3:
        raise ValueError("Нужны два или три checkpoint собственных сетей")
    if image_size < 64 or image_size > 1024:
        raise ValueError("Общее разрешение должно быть от 64 до 1024 пикселей")
    if output.exists():
        raise FileExistsError("Укажите новый output: существующий checkpoint не перезаписывается")
    checkpoints, records = [], []
    for path in checkpoint_paths:
        path = path.resolve()
        payload = path.read_bytes()
        checkpoint = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
        if checkpoint.get("pretrained") is not False:
            raise ValueError("Можно объединять только собственные scratch-checkpoint с pretrained=False")
        config = checkpoint["model_config"]
        if config.get("name", "tiny_resnet18") not in {"tiny_resnet18", "robust_resnet18"}:
            raise ValueError("Членами ансамбля должны быть отдельные TinyResNet или RobustResNet; вложенные ансамбли запрещены")
        if checkpoint["class_names"] != ["cat", "dog"] or checkpoint.get("resize_mode") != "letterbox":
            raise ValueError("Ожидаются классы cat/dog и предобработка letterbox")
        normalization = checkpoint["normalization"]
        mean, std = normalization["mean"], normalization["std"]
        if len(mean) != 3 or len(std) != 3 or any(not math.isfinite(float(value)) for value in [*mean, *std]) or any(value <= 0 for value in std):
            raise ValueError("Некорректная нормализация checkpoint")
        manifest_hash = checkpoint.get("train_config", {}).get("manifest_sha256", "")
        if not isinstance(manifest_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", manifest_hash):
            raise ValueError("Каждый checkpoint должен содержать SHA-256 исходного разделения данных")
        if checkpoints:
            previous = checkpoints[0]
            fields = ("normalization", "class_names", "resize_mode")
            if any(checkpoint[field] != previous[field] for field in fields):
                raise ValueError("Нормализация, классы и resize_mode должны совпадать")
            if manifest_hash != previous["train_config"]["manifest_sha256"]:
                raise ValueError("Члены ансамбля обучены с разными разделениями датасета")
        checkpoints.append(checkpoint)
        records.append({"checkpoint": str(path), "sha256": hashlib.sha256(payload).hexdigest(),
                        "image_size": int(checkpoint["image_size"]), "epoch": int(checkpoint.get("epoch", 0)),
                        "model_config": copy.deepcopy(config), "val_source": checkpoint.get("val_source", "raw")})
    model_config = {"name": "logit_ensemble", "members": [copy.deepcopy(item["model_config"]) for item in checkpoints],
                    "mixture_weights": weights if weights is not None else [1.0] * len(checkpoints)}
    model = build_model(**model_config).eval()
    model_config["mixture_weights"] = model.mixture_weights.tolist()
    for member, checkpoint in zip(model.members, checkpoints, strict=True):
        member.load_state_dict(checkpoint["state_dict"], strict=True)
    for record, weight in zip(records, model_config["mixture_weights"], strict=True):
        record["mixture_weight"] = weight
    candidate = {
        "model_config": model_config,
        "state_dict": model.state_dict(),
        "image_size": image_size,
        "class_names": copy.deepcopy(checkpoints[0]["class_names"]),
        "normalization": copy.deepcopy(checkpoints[0]["normalization"]),
        "resize_mode": checkpoints[0]["resize_mode"],
        "temperature": 1.0, "threshold": 0.5, "tta_horizontal_flip": False,
        "epoch": 0, "val_source": "own_model_ensemble", "pretrained": False,
        "selection_frozen_before_test": False,
        "candidate_validation_pending": True,
        "train_config": {"manifest_sha256": checkpoints[0]["train_config"]["manifest_sha256"],
                         "initialization": "own_model_ensemble", "pretrained": False},
        "provenance": {"method": "weighted_mean_of_logits", "members": records,
                       "common_image_size": image_size,
                       "data_access": "Candidate formation did not read validation or test images"},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, prefix="ensemble-", suffix=".pt.part", delete=False) as file:
            temporary = Path(file.name)
        torch.save(candidate, temporary)
        # Link only if the destination is absent, making publication atomic and exclusive.
        os.link(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {"checkpoint": str(output.resolve()), "bytes": output.stat().st_size,
            "model_config": model_config, "image_size": image_size,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "provenance": candidate["provenance"], "validation_required": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--weights", type=float, nargs="+")
    parser.add_argument("--image-size", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    try:
        result = make_candidate(args.checkpoints, args.output, args.image_size, args.weights)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        parser.exit(1, f"Ошибка подготовки ансамбля: {exc}\n")


if __name__ == "__main__":
    main()
