"""Classify every actual image in a ZIP using the frozen final V2 checkpoint."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
import sys
import time
import zipfile

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from catsdogs.inference import Predictor, decode_image
from catsdogs.preprocessing import preprocess_image

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def checksum(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def entries_with_ids(archive: zipfile.ZipFile):
    entries = []
    skipped = 0
    for entry in archive.infolist():
        path = PurePosixPath(entry.filename)
        if entry.is_dir():
            continue
        if "__MACOSX" in path.parts or path.name.startswith("._"):
            skipped += 1
            continue
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"Неожиданный файл в архиве: {entry.filename}")
        if not path.stem.isdecimal():
            raise ValueError(f"ID должен быть числом в имени фотографии: {entry.filename}")
        image_id = int(path.stem)
        if str(image_id) != path.stem:
            raise ValueError(f"ID содержит ведущие нули: {entry.filename}; требуется сохранить его как текст")
        entries.append((image_id, entry))
    entries.sort(key=lambda item: item[0])
    ids = [image_id for image_id, _ in entries]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Архив пуст либо содержит повторяющиеся ID")
    return entries, skipped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "models/v2_best.pt")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS недоступен этому процессу; запустите с доступом к GPU либо --device cpu")
    expected = json.loads((ROOT / "reports/v2/evaluation.json").read_text())["model"]["checkpoint_sha256"]
    model_sha = checksum(args.checkpoint)
    if model_sha != expected:
        raise ValueError("Веса отличаются от проверенной финальной V2 модели")
    torch.set_num_threads(1)
    predictor = Predictor(args.checkpoint, device=args.device)
    assert predictor.class_names == ["cat", "dog"]
    assert predictor.image_size == 256 and predictor.threshold == 0.5
    assert predictor.tta_horizontal_flip is False
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {"predictions": args.output_dir / f"{args.archive.stem}_predictions.csv",
               "details": args.output_dir / f"{args.archive.stem}_prediction_details.csv",
               "summary": args.output_dir / f"{args.archive.stem}_inference_summary.json"}
    if any(path.exists() for path in outputs.values()):
        raise FileExistsError("Результаты уже существуют; выберите другую output-dir для нового прогона")
    started = time.perf_counter()
    records = []
    cpu_predictor = None
    cpu_rechecked = 0
    last_progress = started
    with zipfile.ZipFile(args.archive) as archive:
        entries, skipped = entries_with_ids(archive)
        print(json.dumps({"status": "started", "images": len(entries), "ignored_macos_metadata": skipped,
                          "device": args.device, "batch_size": args.batch_size, "image_size": predictor.image_size,
                          "checkpoint_sha256": model_sha}), flush=True)
        with torch.inference_mode():
            for offset in range(0, len(entries), args.batch_size):
                batch = entries[offset:offset + args.batch_size]
                tensors = []
                for image_id, entry in batch:
                    try:
                        image = decode_image(archive.read(entry))
                        tensors.append(preprocess_image(image, predictor.image_size,
                                                        {"mean": predictor.mean, "std": predictor.std}))
                    except Exception as exc:
                        raise RuntimeError(f"Не удалось обработать ID {image_id}: {entry.filename}") from exc
                inputs = torch.stack(tensors).to(predictor.device)
                logits = predictor.model(inputs)
                if tuple(logits.shape) != (len(batch), 2):
                    raise RuntimeError("Неверная форма выхода модели")
                probabilities = torch.softmax(logits / predictor.temperature, dim=1).cpu().tolist()
                for (image_id, entry), tensor, values in zip(batch, tensors, probabilities, strict=True):
                    if len(values) != 2 or not all(math.isfinite(v) and 0 <= v <= 1 for v in values):
                        raise RuntimeError(f"Некорректный выход для ID {image_id}")
                    # Resolve very close GPU decisions on the same CPU path used by the site.
                    if args.device == "mps" and abs(values[1] - predictor.threshold) < 0.001:
                        if cpu_predictor is None:
                            cpu_predictor = Predictor(args.checkpoint, device="cpu")
                        values = torch.softmax(cpu_predictor.model(tensor.unsqueeze(0)) /
                                               cpu_predictor.temperature, dim=1)[0].tolist()
                        cpu_rechecked += 1
                    label = int(values[1] >= predictor.threshold)
                    records.append((image_id, label, entry.filename, values[0], values[1]))
                now = time.perf_counter()
                if now - last_progress >= 10 or len(records) == len(entries):
                    elapsed = now - started
                    rate = len(records) / elapsed
                    print(json.dumps({"status": "progress", "done": len(records), "total": len(entries),
                                      "percent": round(100 * len(records) / len(entries), 1),
                                      "elapsed_seconds": round(elapsed, 1),
                                      "estimated_remaining_seconds": round((len(entries) - len(records)) / rate, 1)}), flush=True)
                    last_progress = now
    assert len(records) == len(entries)
    assert len({record[0] for record in records}) == len(records)
    assert all(record[1] in (0, 1) for record in records)
    for key, columns in (("predictions", ["id", "label"]),
                         ("details", ["id", "label", "archive_entry", "p_cat", "p_dog"])):
        path = outputs[key]
        partial = path.with_suffix(path.suffix + ".part")
        with partial.open("x", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(columns)
            writer.writerows(record[:2] if key == "predictions" else record for record in records)
        partial.replace(path)
    counts = Counter(record[1] for record in records)
    summary = {"status": "passed", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
               "archive_filename": args.archive.name, "archive_sha256": checksum(args.archive),
               "checkpoint_filename": args.checkpoint.name, "checkpoint_sha256": model_sha,
               "images": len(records), "unique_ids": len(records), "id_min": records[0][0], "id_max": records[-1][0],
               "labels": {"0_cat": counts[0], "1_dog": counts[1]}, "failed_images": 0,
               "ignored_macos_metadata": skipped, "device": args.device, "batch_size": args.batch_size,
               "image_size": predictor.image_size, "temperature": predictor.temperature,
               "threshold": predictor.threshold, "tta_horizontal_flip": predictor.tta_horizontal_flip,
               "cpu_rechecked_near_threshold": cpu_rechecked, "training_performed": False,
               "elapsed_seconds": round(time.perf_counter() - started, 3),
               "prediction_csv_sha256": checksum(outputs["predictions"])}
    outputs["summary"].write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
