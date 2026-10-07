"""Measure serving cost of the exact checkpoint on the current machine."""

import argparse
import hashlib
import json
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .inference import DEFAULT_CHECKPOINT, Predictor


def main() -> None:
    parser = argparse.ArgumentParser(description="Измерить размер модели, память и скорость CPU inference")
    parser.add_argument("image", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat должен быть положительным")
    image_bytes = args.image.read_bytes()
    start = time.perf_counter()
    predictor = Predictor(args.checkpoint, device="cpu")
    load_seconds = time.perf_counter() - start
    for _ in range(5):
        predictor.predict_bytes(image_bytes)
    latencies_ms = []
    for _ in range(args.repeat):
        start = time.perf_counter()
        predictor.predict_bytes(image_bytes)
        latencies_ms.append((time.perf_counter() - start) * 1000)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_bytes = rss if sys.platform == "darwin" else rss * 1024
    report = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch_version": torch.__version__,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_bytes": args.checkpoint.stat().st_size,
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "model_config": predictor.model_config,
        "image_size": predictor.image_size,
        "tta_horizontal_flip": predictor.tta_horizontal_flip,
        "parameters": sum(parameter.numel() for parameter in predictor.model.parameters()),
        "cpu_threads": torch.get_num_threads(),
        "iterations": args.repeat,
        "model_load_seconds": load_seconds,
        "latency_median_ms": float(np.median(latencies_ms)),
        "latency_p95_ms": float(np.percentile(latencies_ms, 95)),
        "process_peak_rss_mib": rss_bytes / 1024**2,
        "scope": "CPU decode, preprocessing and model forward on the reported machine; HTTP and network latency are excluded.",
    }
    serialized = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
