"""Example: PYTHONPATH=src python -m catsdogs.predict photo.jpg"""

import argparse
import json
from pathlib import Path

from .inference import DEFAULT_CHECKPOINT, Predictor


def main() -> None:
    parser = argparse.ArgumentParser(description="Классификация фотографий кошек и собак")
    parser.add_argument("image", type=Path, help="JPEG, PNG или WebP")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    args = parser.parse_args()
    try:
        if args.image.stat().st_size > 10 * 1024 * 1024:
            parser.error("Файл больше 10 МБ")
        result = Predictor(args.checkpoint, args.device).predict_bytes(args.image.read_bytes())
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        parser.exit(1, f"Ошибка: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
