"""Load a local checkpoint and classify one image on CPU or Apple MPS."""

from __future__ import annotations

import io
import math
import warnings
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageOps, UnidentifiedImageError

from .preprocessing import preprocess_image
from .paths import PROJECT_ROOT

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
SUPPORTED_FORMATS = frozenset({"JPEG", "PNG", "WEBP"})
DEFAULT_CHECKPOINT = PROJECT_ROOT / "models" / "v2_best.pt"


class InvalidImage(ValueError):
    """An uploaded file cannot be decoded safely as a supported image."""


def decode_image(data: bytes) -> Image.Image:
    if not data:
        raise InvalidImage("Файл пуст. Выберите фотографию.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise InvalidImage("Файл больше 10 МБ. Загрузите фотографию меньшего размера.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as image:
                if image.format not in SUPPORTED_FORMATS:
                    raise InvalidImage("Поддерживаются фотографии JPEG, PNG и WebP.")
                if image.width * image.height > MAX_IMAGE_PIXELS:
                    raise InvalidImage("Изображение слишком большое: максимум 20 миллионов пикселей.")
                image.load()
                return ImageOps.exif_transpose(image).convert("RGB")
    except InvalidImage:
        raise
    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        Image.DecompressionBombWarning,
        Image.DecompressionBombError,
    ) as exc:
        raise InvalidImage("Не удалось прочитать изображение. Выберите корректный JPEG, PNG или WebP.") from exc


class Predictor:
    """A single, reusable model instance; no weights are fetched from the network."""

    def __init__(self, checkpoint_path: str | Path = DEFAULT_CHECKPOINT, device: str = "cpu"):
        from .model import build_model

        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        self.device = torch.device(device)
        if self.device.type == "cpu":
            torch.set_num_threads(1)
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
        self.image_size = int(checkpoint["image_size"])
        self.class_names = list(checkpoint["class_names"])
        if self.class_names != ["cat", "dog"]:
            raise ValueError("The checkpoint must use class_names=['cat', 'dog']")
        if checkpoint.get("resize_mode", "letterbox") != "letterbox":
            raise ValueError("Unsupported checkpoint resize_mode")
        normalization = checkpoint["normalization"]
        self.mean = normalization["mean"]
        self.std = normalization["std"]
        self.temperature = float(checkpoint.get("temperature", 1.0))
        self.threshold = float(checkpoint.get("threshold", 0.5))
        self.tta_horizontal_flip = bool(checkpoint.get("tta_horizontal_flip", False))
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be a finite positive number")
        if not 0 < self.threshold < 1:
            raise ValueError("threshold must be between zero and one")
        self.model_config = dict(checkpoint["model_config"])
        self.model = build_model(**self.model_config)
        self.model.load_state_dict(checkpoint["state_dict"], strict=True)
        self.model.to(self.device).eval()

    @torch.inference_mode()
    def predict_image(self, image: Image.Image) -> dict[str, Any]:
        tensor = preprocess_image(image, self.image_size, {"mean": self.mean, "std": self.std})
        logits = self.model(tensor.unsqueeze(0).to(self.device))
        if logits.shape != (1, 2):
            raise RuntimeError("Model output must contain one score for each of two classes")
        probabilities_tensor = torch.softmax(logits / self.temperature, dim=1)
        if self.tta_horizontal_flip:
            flipped_logits = self.model(torch.flip(tensor.unsqueeze(0).to(self.device), dims=(3,)))
            if flipped_logits.shape != (1, 2):
                raise RuntimeError("Model output must contain one score for each of two classes")
            probabilities_tensor = (probabilities_tensor + torch.softmax(flipped_logits / self.temperature, dim=1)) / 2
        probabilities = probabilities_tensor[0].cpu().tolist()
        if not all(math.isfinite(value) for value in probabilities):
            raise RuntimeError("The model produced non-finite probabilities")
        class_index = int(probabilities[1] >= self.threshold)
        return {
            "class_name": self.class_names[class_index],
            "label": "Кошка" if class_index == 0 else "Собака",
            "class_index": class_index,
            "probability": probabilities[class_index],
            "probabilities": dict(zip(self.class_names, probabilities, strict=True)),
            "image_size": {"width": image.width, "height": image.height},
        }

    def predict_bytes(self, data: bytes) -> dict[str, Any]:
        return self.predict_image(decode_image(data))
