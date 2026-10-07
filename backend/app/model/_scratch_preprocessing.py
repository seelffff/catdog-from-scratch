"""The same deterministic preprocessing is used for evaluation and serving."""

from collections.abc import Mapping, Sequence

import numpy as np
import torch
from PIL import Image, ImageOps


def letterbox_image(image: Image.Image, side: int) -> Image.Image:
    """Preserve the whole image and fill the empty space with neutral grey."""
    if side <= 0:
        raise ValueError("image_size must be positive")
    image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = image.size
    scale = min(side / width, side / height)
    resized_size = (
        max(1, min(side, round(width * scale))),
        max(1, min(side, round(height * scale))),
    )
    resized = image.resize(resized_size, Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (side, side), (128, 128, 128))
    canvas.paste(resized, ((side - resized.width) // 2, (side - resized.height) // 2))
    return canvas


def preprocess_image(
    image: Image.Image,
    image_size: int,
    normalization: Mapping[str, Sequence[float]] | None = None,
) -> torch.Tensor:
    """Return a float32 CHW tensor, normalized exactly as during training."""
    normalization = normalization or {"mean": (0.5, 0.5, 0.5), "std": (0.5, 0.5, 0.5)}
    mean, std = normalization["mean"], normalization["std"]
    if len(mean) != 3 or len(std) != 3 or any(value <= 0 for value in std):
        raise ValueError("Normalization requires three means and positive standard deviations")
    pixels = np.asarray(letterbox_image(image, image_size), dtype=np.float32) / 255.0
    pixels = (pixels - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    return torch.from_numpy(np.ascontiguousarray(pixels.transpose(2, 0, 1)))
