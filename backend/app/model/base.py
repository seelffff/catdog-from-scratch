from dataclasses import dataclass
from typing import Protocol

from PIL import Image

Map2D = list[list[float]]


@dataclass(slots=True)
class PredictionResult:
    p_cat: float
    """Вероятность класса «кошка», 0..1."""

    activations: dict[str, list[Map2D] | list[float]] | None = None
    """Необязательно: ключи conv1, relu1, pool1, conv2, pool2 → [каналы][H][W]; dense → [N]."""

    p_dog: float | None = None
    """Исходная вероятность «собаки»; при отсутствии API использует 1 − p_cat."""


class Predictor(Protocol):
    """Интерфейс модели. Любой класс с полем name и методом predict подходит."""

    name: str

    def predict(self, image: Image.Image) -> PredictionResult:
        """image — RGB любого размера; препроцессинг (resize, normalize) делает сама модель."""
        ...
