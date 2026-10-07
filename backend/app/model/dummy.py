from PIL import Image, ImageStat

from .base import PredictionResult


class DummyPredictor:
    """Заглушка, пока нет обученной модели. Ответ зависит от цветов картинки и ничего не значит."""

    name = "dummy"

    def predict(self, image: Image.Image) -> PredictionResult:
        r, g, b = ImageStat.Stat(image.resize((32, 32))).mean
        warmth = (r - b) / 255  # −1..1
        p_cat = min(0.95, max(0.05, 0.5 + warmth * 1.5))
        return PredictionResult(p_cat=p_cat)
