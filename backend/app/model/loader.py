import os

import structlog

from ..config import Settings
from .base import Predictor
from .dummy import DummyPredictor

log = structlog.get_logger(__name__)


def load_predictor(settings: Settings) -> Predictor:
    """Load the configured predictor; see docs/MODEL_CARD.md for the real model."""
    if settings.model_path is None:
        log.warning("model_path_not_set", using="dummy")
        return DummyPredictor()

    if not settings.model_path.exists():
        raise FileNotFoundError(f"MODEL_PATH не найден: {settings.model_path}")

    from .scratch_v2 import ScratchPredictor

    return ScratchPredictor(settings.model_path, expected_sha256=os.getenv("MODEL_SHA256") or None)
