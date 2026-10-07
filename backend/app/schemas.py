"""Контракт API. Фронт ждёт ровно эти поля (frontend/src/api/types.ts)."""

from typing import Literal

from pydantic import BaseModel, Field

Map2D = list[list[float]]


class Probabilities(BaseModel):
    cat: float = Field(ge=0, le=1)
    dog: float = Field(ge=0, le=1)


class Activations(BaseModel):
    """Необязательные промежуточные выходы для визуализации: [каналы][H][W], до 9 каналов на слой."""

    conv1: list[Map2D] | None = None
    relu1: list[Map2D] | None = None
    pool1: list[Map2D] | None = None
    conv2: list[Map2D] | None = None
    pool2: list[Map2D] | None = None
    dense: list[float] | None = None


class PredictResponse(BaseModel):
    label: Literal["cat", "dog"]
    probabilities: Probabilities
    model: str
    activations: Activations | None = None


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    model: str
