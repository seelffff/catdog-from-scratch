import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Self


def _path(name: str) -> Path | None:
    value = os.getenv(name, "").strip()
    return Path(value) if value else None


@dataclass(frozen=True, slots=True)
class Settings:
    """Настройки из переменных окружения (см. .env.example в корне)."""

    model_path: Path | None = None
    static_dir: Path | None = None
    cors_origins: list[str] = field(default_factory=list)
    max_upload_bytes: int = 20 * 1024 * 1024
    log_json: bool = False
    root_path: str = ""

    @classmethod
    def from_env(cls) -> Self:
        origins = os.getenv("CORS_ORIGINS", "")
        return cls(
            model_path=_path("MODEL_PATH"),
            static_dir=_path("STATIC_DIR"),
            cors_origins=[o.strip() for o in origins.split(",") if o.strip()],
            max_upload_bytes=int(os.getenv("MAX_UPLOAD_MB", "20")) * 1024 * 1024,
            log_json=os.getenv("LOG_JSON", "0") == "1",
            root_path=os.getenv("ROOT_PATH", "").rstrip("/"),
        )
