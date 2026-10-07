import asyncio
import math
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, HTTPException, Request, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from .config import Settings
from .images import decode_image
from .log import configure_logging
from .model import Predictor, load_predictor
from .schemas import Activations, HealthResponse, PredictResponse, Probabilities

log = structlog.get_logger(__name__)
MULTIPART_OVERHEAD_BYTES = 64 * 1024


class UploadBodyLimitMiddleware:
    """Bound uploads before the multipart parser allocates files or fields."""

    def __init__(self, app: ASGIApp, max_body_bytes: int, inference_lock: asyncio.Lock,
                 upload_admission_lock: asyncio.Lock):
        self.app = app
        self.max_body_bytes = max_body_bytes
        self.inference_lock = inference_lock
        self.upload_admission_lock = upload_admission_lock

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        path = scope.get("path", "")
        prefix = scope.get("root_path", "").rstrip("/")
        if prefix and path.startswith(prefix + "/"):
            path = path[len(prefix):]
        if scope["type"] != "http" or scope.get("method") != "POST" or path != "/api/predict":
            await self.app(scope, receive, send)
            return

        if self.upload_admission_lock.locked() or self.inference_lock.locked():
            await JSONResponse({"detail": "Модель занята. Попробуйте через пару секунд"},
                               status_code=429)(scope, receive, send)
            return
        # Reserve memory and spool space before reading any body or parsing multipart.
        # This distinct lock covers the complete response without reacquiring inference_lock.
        async with self.upload_admission_lock:
            await self._bounded_upload(scope, receive, send)

    async def _bounded_upload(self, scope: Scope, receive: Receive, send: Send) -> None:
        async def reject(code: int, detail: str) -> None:
            await JSONResponse({"detail": detail}, status_code=code)(scope, receive, send)

        content_length = dict(scope.get("headers", [])).get(b"content-length")
        if content_length is not None:
            try:
                length = int(content_length)
                if length < 0:
                    raise ValueError
            except ValueError:
                await reject(400, "Некорректный размер запроса")
                return
            if length > self.max_body_bytes:
                await reject(413, "Файл слишком большой")
                return

        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunk = message.get("body", b"")
            if len(body) + len(chunk) > self.max_body_bytes:
                await reject(413, "Файл слишком большой")
                return
            body.extend(chunk)
            if not message.get("more_body", False):
                break
        payload = bytes(body)
        body.clear()
        delivered = False

        async def bounded_receive() -> Message:
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": payload, "more_body": False}
            return await receive()

        await self.app(scope, bounded_receive, send)


def create_app(settings: Settings | None = None, predictor: Predictor | None = None) -> FastAPI:
    """Фабрика приложения: `uvicorn app.main:create_app --factory`."""
    settings = settings or Settings.from_env()
    configure_logging(settings.log_json)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.predictor = predictor or load_predictor(settings)
        log.info("model_ready", model=app.state.predictor.name)
        yield

    app = FastAPI(title="Кошка или собака", version="0.1.0", lifespan=lifespan, root_path=settings.root_path.rstrip("/"))
    app.state.inference_lock = asyncio.Lock()
    app.state.upload_admission_lock = asyncio.Lock()
    app.add_middleware(
        UploadBodyLimitMiddleware,
        max_body_bytes=settings.max_upload_bytes + MULTIPART_OVERHEAD_BYTES,
        inference_lock=app.state.inference_lock,
        upload_admission_lock=app.state.upload_admission_lock,
    )

    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_methods=["GET", "POST"],
            allow_headers=["*"],
        )

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=uuid.uuid4().hex[:12], path=request.url.path)
        return await call_next(request)

    @app.get("/api/health", response_model=HealthResponse)
    async def health(request: Request) -> HealthResponse:
        return HealthResponse(model=request.app.state.predictor.name)

    @app.post("/api/predict", response_model=PredictResponse, response_model_exclude_none=True)
    async def predict(request: Request, file: UploadFile) -> PredictResponse:
        lock = request.app.state.inference_lock
        if lock.locked():
            raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "Модель занята. Попробуйте через пару секунд")
        async with lock:
            data = await file.read(settings.max_upload_bytes + 1)
            if len(data) > settings.max_upload_bytes:
                raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, "Файл слишком большой")

            model: Predictor = request.app.state.predictor
            started = time.perf_counter()

            def infer():
                with decode_image(data) as image:
                    return model.predict(image), image.size

            result, image_size = await run_in_threadpool(infer)
            try:
                p_cat = float(result.p_cat)
                raw_p_dog = getattr(result, "p_dog", None)
                p_dog = 1 - p_cat if raw_p_dog is None else float(raw_p_dog)
            except (TypeError, ValueError, OverflowError) as exc:
                raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Модель вернула некорректную вероятность") from exc
            if (not math.isfinite(p_cat) or not 0 <= p_cat <= 1
                    or not math.isfinite(p_dog) or not 0 <= p_dog <= 1
                    or not math.isclose(p_cat + p_dog, 1.0, rel_tol=0.0, abs_tol=1e-6)):
                raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Модель вернула некорректную вероятность")
            label = "dog" if p_dog >= 0.5 else "cat"
            log.info(
                "prediction",
                model=model.name,
                label=label,
                p_cat=round(p_cat, 4),
                size=image_size,
                ms=round((time.perf_counter() - started) * 1000, 1),
            )
            return PredictResponse(
                label=label,
                probabilities=Probabilities(cat=p_cat, dog=p_dog),
                model=model.name,
                activations=Activations.model_validate(result.activations) if result.activations else None,
            )

    # Собранный фронт (frontend/dist). Монтируется последним, чтобы не перекрывать /api.
    if settings.static_dir and settings.static_dir.is_dir():
        app.mount("/", StaticFiles(directory=settings.static_dir, html=True), name="frontend")
    elif settings.static_dir:
        log.warning("static_dir_missing", path=str(settings.static_dir))

    return app
