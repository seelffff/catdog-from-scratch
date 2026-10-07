import asyncio
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import threading

import pytest
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from PIL import Image

from app.config import Settings
from app.images import decode_image
from app.main import MULTIPART_OVERHEAD_BYTES, UploadBodyLimitMiddleware, create_app
from app.model import PredictionResult


def photo(image_format="PNG", size=(15, 31), mode="RGB", orientation=None):
    image = Image.new(mode, size)
    output = BytesIO()
    if orientation is None:
        image.save(output, format=image_format)
    else:
        exif = Image.Exif()
        exif[274] = orientation
        image.save(output, format=image_format, exif=exif)
    return output.getvalue()


class FixedPredictor:
    name = "runtime-fixed"

    def __init__(self, p_cat=0.8):
        self.p_cat = p_cat
        self.size = None

    def predict(self, image):
        assert image.mode == "RGB"
        self.size = image.size
        return PredictionResult(p_cat=self.p_cat)


def upload(client, data=None, path="/api/predict"):
    return client.post(path, files={"file": ("photo.png", photo() if data is None else data, "image/png")})


def test_decoder_applies_exif_before_rgb_and_api_receives_oriented_dimensions():
    data = photo("JPEG", mode="L", orientation=6)
    with decode_image(data) as image:
        assert image.size == (31, 15)
        assert image.mode == "RGB"
        assert 274 not in image.getexif()
    model = FixedPredictor()
    with TestClient(create_app(Settings(), model)) as client:
        assert upload(client, data).status_code == 200
    assert model.size == (31, 15)


@pytest.mark.parametrize("image_format", ["JPEG", "PNG", "WEBP"])
def test_decoder_accepts_only_supported_formats(image_format):
    with decode_image(photo(image_format)) as image:
        assert image.mode == "RGB" and image.size == (15, 31)


@pytest.mark.parametrize("data,expected", [(b"", 400), (b"not a photo", 415), (photo("GIF"), 415), (photo("BMP"), 415)])
def test_runtime_api_preserves_empty_and_unsupported_upload_errors(data, expected):
    with TestClient(create_app(Settings(), FixedPredictor())) as client:
        assert upload(client, data).status_code == expected


def test_decoder_enforces_twenty_megapixel_limit_before_loading(monkeypatch):
    import app.images as images

    monkeypatch.setattr(images, "MAX_PIXELS", 100)
    with pytest.raises(HTTPException) as error:
        decode_image(photo(size=(11, 10)))
    assert error.value.status_code == 413


@pytest.mark.parametrize("size", [(11, 10), (15, 15)])
def test_decoder_handles_pillow_bomb_warning_and_error(monkeypatch, size):
    data = photo(size=size)
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 100)
    with pytest.raises(HTTPException) as error:
        decode_image(data)
    assert error.value.status_code == 413


def test_runtime_api_rejects_oversized_file_inside_allowed_multipart_body():
    data = photo()
    with TestClient(create_app(Settings(max_upload_bytes=len(data) - 1), FixedPredictor())) as client:
        assert upload(client, data).status_code == 413


@pytest.mark.parametrize("chunked", [False, True])
def test_body_limit_rejects_before_multipart_parser_with_and_without_content_length(monkeypatch, chunked):
    async def unexpected_form(*args, **kwargs):
        raise AssertionError("Multipart parser ran on an oversized body")

    monkeypatch.setattr(Request, "_get_form", unexpected_form)
    settings = Settings(max_upload_bytes=100, root_path="/cats")
    limit = settings.max_upload_bytes + MULTIPART_OVERHEAD_BYTES
    with TestClient(create_app(settings, FixedPredictor())) as client:
        headers = {"content-type": "multipart/form-data; boundary=test"}
        if chunked:
            body = iter([b"x" * (limit // 2), b"x" * (limit // 2 + 2)])
        else:
            body = b"x"
            headers["content-length"] = str(limit + 1)
        response = client.post("/cats/api/predict", content=body, headers=headers)
        assert response.status_code == 413
        if chunked:
            assert "content-length" not in response.request.headers


@pytest.mark.parametrize("p_cat", [float("nan"), float("inf"), -float("inf"), -0.01, 1.01, "invalid"])
def test_runtime_api_rejects_invalid_model_probabilities_without_clamping(p_cat):
    model = FixedPredictor(p_cat)
    with TestClient(create_app(Settings(), model)) as client:
        result = upload(client)
        assert result.status_code == 500
        assert "некорректную вероятность" in result.json()["detail"]
        model.p_cat = 0.8
        assert upload(client).status_code == 200


def test_runtime_api_exact_half_probability_selects_dog():
    with TestClient(create_app(Settings(), FixedPredictor(0.5))) as client:
        body = upload(client).json()
    assert body["label"] == "dog"
    assert body["probabilities"] == {"cat": 0.5, "dog": 0.5}


@pytest.mark.parametrize("p_cat,p_dog,label", [
    (0.5, 0.4999999701976776, "cat"),  # Valid float32 softmax rounding near the threshold.
    (1.0, 2.06115369216775e-9, "cat"),  # Preserve the tiny dog probability, instead of 1-p_cat=0.
    (0.2, 0.8, "dog"),
])
def test_api_preserves_both_original_probabilities_and_thresholds_actual_dog(p_cat, p_dog, label):
    class ExactPredictor(FixedPredictor):
        def predict(self, image):
            result = super().predict(image)
            result.p_dog = p_dog
            return result

    with TestClient(create_app(Settings(), ExactPredictor(p_cat))) as client:
        response = upload(client)
    assert response.status_code == 200
    assert response.json()["probabilities"] == {"cat": p_cat, "dog": p_dog}
    assert response.json()["label"] == label


@pytest.mark.parametrize("p_dog", [float("nan"), float("inf"), -0.01, 1.01, "invalid", 0.3])
def test_api_rejects_invalid_exact_dog_probability_or_inconsistent_sum(p_dog):
    class InvalidDogPredictor(FixedPredictor):
        def predict(self, image):
            result = super().predict(image)
            result.p_dog = p_dog
            return result

    with TestClient(create_app(Settings(), InvalidDogPredictor(0.8))) as client:
        assert upload(client).status_code == 500
        assert client.get("/api/health").status_code == 200


def test_prediction_result_keeps_existing_positional_activation_argument():
    activations = {"dense": [0.1, 0.2]}
    result = PredictionResult(0.8, activations)
    assert result.activations is activations and result.p_dog is None


def test_upload_admission_covers_body_parsing_inference_and_complete_response_without_blocking_health():
    async def scenario():
        admission, inference = asyncio.Lock(), asyncio.Lock()
        body_started, body_release = asyncio.Event(), asyncio.Event()
        inference_started, inference_release = asyncio.Event(), asyncio.Event()
        response_started, response_release = asyncio.Event(), asyncio.Event()
        accepted = []

        async def downstream(scope, receive, send):
            if scope["method"] == "GET":
                await JSONResponse({"status": "ok"})(scope, receive, send)
                return
            accepted.append((await receive())["body"])
            assert admission.locked()
            async with inference:
                inference_started.set()
                await inference_release.wait()
            await JSONResponse({"status": "ok"})(scope, receive, send)

        middleware = UploadBodyLimitMiddleware(downstream, 100, inference, admission)
        scope = {"type": "http", "method": "POST", "path": "/cats/api/predict", "root_path": "/cats", "headers": []}
        first_messages = []

        async def first_receive():
            body_started.set()
            await body_release.wait()
            return {"type": "http.request", "body": b"synthetic multipart", "more_body": False}

        async def first_send(message):
            if message["type"] == "http.response.body":
                response_started.set()
                await response_release.wait()
            first_messages.append(message)

        async def no_body_read():
            pytest.fail("A busy request must be rejected before reading or parsing its body")

        async def request_while_busy(method):
            messages = []

            async def send(message):
                messages.append(message)

            request_scope = {**scope, "method": method, "path": "/cats/api/health" if method == "GET" else scope["path"]}
            await middleware(request_scope, no_body_read, send)
            assert messages[0]["status"] == (200 if method == "GET" else 429)

        first = asyncio.create_task(middleware(scope, first_receive, first_send))
        await asyncio.wait_for(body_started.wait(), 1)
        assert admission.locked() and not inference.locked()
        await request_while_busy("POST")
        await request_while_busy("GET")
        body_release.set()
        await asyncio.wait_for(inference_started.wait(), 1)
        assert admission.locked() and inference.locked()
        await request_while_busy("POST")
        inference_release.set()
        await asyncio.wait_for(response_started.wait(), 1)
        assert admission.locked() and not inference.locked()
        await request_while_busy("POST")
        await request_while_busy("GET")
        response_release.set()
        await asyncio.wait_for(first, 1)
        assert not admission.locked() and not inference.locked()
        assert accepted == [b"synthetic multipart"] and first_messages[0]["status"] == 200

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", ["oversize", "disconnect", "downstream_exception"])
def test_upload_admission_is_released_after_rejection_disconnect_or_parser_failure(failure):
    async def scenario():
        admission, inference = asyncio.Lock(), asyncio.Lock()

        async def downstream(scope, receive, send):
            raise RuntimeError("synthetic parser failure")

        middleware = UploadBodyLimitMiddleware(downstream, 10, inference, admission)
        headers = [(b"content-length", b"11")] if failure == "oversize" else []
        scope = {"type": "http", "method": "POST", "path": "/api/predict", "headers": headers}
        messages = []

        async def receive():
            if failure == "disconnect":
                return {"type": "http.disconnect"}
            return {"type": "http.request", "body": b"x", "more_body": False}

        async def send(message):
            messages.append(message)

        if failure == "downstream_exception":
            with pytest.raises(RuntimeError, match="parser failure"):
                await middleware(scope, receive, send)
        else:
            await middleware(scope, receive, send)
        assert not admission.locked() and not inference.locked()
        if failure == "oversize":
            assert messages[0]["status"] == 413

    asyncio.run(scenario())


def test_busy_inference_returns_429_while_health_stays_responsive():
    entered, release = threading.Event(), threading.Event()

    class BlockingPredictor(FixedPredictor):
        def predict(self, image):
            entered.set()
            assert release.wait(5)
            return super().predict(image)

    with TestClient(create_app(Settings(), BlockingPredictor())) as client, ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(upload, client)
        assert entered.wait(2)
        try:
            assert client.get("/api/health").json() == {"status": "ok", "model": "runtime-fixed"}
            assert upload(client).status_code == 429
        finally:
            release.set()
        assert first.result().status_code == 200
        assert upload(client).status_code == 200


def test_root_path_environment_static_assets_and_openapi_prefix(tmp_path, monkeypatch):
    monkeypatch.setenv("ROOT_PATH", "/cats/")
    settings = Settings.from_env()
    assert settings.root_path == "/cats"
    (tmp_path / "index.html").write_text('<title>catdog</title><script src="app.js"></script>')
    (tmp_path / "app.js").write_text("console.log('site');")
    app = create_app(Settings(static_dir=tmp_path, root_path=settings.root_path), FixedPredictor())
    with TestClient(app) as client:
        assert "catdog" in client.get("/cats/").text
        assert client.get("/cats/app.js").status_code == 200
        assert client.get("/cats/api/health").json()["status"] == "ok"
        assert client.get("/api/health").status_code == 200
        assert upload(client, path="/cats/api/predict").status_code == 200
        assert "/cats/openapi.json" in client.get("/cats/docs").text
        assert {"url": "/cats"} in client.get("/cats/openapi.json").json()["servers"]
