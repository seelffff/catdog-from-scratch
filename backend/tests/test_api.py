from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.config import Settings
from app.main import create_app
from app.model import PredictionResult


def png(color: tuple[int, int, int] = (200, 120, 60), size: tuple[int, int] = (40, 30)) -> bytes:
    buf = BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


class FixedPredictor:
    name = "fixed"

    def __init__(self, p_cat: float, activations=None):
        self.p_cat = p_cat
        self.activations = activations

    def predict(self, image: Image.Image) -> PredictionResult:
        assert image.mode == "RGB"
        return PredictionResult(p_cat=self.p_cat, activations=self.activations)


@pytest.fixture
def make_client():
    def factory(predictor=None, **settings) -> TestClient:
        return TestClient(create_app(Settings(**settings), predictor=predictor))

    return factory


def test_health_reports_dummy_without_model(make_client):
    with make_client() as client:
        assert client.get("/api/health").json() == {"status": "ok", "model": "dummy"}


def test_predict_contract(make_client):
    with make_client(FixedPredictor(0.8)) as client:
        res = client.post("/api/predict", files={"file": ("a.png", png(), "image/png")})
    assert res.status_code == 200
    body = res.json()
    assert body["label"] == "cat"
    assert body["model"] == "fixed"
    assert body["probabilities"]["cat"] == pytest.approx(0.8)
    assert body["probabilities"]["dog"] == pytest.approx(0.2)
    assert "activations" not in body


def test_predict_passes_activations(make_client):
    acts = {"conv1": [[[0.0, 1.0], [2.0, 3.0]]], "dense": [0.1, 0.2]}
    with make_client(FixedPredictor(0.1, acts)) as client:
        body = client.post("/api/predict", files={"file": ("a.png", png(), "image/png")}).json()
    assert body["label"] == "dog"
    assert body["activations"] == acts


def test_rejects_non_image(make_client):
    with make_client(FixedPredictor(0.5)) as client:
        res = client.post("/api/predict", files={"file": ("a.txt", b"hello", "text/plain")})
    assert res.status_code == 415
    assert "detail" in res.json()


def test_rejects_too_large(make_client):
    with make_client(FixedPredictor(0.5), max_upload_bytes=100) as client:
        res = client.post("/api/predict", files={"file": ("a.png", png(size=(200, 200)), "image/png")})
    assert res.status_code == 413


def test_serves_frontend(make_client, tmp_path):
    (tmp_path / "index.html").write_text("<p>front</p>")
    with make_client(FixedPredictor(0.5), static_dir=tmp_path) as client:
        assert "front" in client.get("/").text
        assert client.get("/api/health").status_code == 200
