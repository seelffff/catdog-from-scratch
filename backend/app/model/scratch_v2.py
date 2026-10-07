"""Serve the exact frozen V2 CNN or logit ensemble on CPU.

The optional conv1/relu1/pool1 maps are real stem activations from the first
ensemble member, on the unflipped pass. They show at most four channels,
spatially averaged to at most 32x32 for transport. BatchNorm lies between
conv1 and relu1; no later layer or whole-ensemble activation is invented.
"""

from __future__ import annotations

import hashlib
import io
import math
from pathlib import Path
import re
from threading import Lock

import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

from ._scratch_architecture import LogitEnsemble, build_model
from ._scratch_preprocessing import preprocess_image
from .base import PredictionResult

# This project identity prevents an old V1 checkpoint being called scratch-v2.
V2_MANIFEST_SHA256 = "334e10c5e72f18c4cf9c11e4bd64eed3feb0bdf34a7594803ca955f0c0fb992e"
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
OWN_INITIALIZATIONS = frozenset({"random", "continuation_of_own_scratch_training", "own_model_ensemble"})


def validate_checkpoint(checkpoint: dict) -> None:
    config = checkpoint.get("train_config", {})
    if (checkpoint.get("pretrained") is not False or config.get("pretrained") is not False
            or config.get("manifest_sha256") != V2_MANIFEST_SHA256):
        raise ValueError("Only own scratch weights trained on the audited V2 manifest are supported")
    initialization = config.get("initialization")
    if initialization not in OWN_INITIALIZATIONS:
        raise ValueError("Own scratch initialization provenance is required")
    if initialization != "own_model_ensemble" and checkpoint.get("initialization") != initialization:
        raise ValueError("Checkpoint initialization conflicts with its training provenance")
    if initialization == "continuation_of_own_scratch_training" and not SHA256_PATTERN.fullmatch(
        str(config.get("initialization_checkpoint_sha256", ""))
    ):
        raise ValueError("Own scratch continuation must identify its source checkpoint SHA-256")
    if (checkpoint.get("selection_frozen_before_test") is not True
            or checkpoint.get("candidate_validation_pending") is True):
        raise ValueError("A final checkpoint frozen before test is required")
    if checkpoint.get("class_names") != ["cat", "dog"] or checkpoint.get("resize_mode") != "letterbox":
        raise ValueError("Expected saved class order cat/dog and letterbox preprocessing")
    if type(checkpoint.get("image_size")) is not int or not 64 <= checkpoint["image_size"] <= 1024:
        raise ValueError("Checkpoint image_size must be an integer in [64, 1024]")
    normalization = checkpoint.get("normalization", {})
    for key in ("mean", "std"):
        values = normalization.get(key, ())
        if len(values) != 3 or any(type(value) not in (int, float) or not math.isfinite(value) for value in values):
            raise ValueError("Checkpoint normalization must contain three finite RGB values")
    if any(value <= 0 for value in normalization["std"]):
        raise ValueError("Checkpoint normalization standard deviations must be positive")
    temperature = checkpoint.get("temperature")
    if type(temperature) not in (int, float) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Checkpoint temperature must be finite and positive")
    if checkpoint.get("threshold") != 0.5:
        raise ValueError("The fixed API contract requires the final 0.5 classification threshold")
    if type(checkpoint.get("tta_horizontal_flip")) is not bool:
        raise ValueError("Checkpoint must explicitly record horizontal flip TTA")
    if not isinstance(checkpoint.get("model_config"), dict) or "state_dict" not in checkpoint:
        raise ValueError("Saved architecture and state_dict are required")


class ScratchPredictor:
    name = "scratch-v2"
    activation_source = "first member stem, unflipped input; first four channels, average-pooled to <=32x32"

    def __init__(self, weights: Path, expected_sha256: str | None = None, collect_activations: bool = True):
        self.checkpoint_path = Path(weights).expanduser().resolve()
        # Hash and deserialize one immutable snapshot, never two reads of changing weights.
        payload = self.checkpoint_path.read_bytes()
        self.checkpoint_sha256 = hashlib.sha256(payload).hexdigest()
        if expected_sha256 is not None:
            if not SHA256_PATTERN.fullmatch(expected_sha256) or expected_sha256 != self.checkpoint_sha256:
                raise ValueError("MODEL_SHA256 differs from the model byte snapshot")
        checkpoint = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
        validate_checkpoint(checkpoint)
        torch.set_num_threads(1)
        self.model_config = dict(checkpoint["model_config"])
        self.image_size = checkpoint["image_size"]
        self.normalization = {key: list(checkpoint["normalization"][key]) for key in ("mean", "std")}
        self.temperature = float(checkpoint["temperature"])
        self.tta_horizontal_flip = checkpoint["tta_horizontal_flip"]
        self.model = build_model(**self.model_config).cpu().eval()
        self.model.load_state_dict(checkpoint["state_dict"], strict=True)
        self._lock = Lock()
        self._collect_activations = collect_activations
        self._capture_unflipped = False
        self._maps: dict[str, list[list[list[float]]]] = {}
        self._hook_handles = []
        if collect_activations:
            member = self.model.members[0] if isinstance(self.model, LogitEnsemble) else self.model
            first_conv = next(module for module in member.stem if isinstance(module, nn.Conv2d))
            first_relu = next(module for module in member.stem if isinstance(module, nn.ReLU))
            first_pool = next(module for module in member.stem if isinstance(module, nn.MaxPool2d))
            for key, layer in (("conv1", first_conv), ("relu1", first_relu), ("pool1", first_pool)):
                self._hook_handles.append(layer.register_forward_hook(self._capture(key)))

    def _capture(self, key):
        def hook(_module, _inputs, output):
            if not self._capture_unflipped:
                return
            # Clone immediately: later in-place ReLU must not alter this snapshot.
            snapshot = output.detach()[:1, :4].clone()
            if not torch.isfinite(snapshot).all().item():
                raise ValueError("The model produced non-finite feature maps")
            height, width = snapshot.shape[-2:]
            if height > 32 or width > 32:
                snapshot = F.adaptive_avg_pool2d(snapshot, (min(height, 32), min(width, 32)))
            self._maps[key] = snapshot[0].tolist()
        return hook

    @staticmethod
    def _probabilities(logits: torch.Tensor, temperature: float) -> torch.Tensor:
        if logits.shape != (1, 2) or not torch.isfinite(logits).all().item():
            raise ValueError("The model must produce two finite class logits")
        probabilities = torch.softmax(logits / temperature, dim=1)
        if not torch.isfinite(probabilities).all().item():
            raise ValueError("The model produced non-finite probabilities")
        return probabilities

    @torch.inference_mode()
    def predict(self, image: Image.Image) -> PredictionResult:
        # Hooks, preprocessing and both TTA passes form one serialized transaction.
        with self._lock:
            self._maps = {}
            self._capture_unflipped = self._collect_activations
            try:
                tensor = preprocess_image(image, self.image_size, self.normalization).unsqueeze(0).cpu()
                probabilities = self._probabilities(self.model(tensor), self.temperature)
                self._capture_unflipped = False
                if self.tta_horizontal_flip:
                    flipped = self._probabilities(self.model(torch.flip(tensor, dims=(3,))), self.temperature)
                    probabilities = (probabilities + flipped) / 2
                p_cat = float(probabilities[0, 0].item())
                if not math.isfinite(p_cat) or not 0 <= p_cat <= 1:
                    raise ValueError("The model produced an invalid cat probability")
                return PredictionResult(p_cat=p_cat, activations=dict(self._maps) if self._maps else None,
                                        p_dog=float(probabilities[0, 1].item()))
            finally:
                self._capture_unflipped = False
                self._maps = {}
