"""Scratch residual CNNs and optional ensembles; no pretrained weights are fetched."""

import math
from collections.abc import Sequence

import torch
from torch import nn


class BasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.skip = (
            nn.Identity()
            if stride == 1 and in_channels == out_channels
            else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.relu(x + residual)


class TinyResNet(nn.Module):
    def __init__(self, num_classes: int = 2, width: int = 32, dropout: float = 0.15):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, width, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        stages = []
        in_channels = width
        for stage_index, out_channels in enumerate([width, width * 2, width * 4, width * 8]):
            stages.extend([
                BasicBlock(in_channels, out_channels, stride=1 if stage_index == 0 else 2),
                BasicBlock(out_channels, out_channels),
            ])
            in_channels = out_channels
        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(in_channels, num_classes)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.01)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stages(self.stem(x))
        x = self.pool(x).flatten(1)
        return self.classifier(self.dropout(x))


class SqueezeExcitation(nn.Module):
    """Learn channel gates from the spatial average of a residual branch."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gate(x)


class RobustBasicBlock(nn.Module):
    """Residual block with channel attention and an average-pooled skip."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.attention = SqueezeExcitation(out_channels)
        if stride == 1 and in_channels == out_channels:
            self.skip = nn.Identity()
        else:
            layers = []
            if stride == 2:
                # Ceil mode matches the 3x3 stride-2 branch on odd input sizes.
                layers.append(nn.AvgPool2d(2, stride=2, ceil_mode=True,
                                          count_include_pad=False))
            layers.extend([
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            ])
            self.skip = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.attention(self.bn2(self.conv2(x)))
        return self.relu(x + residual)


class RobustResNet(nn.Module):
    """A separate scratch architecture; TinyResNet checkpoints remain unchanged."""

    def __init__(self, num_classes: int = 2, width: int = 32, dropout: float = 0.15):
        super().__init__()
        stem_width = width // 2
        self.stem = nn.Sequential(
            nn.Conv2d(3, stem_width, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(stem_width, stem_width, 3, padding=1, bias=False),
            nn.BatchNorm2d(stem_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(stem_width, width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        stages = []
        in_channels = width
        for stage_index, out_channels in enumerate([width, width * 2, width * 4, width * 8]):
            stages.extend([
                RobustBasicBlock(in_channels, out_channels, stride=1 if stage_index == 0 else 2),
                RobustBasicBlock(out_channels, out_channels),
            ])
            in_channels = out_channels
        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(in_channels, num_classes)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.01)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stages(self.stem(x))
        x = self.pool(x).flatten(1)
        return self.classifier(self.dropout(x))


class LogitEnsemble(nn.Module):
    """Combine two or three independent scratch networks at the same input size."""

    def __init__(self, members: Sequence[nn.Module], mixture_weights: Sequence[float] | None = None):
        super().__init__()
        if not 2 <= len(members) <= 3 or any(isinstance(member, LogitEnsemble) for member in members):
            raise ValueError("An ensemble must contain two or three non-ensemble networks")
        weights = list(mixture_weights) if mixture_weights is not None else [1.0] * len(members)
        if len(weights) != len(members) or any(not math.isfinite(float(value)) or float(value) <= 0 for value in weights):
            raise ValueError("Every ensemble member requires a finite positive mixture weight")
        # Normalize before creating float32 values to avoid overflowing large weights.
        largest = max(weights)
        scaled = [float(value) / largest for value in weights]
        total = sum(scaled)
        normalized = torch.tensor([value / total for value in scaled], dtype=torch.float32)
        if not torch.all(normalized > 0):
            raise ValueError("Mixture weights differ too much to represent in float32")
        self.members = nn.ModuleList(members)
        self.register_buffer("mixture_weights", normalized)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.members[0](x) * self.mixture_weights[0]
        for index in range(1, len(self.members)):
            result = result + self.members[index](x) * self.mixture_weights[index]
        return result


def build_model(name: str = "tiny_resnet18", num_classes: int = 2,
                width: int = 32, dropout: float = 0.15,
                members: Sequence[dict] | None = None,
                mixture_weights: Sequence[float] | None = None) -> nn.Module:
    if name == "logit_ensemble":
        if num_classes != 2 or members is None or not 2 <= len(members) <= 3:
            raise ValueError("Expected two classes and two or three ensemble member configurations")
        if any(not isinstance(config, dict) or config.get("name", "tiny_resnet18") == "logit_ensemble" for config in members):
            raise ValueError("Nested ensembles are not supported")
        return LogitEnsemble([build_model(**config) for config in members], mixture_weights)
    if name not in {"tiny_resnet18", "robust_resnet18"}:
        raise ValueError(f"Unknown architecture: {name}")
    if members is not None or mixture_weights is not None:
        raise ValueError("Ensemble settings require name='logit_ensemble'")
    if num_classes != 2 or width < 8 or not 0 <= dropout < 1:
        raise ValueError("Expected two classes, width >= 8, and dropout in [0, 1).")
    architecture = TinyResNet if name == "tiny_resnet18" else RobustResNet
    return architecture(num_classes=num_classes, width=width, dropout=dropout)
