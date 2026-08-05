from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


class ResidualTCNBlock(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        padding = 2 * dilation
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=5, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation),
            nn.BatchNorm1d(channels),
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.net(x))


class RadarEncoder(nn.Module):
    def __init__(self, in_channels: int, embedding_dim: int = 192) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 96, kernel_size=7, padding=3),
            nn.BatchNorm1d(96),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            ResidualTCNBlock(96, 1),
            ResidualTCNBlock(96, 2),
            ResidualTCNBlock(96, 4),
        )
        self.projection = nn.Sequential(
            nn.Linear(192, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks(self.stem(x))
        pooled = torch.cat([x.mean(dim=-1), x.amax(dim=-1)], dim=-1)
        return self.projection(pooled)


class TemporalAttention(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=dimension,
            nhead=4,
            dim_feedforward=dimension * 3,
            dropout=0.10,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.score = nn.Linear(dimension, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        weights = torch.softmax(self.score(x).squeeze(-1), dim=-1)
        return torch.sum(x * weights.unsqueeze(-1), dim=1)


class InfraredEncoder(nn.Module):
    def __init__(self, embedding_dim: int = 192, *, pretrained: bool) -> None:
        super().__init__()
        weights = None
        if pretrained:
            try:
                weights = MobileNet_V3_Small_Weights.DEFAULT
            except Exception as exc:
                warnings.warn(f"Could not select pretrained MobileNet weights: {exc}")
        try:
            backbone = mobilenet_v3_small(weights=weights)
        except Exception as exc:
            warnings.warn(
                "Could not download/load pretrained MobileNet weights; using random initialization. "
                f"Reason: {exc}"
            )
            backbone = mobilenet_v3_small(weights=None)
        self.backbone = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.frame_projection = nn.Sequential(
            nn.Linear(576, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
        )
        self.temporal = TemporalAttention(embedding_dim)

    def set_backbone_trainable(self, trainable: bool) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = trainable

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, frames, channels, height, width = x.shape
        x = x.reshape(batch * frames, channels, height, width)
        x = self.pool(self.backbone(x)).flatten(1)
        x = self.frame_projection(x).reshape(batch, frames, -1)
        return self.temporal(x)


class MultiModalClassifier(nn.Module):
    def __init__(
        self,
        radar_channels: int,
        num_classes: int,
        *,
        pretrained_vision: bool,
        embedding_dim: int = 192,
    ) -> None:
        super().__init__()
        self.radar = RadarEncoder(radar_channels, embedding_dim)
        self.infrared = InfraredEncoder(embedding_dim, pretrained=pretrained_vision)
        self.gate = nn.Sequential(
            nn.Linear(embedding_dim * 2 + 4, 96),
            nn.GELU(),
            nn.Linear(96, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim * 2 + 4, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(256, num_classes),
        )

    def forward(
        self,
        radar: torch.Tensor,
        infrared: torch.Tensor,
        quality: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        radar_embedding = self.radar(radar)
        infrared_embedding = self.infrared(infrared)
        gate_logits = self.gate(torch.cat([radar_embedding, infrared_embedding, quality], dim=-1))
        gates = torch.softmax(gate_logits, dim=-1)
        fused = gates[:, :1] * radar_embedding + gates[:, 1:] * infrared_embedding
        disagreement = torch.abs(radar_embedding - infrared_embedding)
        logits = self.classifier(torch.cat([fused, disagreement, quality], dim=-1))
        return {"logits": logits, "gates": gates}


@dataclass(frozen=True)
class CheckpointMetadata:
    classes: list[str]
    radar_schema: list[str]
    config: dict[str, Any]
    model_version: str = "2.0.0"


def build_model(
    config: dict[str, Any],
    *,
    num_classes: int,
    pretrained: bool,
) -> MultiModalClassifier:
    return MultiModalClassifier(
        int(config["radar_channels"]),
        num_classes,
        pretrained_vision=pretrained,
    )
