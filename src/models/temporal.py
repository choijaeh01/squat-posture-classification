"""
Model architectures for squat posture classification.

The default network combines a lightweight 1D CNN feature extractor followed
by a GRU head. The implementation is intentionally simple so it can serve as
a starting point for experiments on embedded hardware.
"""

from __future__ import annotations

import torch
from torch import nn


class TemporalCNNGRU(nn.Module):
    """Two-layer 1D CNN encoder with a GRU classification head."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        conv_channels: int = 64,
        gru_hidden_size: int = 128,
        gru_num_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, conv_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(conv_channels, conv_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.gru = nn.GRU(
            input_size=conv_channels,
            hidden_size=gru_hidden_size,
            num_layers=gru_num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if gru_num_layers > 1 else 0.0,
        )

        self.head = nn.Sequential(
            nn.LayerNorm(gru_hidden_size * 2),
            nn.Linear(gru_hidden_size * 2, gru_hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(gru_hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor shaped `(batch, channels, timesteps)`.
        Returns:
            Logits shaped `(batch, num_classes)`.
        """

        features = self.encoder(x)  # (batch, conv_channels, time)
        sequence = features.transpose(1, 2)  # (batch, time, conv_channels)
        output, _ = self.gru(sequence)
        pooled = output.mean(dim=1)
        return self.head(pooled)
