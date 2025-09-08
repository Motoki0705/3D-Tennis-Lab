from __future__ import annotations

import torch
import torch.nn as nn


class SequencePredictor(nn.Module):
    """RNN-based sequence model (LSTM/GRU) for next-state prediction."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_layers: int,
        output_dim: int,
        rnn_type: str = "lstm",
        dropout: float = 0.0,
        bidirectional: bool = False,
    ):
        super().__init__()
        rnn_type = rnn_type.lower()
        rnn_cls = nn.LSTM if rnn_type == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        head_in = hidden_dim * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Linear(head_in, head_in),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(head_in, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        last = out[:, -1, :]
        return self.head(last)
