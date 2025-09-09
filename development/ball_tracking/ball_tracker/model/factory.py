from __future__ import annotations

from typing import Any

import torch.nn as nn

from .sequence_predictor import SequencePredictor
from .transformer import TransformerSequenceModel


def build_sequence_model(cfg: Any) -> nn.Module:
    mcfg = cfg.model
    model_type = getattr(mcfg, "type", "lstm").lower()
    if model_type == "lstm":
        return SequencePredictor(
            input_dim=mcfg.input_dim,
            hidden_dim=mcfg.hidden_dim,
            n_layers=mcfg.n_layers,
            output_dim=mcfg.output_dim,
            dropout=getattr(mcfg, "dropout", 0.0),
            bidirectional=getattr(mcfg, "bidirectional", False),
        )
    elif model_type == "gru":
        return SequencePredictor(
            input_dim=mcfg.input_dim,
            hidden_dim=mcfg.hidden_dim,
            n_layers=mcfg.n_layers,
            output_dim=mcfg.output_dim,
            rnn_type="gru",
            dropout=getattr(mcfg, "dropout", 0.0),
            bidirectional=getattr(mcfg, "bidirectional", False),
        )
    elif model_type == "transformer":
        return TransformerSequenceModel(
            input_dim=mcfg.input_dim,
            d_model=getattr(mcfg, "d_model", 128),
            nhead=getattr(mcfg, "nhead", 4),
            num_layers=mcfg.n_layers,
            dim_feedforward=getattr(mcfg, "dim_feedforward", 256),
            dropout=getattr(mcfg, "dropout", 0.1),
            output_dim=mcfg.output_dim,
            use_causal_mask=getattr(mcfg, "causal", True),
        )
    else:
        raise ValueError(f"Unsupported model.type: {model_type}")
