"""Dataset exports for core sequential tasks."""

from .base_sequence import BaseSequenceDataset
from .ball import BallSequenceDataset
from .court import CourtKeypointDataset
from .player import PlayerSequenceDataset

# Backwards-compatible aliases (deprecated).
BaseBallDataset = BallSequenceDataset
BallClipDataset = BallSequenceDataset
BaseCourtKeypointDataset = CourtKeypointDataset
BasePlayerDetectionDataset = PlayerSequenceDataset

__all__ = [
    "BallClipDataset",
    "BallSequenceDataset",
    "BaseBallDataset",
    "BaseCourtKeypointDataset",
    "BasePlayerDetectionDataset",
    "BaseSequenceDataset",
    "CourtKeypointDataset",
    "PlayerSequenceDataset",
]
