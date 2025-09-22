from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import pandas as pd


@dataclass
class ClipFrame:
    frame_idx: int
    t: float
    xc: float
    yc: float
    conf: float


@dataclass
class Clip:
    clip_index: int
    frames: List[ClipFrame]

    @property
    def start_frame(self) -> int:
        return self.frames[0].frame_idx

    @property
    def end_frame(self) -> int:
        return self.frames[-1].frame_idx

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({
            "frame_idx": [f.frame_idx for f in self.frames],
            "t": [f.t for f in self.frames],
            "xc": [f.xc for f in self.frames],
            "yc": [f.yc for f in self.frames],
            "conf": [f.conf for f in self.frames],
        })


class ClipExtractor:
    def extract(self, df_frames: pd.DataFrame) -> List[Clip]:  # pragma: no cover - interface
        raise NotImplementedError


def dataframe_to_frames(df: pd.DataFrame, indices: Iterable[int]) -> List[ClipFrame]:
    rows = df.loc[list(indices)]
    return [
        ClipFrame(
            frame_idx=int(row.frame_idx),
            t=float(row.t),
            xc=float(row.xc),
            yc=float(row.yc),
            conf=float(row.conf),
        )
        for row in rows.itertuples(index=False)
    ]
