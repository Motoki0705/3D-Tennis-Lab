from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd

from .base import Clip, ClipFrame


@dataclass
class ContiguousConfig:
    max_gap_frames: int
    min_clip_len: int


class ContiguousClipExtractor:
    def __init__(self, config: ContiguousConfig):
        self.config = config

    def extract(self, df_frames: pd.DataFrame) -> List[Clip]:
        has_ball_df = df_frames[df_frames["has_ball"]].sort_values("frame_idx")
        clips: List[Clip] = []
        if has_ball_df.empty:
            return clips

        current_frames: List[ClipFrame] = []
        last_frame_idx: int | None = None
        clip_index = 1

        for row in has_ball_df.itertuples(index=False):
            frame_idx = int(row.frame_idx)
            if last_frame_idx is None:
                current_frames = [
                    ClipFrame(
                        frame_idx=frame_idx,
                        t=float(row.t),
                        xc=float(row.xc),
                        yc=float(row.yc),
                        conf=float(row.conf),
                    )
                ]
                last_frame_idx = frame_idx
                continue

            gap = frame_idx - last_frame_idx
            if gap - 1 > self.config.max_gap_frames:
                if len(current_frames) >= self.config.min_clip_len:
                    clips.append(Clip(clip_index=clip_index, frames=current_frames))
                    clip_index += 1
                current_frames = []

            current_frames.append(
                ClipFrame(
                    frame_idx=frame_idx,
                    t=float(row.t),
                    xc=float(row.xc),
                    yc=float(row.yc),
                    conf=float(row.conf),
                )
            )
            last_frame_idx = frame_idx

        if len(current_frames) >= self.config.min_clip_len:
            clips.append(Clip(clip_index=clip_index, frames=current_frames))

        return clips
