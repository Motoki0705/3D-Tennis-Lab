from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd

from .base import Clip, ClipFrame


@dataclass
class ClusteringConfig:
    algo: str
    features: list[str]
    eps: float | None = None
    min_samples: int | None = None
    min_clip_len: int = 1


class ClusteringClipExtractor:
    def __init__(self, config: ClusteringConfig):
        self.config = config

    def extract(self, df_frames: pd.DataFrame) -> List[Clip]:
        data = df_frames[df_frames["has_ball"]]
        if data.empty:
            return []

        features = self.config.features
        missing = [f for f in features if f not in data.columns]
        if missing:
            raise ValueError(f"Missing features for clustering: {missing}")

        labels = self._run_clustering(data[features])
        data = data.assign(cluster=labels)
        clips: List[Clip] = []
        clip_index = 1
        for cluster_id, cluster_df in data.groupby("cluster"):
            if cluster_id == -1:
                continue  # noise
            if len(cluster_df) < self.config.min_clip_len:
                continue
            sorted_df = cluster_df.sort_values("frame_idx")
            frames = [
                ClipFrame(
                    frame_idx=int(row.frame_idx),
                    t=float(row.t),
                    xc=float(row.xc),
                    yc=float(row.yc),
                    conf=float(row.conf),
                )
                for row in sorted_df.itertuples(index=False)
            ]
            clips.append(Clip(clip_index=clip_index, frames=frames))
            clip_index += 1
        return clips

    def _run_clustering(self, feature_df: pd.DataFrame):
        algo = self.config.algo.lower()
        if algo == "dbscan":
            try:
                from sklearn.cluster import DBSCAN
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise RuntimeError("DBSCAN clustering requires scikit-learn to be installed") from exc
            model = DBSCAN(eps=self.config.eps or 0.8, min_samples=self.config.min_samples or 5)
            return model.fit_predict(feature_df.values)
        else:
            raise NotImplementedError(f"Clustering algorithm '{self.config.algo}' is not supported yet")
