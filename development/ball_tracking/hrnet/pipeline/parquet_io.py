from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

COLUMNS = ["frame_idx", "t", "has_ball", "xc", "yc", "conf"]


def empty_frame_df() -> pd.DataFrame:
    return pd.DataFrame(columns=COLUMNS)


def load_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return empty_frame_df()
    return pd.read_parquet(path)


def append_records(path: Path, records: Iterable[dict]) -> None:
    new_df = pd.DataFrame.from_records(list(records), columns=COLUMNS)
    if new_df.empty:
        return
    if path.exists():
        existing = pd.read_parquet(path)
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(path, index=False)
