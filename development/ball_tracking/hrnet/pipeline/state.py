from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional


@dataclass
class VideoRecord:
    video_id: str
    video_path: str
    status: str
    last_processed_frame: int
    total_frames: Optional[int]


class StateRepository:
    """SQLite-backed repository tracking video processing progress."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS videos (
                    video_id TEXT PRIMARY KEY,
                    video_path TEXT UNIQUE NOT NULL,
                    status TEXT NOT NULL,
                    last_processed_frame INTEGER NOT NULL,
                    total_frames INTEGER,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_videos_status
                ON videos(status)
                """
            )

    @staticmethod
    def compute_video_id(video_path: Path) -> str:
        rel = str(video_path).encode("utf-8")
        return hashlib.sha1(rel).hexdigest()[:16]

    def register_video(self, video_path: Path) -> str:
        video_id = self.compute_video_id(video_path)
        now = datetime.utcnow().isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO videos (video_id, video_path, status, last_processed_frame, total_frames, updated_at)
                VALUES (?, ?, 'queued', -1, NULL, ?)
                """,
                (video_id, str(video_path), now),
            )
        return video_id

    def update_progress(
        self,
        video_id: str,
        *,
        last_processed_frame: Optional[int] = None,
        status: Optional[str] = None,
        total_frames: Optional[int] = None,
    ) -> None:
        fields: list[str] = []
        params: list[object] = []
        if last_processed_frame is not None:
            fields.append("last_processed_frame = ?")
            params.append(last_processed_frame)
        if status is not None:
            fields.append("status = ?")
            params.append(status)
        if total_frames is not None:
            fields.append("total_frames = ?")
            params.append(total_frames)
        fields.append("updated_at = ?")
        params.append(datetime.utcnow().isoformat())
        params.append(video_id)

        with self._connect() as conn:
            conn.execute(
                f"UPDATE videos SET {' , '.join(fields)} WHERE video_id = ?",
                params,
            )

    def list_pending(self) -> list[VideoRecord]:
        pending_status = ("queued", "running", "failed")
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT video_id, video_path, status, last_processed_frame, total_frames "
                "FROM videos WHERE status IN ({}) ORDER BY updated_at".format(",".join("?" for _ in pending_status)),
                pending_status,
            ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def list_all(self) -> Iterable[VideoRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT video_id, video_path, status, last_processed_frame, total_frames FROM videos"
            ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def get(self, video_id: str) -> Optional[VideoRecord]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT video_id, video_path, status, last_processed_frame, total_frames FROM videos WHERE video_id = ?",
                (video_id,),
            ).fetchone()
        return self._row_to_record(row) if row else None

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> VideoRecord:
        return VideoRecord(
            video_id=row["video_id"],
            video_path=row["video_path"],
            status=row["status"],
            last_processed_frame=row["last_processed_frame"],
            total_frames=row["total_frames"],
        )
