from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from omegaconf import DictConfig, OmegaConf

from .annotation import (
    ExportPaths,
    VideoFrameExtractor,
    determine_next_clip_index,
    export_clip,
    iter_clip_json,
    load_clip_json,
    merge_accepted_clips,
    save_clip_json,
)
from .clip_extractor import Clip, build_clip_extractor
from .inference import InferenceEngine
from .parquet_io import load_parquet
from .scanner import scan_new_videos
from .state import StateRepository

LOG = logging.getLogger(__name__)


class AnnotationPipeline:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        paths_cfg = cfg["paths"]
        self.videos_root = Path(paths_cfg["videos_root"])
        self.export_paths = ExportPaths(
            images_root=Path(paths_cfg["images_dir"]),
            ann_clips_root=Path(paths_cfg["ann_clips_dir"]),
        )
        self.final_annotations = Path(paths_cfg["ann_final"])
        self.repo = StateRepository(Path(paths_cfg["state_db"]))

        detection_cfg = self._compose_detection_cfg(cfg)
        self.inference_engine = InferenceEngine(detection_cfg)
        self.clip_extractor = build_clip_extractor(cfg["clip_extractor"])

    # ------------------------------------------------------------------
    def run(self) -> None:
        LOG.info("Scanning for new videos under %s", self.videos_root)
        scan_new_videos(self.videos_root, self.repo)

        pending = self.repo.list_pending()
        if not pending:
            LOG.info("No pending videos to process")
            return

        for record in pending:
            video_path = Path(record.video_path)
            if not video_path.exists():
                LOG.warning("Video missing on disk: %s", video_path)
                continue

            LOG.info("Processing video %s (status=%s)", video_path, record.status)
            start_frame = max(0, record.last_processed_frame + 1)
            parquet_path = Path(self.cfg["paths"]["inference_dir"]) / f"{record.video_id}.parquet"

            # Run inference if needed
            if record.status != "done" or start_frame == 0:
                try:
                    self.inference_engine.run(
                        video_path,
                        parquet_path,
                        self.repo,
                        record.video_id,
                        start_frame=start_frame,
                        save_every_n_frames=int(self.cfg["inference"]["save_every_n_frames"]),
                    )
                except Exception:
                    LOG.exception("Inference failed for %s", video_path)
                    self.repo.update_progress(record.video_id, status="failed")
                    continue

            df = load_parquet(parquet_path)
            if df.empty:
                LOG.info("No inference results for %s", video_path)
                continue

            clips = self.clip_extractor.extract(df)
            if not clips:
                LOG.info("No clips extracted for %s", video_path)
                continue

            self._export_clips_for_video(video_path, record.video_id, clips)

    # ------------------------------------------------------------------
    def accept(self, clip_ref: str) -> None:
        clip_path = self._clip_json_path(clip_ref)
        data = load_clip_json(clip_path)
        data["accepted"] = True
        save_clip_json(clip_path, data)
        LOG.info("Marked clip %s as accepted", clip_ref)

    def reject(self, clip_ref: str) -> None:
        clip_path = self._clip_json_path(clip_ref)
        data = load_clip_json(clip_path)
        data["accepted"] = False
        save_clip_json(clip_path, data)
        LOG.info("Marked clip %s as rejected", clip_ref)

    def finalize(self) -> None:
        clip_paths = iter_clip_json(self.export_paths.ann_clips_root)
        merge_accepted_clips(clip_paths, self.final_annotations)

    # ------------------------------------------------------------------
    def _clip_json_path(self, clip_ref: str) -> Path:
        clip_ref = clip_ref.strip().rstrip(".json")
        path = self.export_paths.ann_clips_root / clip_ref
        if not path.suffix:
            path = path.with_suffix(".json")
        if not path.exists():
            raise FileNotFoundError(f"Clip annotation not found: {path}")
        return path

    def _export_clips_for_video(self, video_path: Path, video_id: str, clips: List[Clip]) -> None:
        game_id = self._derive_game_id(video_path)
        existing_meta = self._load_existing_clip_metadata(game_id)
        next_index = determine_next_clip_index(self.export_paths.images_root / game_id)

        new_clips = [clip for clip in clips if self._clip_is_new(video_id, clip, existing_meta)]
        if not new_clips:
            LOG.info("No new clips to export for %s", video_path)
            return

        reader = VideoFrameExtractor(video_path)
        try:
            for clip in new_clips:
                ann_path = export_clip(
                    clip,
                    clip_index=next_index,
                    game_id=game_id,
                    video_id=video_id,
                    reader=reader,
                    export_paths=self.export_paths,
                    jpg_quality=int(self.cfg["export"]["jpg_quality"]),
                    write_frames=bool(self.cfg["export"]["write_frames"]),
                )
                existing_meta.add((video_id, clip.start_frame, clip.end_frame))
                next_index += 1
                LOG.debug("Exported clip annotation to %s", ann_path)
        finally:
            reader.release()

    def _derive_game_id(self, video_path: Path) -> str:
        try:
            rel = video_path.relative_to(self.videos_root)
        except ValueError:
            return video_path.stem
        if rel.parent == Path("."):
            return rel.stem
        return rel.parts[0]

    def _load_existing_clip_metadata(self, game_id: str) -> set[tuple[str, int, int]]:
        game_dir = self.export_paths.ann_clips_root / game_id
        metadata: set[tuple[str, int, int]] = set()
        if not game_dir.exists():
            return metadata
        for path in game_dir.glob("Clip*.json"):
            data = load_clip_json(path)
            meta = data.get("metadata") or {}
            if {
                "video_id",
                "start_frame",
                "end_frame",
            }.issubset(meta.keys()):
                metadata.add((meta["video_id"], int(meta["start_frame"]), int(meta["end_frame"])))
        return metadata

    def _clip_is_new(self, video_id: str, clip: Clip, existing: set[tuple[str, int, int]]) -> bool:
        key = (video_id, clip.start_frame, clip.end_frame)
        return key not in existing

    @staticmethod
    def _compose_detection_cfg(cfg: DictConfig) -> DictConfig:
        groups = ("runner", "model", "detector", "transform", "tracker", "dataloader")
        detection_cfg = OmegaConf.create()
        for name in groups:
            if name not in cfg:
                raise KeyError(f"Missing detection config group: {name}")
            detection_cfg[name] = OmegaConf.create(OmegaConf.to_container(cfg[name], resolve=True))
        return detection_cfg
