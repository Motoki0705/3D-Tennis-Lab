# Step 8.2: Main application entry point using Hydra
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
import yaml
from pathlib import Path
import logging

from ..core.types import CourtSpec
from ..io.video import VideoReader
from ..ui.loop import run_ui_loop
from ..pipeline.runner import PipelineRunner
from ..pipeline.source import SourceStage
from ..pipeline.state_fetch import StateFetchStage
from ..pipeline.control.stage import ControlStage
from ..pipeline.fit import FitStage
from ..pipeline.validate import ValidateStage
from ..pipeline.render import RenderStage
from ..pipeline.persist import PersistStage
from ..pipeline.export import ExportStage
from .bootstrap import setup_logging
from .discover import discover_videos, sort_videos
from ..io.resume_manager import load_resume, save_resume, ensure_video, update_video_progress
from ..io.meta import update_video_meta


@hydra.main(version_base=None, config_path="../configs", config_name="annotator")
def main(cfg: DictConfig):
    """Main function orchestrated by Hydra."""
    # --- 0. Logging ---
    out_dir_abs = Path(to_absolute_path(cfg.out_dir))
    # Unified log file in aggregated out_dir
    log_file = out_dir_abs / cfg.logging.file
    setup_logging(level=str(cfg.logging.level), file_path=str(log_file))

    # --- 1. Prepare aggregated outputs + resume ---
    out_dir_abs.mkdir(parents=True, exist_ok=True)
    resume_state = load_resume(out_dir_abs)

    court_spec_path = to_absolute_path(cfg.court)
    with open(court_spec_path, "r", encoding="utf-8") as f:
        court_data = yaml.safe_load(f)
    spec_data = {
        "names": court_data["names"],
        "skeleton": court_data["skeleton"],
        "template_xy": court_data["template_xy"],
    }
    court_spec = CourtSpec(**spec_data)
    logging.info(f"Loaded court spec: {cfg.court}")

    # --- Load other configs and merge them ---
    thresholds_path = to_absolute_path(cfg.thresholds)
    with open(thresholds_path, "r", encoding="utf-8") as f:
        thresholds_cfg = yaml.safe_load(f)

    ui_path = to_absolute_path(cfg.ui)
    with open(ui_path, "r", encoding="utf-8") as f:
        ui_cfg = yaml.safe_load(f)

    # Convert main cfg to dict and merge sub-configs
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg_dict["thresholds"] = thresholds_cfg
    cfg_dict["ui"] = ui_cfg

    # --- 2. Initialize Pipeline ---
    logging.info("Initializing pipeline stages...")
    stages = [
        SourceStage(),
        StateFetchStage(),
        ControlStage(),
        FitStage(court_spec, cfg.completion_strategy, thresholds_cfg),
        ValidateStage(thresholds_cfg),
        RenderStage(),
        PersistStage(),
        ExportStage(),
    ]
    runner = PipelineRunner(stages)
    logging.info(f"Pipeline initialized with {len(stages)} stages.")

    # --- 3. Single video (legacy) or multi-video ---
    if hasattr(cfg, "video_dir") and cfg.video_dir:
        # Multi-video aggregated mode
        exts = list(cfg.get("discover", {}).get("exts", [".mp4", ".mov", ".mkv", ".avi"]))
        sort_mode = str(cfg.get("discover", {}).get("sort", "name"))
        video_dir_abs = to_absolute_path(cfg.video_dir)

        while True:
            # Discover and register new videos
            paths = sort_videos(discover_videos(video_dir_abs, exts), sort_mode)
            existing_paths = set(resume_state.get("videos", {}).keys())
            for p in paths:
                sp = str(p)
                if sp not in existing_paths:
                    ensure_video(resume_state, sp)
                    # Update meta for new videos
                    try:
                        vinfo = VideoReader(to_absolute_path(sp))
                        update_video_meta(
                            out_dir_abs,
                            int(resume_state["videos"][sp]["video_id"]),
                            sp,
                            vinfo.fps,
                            vinfo.nframes,
                            vinfo.w,
                            vinfo.h,
                        )
                        vinfo.release()
                    except Exception as e:
                        logging.warning(f"Failed to probe video meta for {sp}: {e}")
            save_resume(out_dir_abs, resume_state)

            # Pick next unfinished
            next_path = None
            for p in paths:
                v = resume_state["videos"].get(str(p))
                if v and not v.get("done", False):
                    next_path = str(p)
                    break
            if not next_path:
                logging.info("No unfinished videos. All done.")
                break

            # Open and process this video
            video = VideoReader(to_absolute_path(next_path))
            logging.info(f"Opened video: {next_path}")
            logging.info(f"FPS: {video.fps}, Frames: {video.nframes}, Size: {video.w}x{video.h}")
            # Update meta.json
            update_video_meta(
                out_dir_abs,
                int(resume_state["videos"][next_path]["video_id"]),
                next_path,
                video.fps,
                video.nframes,
                video.w,
                video.h,
            )

            # Update start from resume
            start_idx = int(resume_state["videos"][next_path].get("next_frame", 0))
            cfg_dict["start"] = start_idx

            # Pass resume + video_id through to the loop
            video_id = int(resume_state["videos"][next_path]["video_id"])

            logging.info("Starting interactive UI loop for current video...")
            result = run_ui_loop(cfg_dict, runner, court_spec, video, video_id=video_id, resume_state=resume_state)
            last_idx = int(result.get("last_frame_idx", start_idx))
            stopped_by_quit = bool(result.get("stopped_by_quit", False))
            stopped_by_done = bool(result.get("stopped_by_done", False))

            # Update progress
            # If user quit, keep done=False and next_frame=last_idx; if ended, mark done
            if stopped_by_quit:
                update_video_progress(resume_state, next_path, next_frame=last_idx)
            elif stopped_by_done:
                update_video_progress(resume_state, next_path, next_frame=last_idx, done=True)
            else:
                # Consider done if UI loop reached beyond last frame
                done_flag = last_idx >= (video.nframes - 1)
                update_video_progress(resume_state, next_path, next_frame=last_idx, done=done_flag)
            save_resume(out_dir_abs, resume_state)

            video.release()

            if stopped_by_quit:
                logging.info("Stopped by user request. Exiting.")
                break

            # Re-scan to pick up any newly added videos before next iteration
            logging.info("Rescanning for new videos...")

        logging.info("Application finished (multi-video mode).")
    else:
        # Legacy single-video mode
        video_path_abs = to_absolute_path(cfg.video)
        video = VideoReader(video_path_abs)
        logging.info(f"Opened video: {cfg.video}")
        logging.info(f"FPS: {video.fps}, Frames: {video.nframes}, Size: {video.w}x{video.h}")
        if "interactive" in cfg.preset:
            logging.info("Starting interactive UI loop...")
            run_ui_loop(cfg_dict, runner, court_spec, video, video_id=None, resume_state=resume_state)
        else:
            logging.info("Running in headless mode (not implemented yet)...")
        video.release()
        logging.info("Application finished (single-video mode).")


if __name__ == "__main__":
    main()
