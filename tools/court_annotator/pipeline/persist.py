from .runner import Stage
from ..io.image_saver import save_once
from pathlib import Path
import logging
from ..app.discover import first_token


class PersistStage(Stage):
    """A pipeline stage to save the frame image upon first annotation."""

    name = "persist"

    def process(self, bundle):
        if not bundle.get("should_persist_frame", False):
            return bundle

        cfg = bundle["config"]
        frame_idx = bundle["frame_idx"]
        session_state = bundle["session_state"]

        if frame_idx in session_state["saved_frames"]:
            return bundle

        video_path = Path(bundle["video_path"])
        images_dir = Path(cfg["out_dir"]) / "images"
        # Compute collision-free file name: <first_token(stem)>_<frame_idx:06d>(-<video_id>).jpg
        token = first_token(video_path.stem)
        base = f"{token}_{frame_idx:06d}.jpg"
        file_name = base
        if (images_dir / file_name).exists():
            # Disambiguate with video_id
            video_id = int(bundle.get("video_id", 0) or 0)
            if video_id > 0:
                file_name = f"{token}_{frame_idx:06d}-{video_id}.jpg"
        file_name = save_once(images_dir, file_name, bundle["frame"])

        # Record that this frame has been saved
        session_state["saved_frames"].add(frame_idx)
        session_state["images_meta"][frame_idx] = {
            "file_name": file_name,
            "width": bundle["frame"].shape[1],
            "height": bundle["frame"].shape[0],
        }

        logging.info(f"Saved frame {frame_idx} to {images_dir / file_name}")
        # clear flag to avoid repeated checks/writes
        bundle["should_persist_frame"] = False
        return bundle
