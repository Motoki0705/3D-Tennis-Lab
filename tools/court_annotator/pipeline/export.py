from .runner import Stage
from pathlib import Path
import logging
from ..io.ann_aggregator import append_or_update
from ..io.resume_manager import save_resume


class ExportStage(Stage):
    """A pipeline stage to export all annotations to COCO format."""

    name = "export"

    def process(self, bundle):
        if not bundle.get("should_export", False):
            return bundle

        cfg = bundle["config"]
        session_state = bundle["session_state"]
        video_path = Path(bundle["video_path"])
        out_dir = Path(cfg["out_dir"])  # aggregated output root

        images_meta = session_state["images_meta"]
        if not images_meta:
            bundle["should_export"] = False
            return bundle

        frames = {img_id: session_state["frame_states"][img_id].kps for img_id in images_meta}
        categories = [
            {
                "id": 1,
                "name": "court",
                "supercategory": "sports",
                "keypoints": bundle["court_spec"].names,
                "skeleton": bundle["court_spec"].skeleton,
            }
        ]

        # Append or update aggregated ann.json
        append_or_update(
            out_dir=out_dir,
            categories=categories,
            images_meta=images_meta,
            frames=frames,
            resume=bundle.get("resume_state", {}),
            video_path=str(video_path),
        )

        # Persist resume updates (image_id_map, next_image_id)
        save_resume(out_dir, bundle.get("resume_state", {}))

        logging.info(f"Exported annotations to {out_dir / 'ann.json'}")
        # Prevent re-exporting on the same key press
        bundle["should_export"] = False
        return bundle
