import json
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Optional, Sequence, Tuple
import logging

import cv2
import numpy as np
import pytest
import torch
import yaml

from development.core.datasets import (
    BallSequenceDataset,
    CourtKeypointDataset,
    PlayerSequenceDataset,
)

logger = logging.getLogger(__name__)


REPO_ROOT = Path(__file__).resolve().parents[4]
CONFIG_ROOT = REPO_ROOT / "development/core/configs/data"
ARTIFACT_ROOT = Path(__file__).resolve().parent / "artifacts"


def _dump_debug_yaml(name: str, data: Any) -> None:
    out = ARTIFACT_ROOT / "debug"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.yaml").write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _load_yaml_config(name: str) -> Mapping[str, Any]:
    config_path = CONFIG_ROOT / f"{name}.yaml"
    if not config_path.exists():
        pytest.skip(f"Missing config file: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _resolve_train_paths(cfg: Mapping[str, Any]) -> Tuple[Path, Path]:
    train_cfg = cfg.get("paths", {}).get("train", {})
    image_path = train_cfg.get("images")
    annotation_path = train_cfg.get("annotation")
    if not image_path or not annotation_path:
        pytest.skip("Training paths are not fully specified in config.")
    image_dir = Path(image_path)
    ann_file = Path(annotation_path)
    if not image_dir.is_absolute():
        image_dir = (REPO_ROOT / image_dir).resolve()
    if not ann_file.is_absolute():
        ann_file = (REPO_ROOT / ann_file).resolve()
    if not image_dir.exists() or not any(image_dir.iterdir()):
        pytest.skip(f"Image directory missing or empty: {image_dir}")
    if not ann_file.exists():
        pytest.skip(f"Annotation file missing: {ann_file}")
    return image_dir, ann_file


def _ensure_hw(value: Optional[Any]) -> Optional[Tuple[int, int]]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        if len(value) == 2:
            return int(value[0]), int(value[1])
        if len(value) == 1:
            return int(value[0]), int(value[0])
    if isinstance(value, int):
        return int(value), int(value)
    if isinstance(value, Mapping):
        default_entry = value.get("default")
        if default_entry is not None:
            return _ensure_hw(default_entry)
        return None
    raise ValueError(f"Cannot convert value to (H, W): {value!r}")


def _first_or(value: Optional[Iterable[str]], default: Optional[str] = None) -> Optional[str]:
    if value is None:
        return default
    for item in value:
        return item
    return default


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _denormalize_clip(clip: torch.Tensor, mean: Sequence[float], std: Sequence[float]) -> np.ndarray:
    clip = clip.detach().cpu().float()
    mean_tensor = torch.tensor(mean, dtype=clip.dtype).view(1, -1, 1, 1)
    std_tensor = torch.tensor(std, dtype=clip.dtype).view(1, -1, 1, 1)
    restored = clip * std_tensor + mean_tensor
    restored = restored.clamp(0.0, 1.0)
    restored = (restored * 255.0).round().to(torch.uint8)
    return restored.permute(0, 2, 3, 1).numpy()


def _save_rgb_image(path: Path, image_rgb: np.ndarray) -> Path:
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    success = cv2.imwrite(str(path), bgr)
    if not success:
        raise RuntimeError(f"Failed to write image: {path}")
    return path


def _write_metadata(path: Path, metadata: Mapping[str, Any]) -> Path:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
    return path


def _capture_ball_sample(sample: Mapping[str, Any], dataset: BallSequenceDataset, out_dir: Path) -> List[Path]:
    outputs: List[Path] = []
    clip = sample["inputs"]
    frames_rgb = _denormalize_clip(clip, dataset.normalize_mean, dataset.normalize_std)
    targets = sample.get("targets", {})
    heatmaps = targets.get("heatmaps")
    if isinstance(heatmaps, torch.Tensor):
        heatmaps = heatmaps.detach().cpu().numpy()
    metadata = sample.get("metadata", {})

    for idx, frame in enumerate(frames_rgb):
        frame_path = out_dir / f"ball_clip_frame{idx}.png"
        outputs.append(_save_rgb_image(frame_path, frame))
        if isinstance(heatmaps, np.ndarray) and heatmaps.shape[0] > idx:
            heatmap = heatmaps[idx]
            if heatmap.ndim == 3:
                heatmap = heatmap[0]
            resized = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
            norm = cv2.normalize(resized, None, 0, 255, cv2.NORM_MINMAX)
            colored = cv2.applyColorMap(norm.astype(np.uint8), cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), 0.7, colored, 0.3, 0)
            heatmap_path = out_dir / f"ball_clip_frame{idx}_heatmap.png"
            success = cv2.imwrite(str(heatmap_path), overlay)
            if not success:
                raise RuntimeError(f"Failed to write heatmap overlay: {heatmap_path}")
            outputs.append(heatmap_path)

    outputs.append(_write_metadata(out_dir / "ball_clip_metadata.json", metadata))
    return outputs


def _capture_player_sample(sample: Mapping[str, Any], dataset: PlayerSequenceDataset, out_dir: Path) -> List[Path]:
    outputs: List[Path] = []
    clip = sample["inputs"]
    frames_rgb = _denormalize_clip(clip, dataset.normalize_mean, dataset.normalize_std)
    targets = sample.get("targets", {})
    bboxes_seq = targets.get("bboxes", [])
    classes_seq = targets.get("classes", [])

    for idx, frame in enumerate(frames_rgb):
        vis = frame.copy()
        bboxes = bboxes_seq[idx] if idx < len(bboxes_seq) else []
        class_ids = classes_seq[idx] if idx < len(classes_seq) else []
        for bbox_idx, bbox in enumerate(bboxes):
            x, y, w, h = map(float, bbox)
            pt1 = (int(round(x)), int(round(y)))
            pt2 = (int(round(x + w)), int(round(y + h)))
            cv2.rectangle(vis, pt1, pt2, (0, 255, 0), 2)
            if bbox_idx < len(class_ids):
                label = str(class_ids[bbox_idx])
                cv2.putText(
                    vis, label, (pt1[0], max(0, pt1[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA
                )
        outputs.append(_save_rgb_image(out_dir / f"player_clip_frame{idx}.png", vis))

    outputs.append(_write_metadata(out_dir / "player_clip_metadata.json", sample.get("metadata", {})))
    return outputs


def _capture_court_sample(sample: Mapping[str, Any], dataset: CourtKeypointDataset, out_dir: Path) -> List[Path]:
    outputs: List[Path] = []
    clip = sample["inputs"]
    frames_rgb = _denormalize_clip(clip, dataset.normalize_mean, dataset.normalize_std)
    targets = sample.get("targets", {})
    heatmaps = targets.get("heatmaps")
    if isinstance(heatmaps, torch.Tensor):
        heatmaps = heatmaps.detach().cpu().numpy()
    keypoints = targets.get("keypoints", [])

    for idx, frame in enumerate(frames_rgb):
        vis = frame.copy()
        if isinstance(heatmaps, np.ndarray) and heatmaps.ndim == 3 and heatmaps.size > 0:
            aggregate = np.max(heatmaps, axis=0)
            resized = cv2.resize(aggregate, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
            norm = cv2.normalize(resized, None, 0, 255, cv2.NORM_MINMAX)
            colored = cv2.applyColorMap(norm.astype(np.uint8), cv2.COLORMAP_MAGMA)
            vis = cv2.addWeighted(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), 0.65, colored, 0.35, 0)
            vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        if isinstance(keypoints, Sequence):
            for point in keypoints:
                if not point:
                    continue
                x, y = map(int, map(round, point))
                cv2.circle(vis, (x, y), 4, (255, 255, 0), -1)
        outputs.append(_save_rgb_image(out_dir / f"court_frame{idx}.png", vis))

    outputs.append(_write_metadata(out_dir / "court_sample_metadata.json", sample.get("metadata", {})))
    return outputs


@pytest.mark.dataset
def test_ball_dataset_visualization():
    cfg = _load_yaml_config("ball")
    _dump_debug_yaml("ball_cfg", cfg)
    assert cfg
    image_dir, ann_file = _resolve_train_paths(cfg)
    dataset_cfg = cfg.get("dataset", {})
    sequence_cfg = dataset_cfg.get("sequence", {})
    image_size = _ensure_hw(dataset_cfg.get("image_size"))
    heatmap_size = dataset_cfg.get("heatmap_size")
    if heatmap_size is None:
        if image_size is None:
            pytest.skip("Ball config lacks image_size to derive heatmap size.")
        stride = int(dataset_cfg.get("output_stride", 1))
        heatmap_size = (max(1, image_size[0] // stride), max(1, image_size[1] // stride))
    else:
        heatmap_size = _ensure_hw(heatmap_size)
    dataset = BallSequenceDataset(
        annotation_file=str(ann_file),
        image_dir=str(image_dir),
        sequence_length=int(sequence_cfg.get("length", 3)),
        frame_stride=int(sequence_cfg.get("stride", 1)),
        allow_partial_last=bool(sequence_cfg.get("allow_partial_last", False)),
        drop_short_clips=bool(sequence_cfg.get("drop_short_clips", False)),
        heatmap_size=heatmap_size,
        heatmap_sigma=float(dataset_cfg.get("heatmap_sigma", 3.0)),
        image_size=image_size,
        category_name=_first_or(dataset_cfg.get("categories", {}).get("names"), "ball"),
        normalize_mean=dataset_cfg.get("normalization", {}).get("mean", (0.485, 0.456, 0.406)),
        normalize_std=dataset_cfg.get("normalization", {}).get("std", (0.229, 0.224, 0.225)),
    )
    assert len(dataset) > 0
    sample = dataset[0]
    out_dir = _ensure_dir(ARTIFACT_ROOT / "ball")
    saved = _capture_ball_sample(sample, dataset, out_dir)
    for path in saved:
        assert path.exists() and path.stat().st_size > 0


@pytest.mark.dataset
def test_player_dataset_visualization():
    cfg = _load_yaml_config("player")
    _dump_debug_yaml("player_cfg", cfg)
    assert cfg
    image_dir, ann_file = _resolve_train_paths(cfg)
    dataset_cfg = cfg.get("dataset", {})
    sequence_cfg = dataset_cfg.get("sequence", {})
    sequence_length = int(sequence_cfg.get("length", 1))
    frame_stride = int(sequence_cfg.get("stride", 1))
    dataset = PlayerSequenceDataset(
        annotation_file=str(ann_file),
        image_dir=str(image_dir),
        sequence_length=sequence_length,
        frame_stride=frame_stride,
        allow_partial_last=bool(sequence_cfg.get("allow_partial_last", False)),
        drop_short_clips=bool(sequence_cfg.get("drop_short_clips", False)),
        target_category=dataset_cfg.get("categories", {}).get("target", "player"),
        min_box_size=float(dataset_cfg.get("categories", {}).get("min_size", 1.0)),
        image_size=_ensure_hw(dataset_cfg.get("image_size")),
        normalize_mean=dataset_cfg.get("normalization", {}).get("mean", (0.485, 0.456, 0.406)),
        normalize_std=dataset_cfg.get("normalization", {}).get("std", (0.229, 0.224, 0.225)),
    )
    if len(dataset) == 0:
        pytest.skip("Player dataset is empty – nothing to visualise.")
    sample = dataset[0]
    out_dir = _ensure_dir(ARTIFACT_ROOT / "player")
    saved = _capture_player_sample(sample, dataset, out_dir)
    for path in saved:
        assert path.exists() and path.stat().st_size > 0


@pytest.mark.dataset
def test_court_dataset_visualization():
    cfg = _load_yaml_config("court")
    _dump_debug_yaml("court_cfg", cfg)
    assert cfg
    image_dir, ann_file = _resolve_train_paths(cfg)
    dataset_cfg = cfg.get("dataset", {})
    sequence_cfg = dataset_cfg.get("sequence", {})
    heatmap_size = _ensure_hw(dataset_cfg.get("heatmap_size"))
    if heatmap_size is None:
        pytest.skip("Court config must specify heatmap_size for visualisation.")
    dataset = CourtKeypointDataset(
        annotation_file=str(ann_file),
        image_dir=str(image_dir),
        sequence_length=int(sequence_cfg.get("length", 1)),
        frame_stride=int(sequence_cfg.get("stride", 1)),
        allow_partial_last=bool(sequence_cfg.get("allow_partial_last", False)),
        drop_short_clips=bool(sequence_cfg.get("drop_short_clips", False)),
        heatmap_size=heatmap_size,
        heatmap_sigma=float(dataset_cfg.get("heatmap_sigma", 3.0)),
        image_size=_ensure_hw(dataset_cfg.get("image_size")),
        normalize_mean=dataset_cfg.get("normalization", {}).get("mean", (0.485, 0.456, 0.406)),
        normalize_std=dataset_cfg.get("normalization", {}).get("std", (0.229, 0.224, 0.225)),
    )
    if len(dataset) == 0:
        pytest.skip("Court dataset is empty – nothing to visualise.")
    sample = dataset[0]
    out_dir = _ensure_dir(ARTIFACT_ROOT / "court")
    saved = _capture_court_sample(sample, dataset, out_dir)
    for path in saved:
        assert path.exists() and path.stat().st_size > 0
