from __future__ import annotations


from typing import List, Optional
from contextlib import nullcontext

import torch
from omegaconf import DictConfig

from ..utils.model_io import build_model_from_cfg
from ..training.transforms import get_infer_transforms, _compute_target_dims


class InferRunner:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    @torch.no_grad()
    def run(self):
        model = build_model_from_cfg(self.cfg)
        model.eval()
        # Device/precision setup
        device = self._get_device()
        model.to(device)
        amp_ctx = self._get_amp_context(device)

        # Prefer video inference if configured
        infer_cfg = getattr(self.cfg, "infer", None)
        if infer_cfg is not None and infer_cfg.get("video", {}).get("input", None):
            self._run_video(model)
            return

        # Fallback: single-image inference via cfg.data.test.images
        img_path = self.cfg.data.test.images
        if not img_path:
            print("No test image (data.test.images) or video (infer.video.input). Inference runner exiting.")
            return

        import cv2
        import torchvision

        image_size = getattr(self.cfg.data, "image_size", 1024)
        aspect_ratio = getattr(self.cfg.data, "aspect_ratio", None)
        tfm = get_infer_transforms(image_size=image_size, aspect_ratio=aspect_ratio)

        bgr = cv2.imread(img_path)
        assert bgr is not None, f"Failed to read: {img_path}"
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        tx = tfm(image=rgb)["image"].to(device)  # tensor CHW in [0,1]
        with amp_ctx:
            outputs = model([tx])  # type: ignore

        # Draw and save
        preds = outputs[0]
        scores = preds.get("scores")
        boxes = preds.get("boxes")
        labels = preds.get("labels")
        score_thr = float(infer_cfg.get("image", {}).get("score_threshold", 0.5)) if infer_cfg else 0.5

        mask = scores > score_thr if scores is not None else None
        if mask is not None:
            boxes = boxes[mask]
            scores = scores[mask]
            labels = labels[mask]

        # to uint8
        img_uint8 = (tx.detach().cpu().clamp(0, 1) * 255).to(torch.uint8)
        label_strs = [f"{int(l.item())}:{float(s):.2f}" for l, s in zip(labels, scores)] if labels is not None else None
        drawn = torchvision.utils.draw_bounding_boxes(img_uint8, boxes=boxes, labels=label_strs, colors="red", width=2)
        out_path = infer_cfg.get("image", {}).get("output", None) if infer_cfg else None
        out_path = out_path or "inference.jpg"
        # Save via cv2 (convert to BGR HWC)
        drawn_hwc = drawn.permute(1, 2, 0).cpu().numpy()[:, :, ::-1]
        cv2.imwrite(out_path, drawn_hwc)
        print(f"Saved: {out_path}")

    def _run_video(self, model: torch.nn.Module) -> None:
        import cv2

        infer_cfg = self.cfg.infer
        vcfg = infer_cfg.video
        path_in: str = vcfg.input
        path_out: Optional[str] = vcfg.get("output", None)
        path_out = path_out or "inference.mp4"
        stride: int = int(vcfg.get("stride", 1))
        batch_size: int = int(vcfg.get("batch_size", 1))
        score_thr: float = float(vcfg.get("score_threshold", 0.5))
        draw_labels: bool = bool(vcfg.get("draw_labels", True))
        thickness: int = int(vcfg.get("thickness", 2))

        cap = cv2.VideoCapture(path_in)
        assert cap.isOpened(), f"Failed to open video: {path_in}"
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        # Build transforms and determine output canvas size
        image_size = getattr(self.cfg.data, "image_size", 1024)
        aspect_ratio = getattr(self.cfg.data, "aspect_ratio", None)
        tfm = get_infer_transforms(image_size=image_size, aspect_ratio=aspect_ratio)
        target_w, target_h = _compute_target_dims(image_size, aspect_ratio)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(path_out, fourcc, fps, (int(target_w), int(target_h)))

        frames_rgb: List = []
        frames_tx: List[torch.Tensor] = []
        frame_idx = 0
        written = 0

        device = self._get_device()
        amp_ctx = self._get_amp_context(device)

        while True:
            ok, bgr = cap.read()
            if not ok:
                break

            if stride > 1 and (frame_idx % stride != 0):
                frame_idx += 1
                continue

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            tx = tfm(image=rgb)["image"].to(device)
            frames_rgb.append(rgb)
            frames_tx.append(tx)
            frame_idx += 1

            if len(frames_tx) >= batch_size:
                self._process_and_write(
                    model, frames_tx, frames_rgb, writer, score_thr, draw_labels, thickness, amp_ctx
                )
                frames_tx.clear()
                frames_rgb.clear()

        if frames_tx:
            self._process_and_write(model, frames_tx, frames_rgb, writer, score_thr, draw_labels, thickness, amp_ctx)

        writer.release()
        cap.release()
        print(f"Saved video: {path_out}")

    def _process_and_write(
        self,
        model: torch.nn.Module,
        frames_tx: List[torch.Tensor],
        frames_rgb: List,
        writer,
        score_thr: float,
        draw_labels: bool,
        thickness: int,
        amp_ctx,
    ) -> None:
        import torchvision

        with amp_ctx:
            outputs = model(frames_tx)  # type: ignore
        for tx, rgb, pred in zip(frames_tx, frames_rgb, outputs):
            boxes = pred.get("boxes")
            scores = pred.get("scores")
            labels = pred.get("labels")
            mask = scores > score_thr if scores is not None else None
            if mask is not None:
                boxes = boxes[mask]
                scores = scores[mask]
                labels = labels[mask]

            img_uint8 = (tx.detach().cpu().clamp(0, 1) * 255).to(torch.uint8)
            label_strs = None
            if draw_labels and labels is not None and scores is not None:
                label_strs = [f"{int(l.item())}:{float(s):.2f}" for l, s in zip(labels, scores)]
            drawn = torchvision.utils.draw_bounding_boxes(
                img_uint8, boxes=boxes, labels=label_strs, colors="red", width=thickness
            )
            drawn_hwc = drawn.permute(1, 2, 0).cpu().numpy()[:, :, ::-1]  # RGB->BGR
            writer.write(drawn_hwc)

    # --- device/precision helpers ---
    def _get_device(self):
        prefer = getattr(self.cfg, "infer", {}).get("device", "auto")
        if prefer == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            return torch.device(prefer)
        except Exception:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _get_amp_context(self, device):
        prec = str(getattr(self.cfg, "infer", {}).get("precision", "fp32")).lower()
        if prec in ("fp16", "half", "float16") and device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return nullcontext()
