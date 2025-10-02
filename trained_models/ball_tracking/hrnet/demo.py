import csv
from pathlib import Path

import cv2
import hydra
import torch
from PIL import Image

from .hrnet_loader import HRNetLoadConfig, load_hrnet_with_ckpt


def video_io(video_path: str | Path, output_path: str | Path) -> tuple[cv2.VideoCapture, cv2.VideoWriter]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open input video.")
        exit()

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"Width: {width}, Height: {height}, FPS: {fps}")

    fourcc = cv2.VideoWriter_fourcc("mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    return cap, out


def save_csv(results, csv_path):
    with open(csv_path, mode="w", newline=" ", encoding="uft-8") as f:
        writer = csv.DictWriter(f, fieldnames=["x", "y", "visi"])
        writer.writeheader()
        writer.writerow(results)


def infer_batch(batch: torch.Tensor, detector, tracker, affine_mats, device, use_half):
    batch_size, out_frames, height, width = batch.shape()
    with torch.no_grad():
        if use_half:
            with torch.autocast(device_type=device, dtype=torch.float16):
                detections, _hms_vis = detector(batch, affine_mats)
            results = []
            for batch_id in range(batch_size):
                for out_id in range(out_frames):
                    frame_dets = detections[batch_id][out_id]
                    frame_result = tracker.update(frame_dets)
                    results.append(frame_result)

    return results


def apply_transform(transform, build_affine_transform, frame_bgr, input_wh):
    trans_input = build_affine_transform(frame_bgr, input_wh)
    warped = cv2.warpAffine(frame_bgr, trans_input, input_wh, flags=cv2.INTER_LINEAR)
    pil_image = Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
    tensor = transform(pil_image)
    return tensor


def get_trans_output_inv(build_affine_transform, frame_bgr, output_wh, out_scales, device):
    # Prepare inverse transforms for post-processing
    trans_outputs = {}
    out_w, out_h = output_wh
    for scale in out_scales:
        trans_output_inv = build_affine_transform(frame_bgr, (out_w, out_h), inv=1)
        trans_outputs[int(scale)] = torch.tensor(trans_output_inv, dtype=torch.float32, device=device)
        out_w = max(out_w / 2, 1)
        out_h = max(out_h / 2, 1)
    return trans_outputs


def infer_video(
    cap,
    out,
    detector,
    tracker,
    transform,
    build_affine_transform,
    device,
    use_half,
    batch_size,
    frame_in,
    frame_out,
    input_wh,
    output_wh,
    out_scales,
):
    batch = []
    clip = []
    results = []
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        tensor = apply_transform(transform, build_affine_transform, frame_bgr, input_wh)
        trans_outputs = get_trans_output_inv(build_affine_transform, frame_bgr, output_wh, out_scales)

        clip.append(tensor)
        if len(clip) == frame_in:
            clip_tensor = torch.cat(clip, dim=0).unsqueeze(0)
            batch.append(clip_tensor)
            clip = []
        if len(batch) == batch_size:
            batch_tensor = torch.cat(batch, dim=0)
            batch_results = infer_batch(batch_tensor, detector, tracker, trans_outputs, device, use_half)
            batch_size, output_frames, height, width = batch_tensor.shape
            for idx in range(batch_size * output_frames):
                frame_result = batch_results[idx]
                x, y = frame_result["x"], frame_result["y"]
                visi = frame_result["visi"]
                if visi:
                    cv2.circle(frame_bgr, center=(x, y), radius=3, color=(0, 0, 255), thickness=-1)
                out.write(frame_bgr)
                results.append(frame_result)
            batch = []

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    return results


@hydra.main(config_path=str(Path(__file__).parent / "configs"), config_name="detect")
def main(cfg):
    hrnet_cfg: HRNetLoadConfig = HRNetLoadConfig(cfg)
    detector, tracker, transform, build_affine_transform, device = load_hrnet_with_ckpt(load_cfg=hrnet_cfg)
    input_wh = cfg.model.inp_width, cfg.model.inp_height
    output_wh = cfg.model.out_width, cfg.model.out_height
    use_half = cfg.use_half
    batch_size = cfg.batch_size
    frame_in, frame_out = cfg.model.frame_in, cfg.model.frame_out
    out_scales = cfg.model.out_scales

    cap, out = video_io(Path(cfg.video_path), Path(cfg.output_path))
    results = infer_video(
        cap,
        out,
        detector,
        tracker,
        transform,
        build_affine_transform,
        device,
        use_half,
        batch_size,
        frame_in,
        frame_out,
        input_wh,
        output_wh,
        out_scales,
    )
    csv_path = cfg.csv_path
    if csv_path is not None:
        save_csv(results, csv_path=cfg.csv_path)
