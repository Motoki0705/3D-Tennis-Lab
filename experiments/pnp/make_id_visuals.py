#!/usr/bin/env python3
import argparse
import ast
import json
import os
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import pandas as pd
import yaml

CANONICAL_ORDER = [
    "Net-Post_Left",
    "Net-Post_Right",
    "Net-Top_Left",
    "Net-Top_Right",
    "T-Service_Left",
    "T-Service_Right",
    "Center-Service_Line",
    "Sideline-Service_Left",
    "Sideline-Service_Right",
    "Baseline-T_Left",
    "Baseline-T_Right",
    "Baseline-Corner_Left",
    "Baseline-Corner_Right",
    "Sideline-Baseline_Corner_Left",
    "Sideline-Baseline_Corner_Right",
]


def parse_args():
    ap = argparse.ArgumentParser(description="Generate ID-labeled visuals for 2D annotations and 3D court BEV")
    ap.add_argument("--ann", required=True, type=str, help="Path to COCO annotation.json")
    ap.add_argument("--img-root", required=True, type=str, help="Root directory of images")
    ap.add_argument("--court-spec", required=True, type=str, help="YAML file describing 3D keypoints and dimensions")
    ap.add_argument("--out", default="out_id_visuals", type=str, help="Output directory")
    ap.add_argument("--name-source", default="auto", choices=["auto", "coco", "spec"], help="Name resolution strategy")
    ap.add_argument("--draw-names", action="store_true", help="Draw name along with kp_index on images")
    ap.add_argument("--max-images", type=int, default=0, help="Deprecated: use --select-image-id or --select-file-name")
    ap.add_argument("--select-image-id", type=int, default=None, help="Select a specific image_id to visualize")
    ap.add_argument("--select-file-name", type=str, default=None, help="Select a specific file_name to visualize")
    ap.add_argument(
        "--selection",
        type=str,
        default="first",
        choices=["first", "best"],
        help="If none selected, pick the first or the best by valid points",
    )
    ap.add_argument("--image-id", type=int, default=None, help="Process only this image id if provided")
    ap.add_argument("--image-file", type=str, default=None, help="Process only this image file name if provided")
    ap.add_argument(
        "--export-single",
        action="store_true",
        help="Also save one fixed-name overlay as annotation_ids.png in out root",
    )
    return ap.parse_args()


def load_coco(ann_path: str):
    with open(ann_path, encoding="utf-8") as f:
        coco = json.load(f)
    images = {im["id"]: im for im in coco["images"]}
    anns_per_img = {}
    for ann in coco["annotations"]:
        if ann.get("category_id", 1) != 1:
            continue
        anns_per_img.setdefault(ann["image_id"], []).append(ann)
    kp_names = []
    if "categories" in coco and len(coco["categories"]) > 0:
        kp_names = [str(x) for x in coco["categories"][0].get("keypoints", [])]
    return images, anns_per_img, kp_names


def load_court_spec(path: str):
    with open(path, encoding="utf-8") as f:
        spec = yaml.safe_load(f)
    dims = spec.get("dimensions", {})
    name2xyz = spec.get("keypoints_3d_m", {})

    def f(e):
        if isinstance(e, int | float):
            return float(e)
        if isinstance(e, str):
            if not all(c in "0123456789.+-() " for c in e):
                sys.exit(f"Unsafe expression in court_spec: {e}")
            return float(ast.literal_eval(e))
        return float(e)

    parsed = {}
    for k, v in name2xyz.items():
        if v is None:
            continue
        x, y, z = v
        parsed[str(k)] = (f(x), f(y), f(z))
    # extract dimensions with defaults
    half_length = float(dims.get("half_length", 11.885))
    half_singles = float(dims.get("half_singles", 4.115))
    half_doubles = float(dims.get("half_doubles", 5.4865))
    service_from_net = float(dims.get("service_from_net", 6.401))
    return parsed, {
        "half_length": half_length,
        "half_singles": half_singles,
        "half_doubles": half_doubles,
        "service_from_net": service_from_net,
    }


def resolve_names(kp_names_from_coco: list[str], name_source: str) -> list[str]:
    if name_source == "spec":
        return CANONICAL_ORDER[:]
    if name_source == "coco":
        return [str(x) for x in kp_names_from_coco]
    names = [str(x) for x in kp_names_from_coco] if kp_names_from_coco else []

    def is_numeric_label(s: str) -> bool:
        s = s.strip()
        return s.isdigit() or (s.startswith("kp_") and s[3:].isdigit())

    if len(names) == 0 or all(is_numeric_label(n) for n in names):
        return CANONICAL_ORDER[:]
    return names


def draw_text(img, text, x, y):
    # Simple outlined text for readability
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    # black outline
    cv2.putText(img, text, (int(x) + 1, int(y) + 1), font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    # white text
    cv2.putText(img, text, (int(x), int(y)), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def draw_annotation_ids(
    img_path: str, keypoints: list[float], resolved_names: list[str], draw_names: bool, out_path: str
):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        sys.exit(f"Image not found: {img_path}")
    n = len(keypoints) // 3
    for i in range(n):
        x = keypoints[3 * i + 0]
        y = keypoints[3 * i + 1]
        v = keypoints[3 * i + 2]
        if v <= 0:
            continue
        name = resolved_names[i] if i < len(resolved_names) else f"kp_{i}"
        cv2.circle(img, (int(x), int(y)), 3, (0, 255, 255), -1)
        label = f"{i}:{name}" if draw_names else f"{i}"
        draw_text(img, label, x + 4, y - 4)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(out_path, img)


def plot_court_bev(
    name2xyz: dict[str, tuple[float, float, float]], resolved_names: list[str], dims: dict[str, float], out_path: str
):
    # Build ordered lists by resolved_names
    xy = []
    labels = []
    for i, name in enumerate(resolved_names):
        if name in name2xyz:
            X, Y, Z = name2xyz[name]
            xy.append((X, Y))
            labels.append(f"{i}:{name}")
    # Court lines in XY plane where Z ignored
    hl = dims["half_length"]
    hs = dims["half_singles"]
    hd = dims["half_doubles"]
    sdist = dims["service_from_net"]

    plt.figure()
    ax = plt.gca()
    ax.set_aspect("equal", adjustable="box")

    # Baselines
    ax.plot([-hl, -hl], [-hd, hd])
    ax.plot([hl, hl], [-hd, hd])
    # Doubles sidelines
    ax.plot([-hl, hl], [-hd, -hd])
    ax.plot([-hl, hl], [hd, hd])
    # Singles sidelines
    ax.plot([-hl, hl], [-hs, -hs])
    ax.plot([-hl, hl], [hs, hs])
    # Net
    ax.plot([0, 0], [-hd, hd])
    # Service lines
    ax.plot([-sdist, -sdist], [-hs, hs])
    ax.plot([sdist, sdist], [-hs, hs])
    # Center service lines
    ax.plot([-sdist, 0], [0, 0])
    ax.plot([0, sdist], [0, 0])

    if xy:
        xs = [p[0] for p in xy]
        ys = [p[1] for p in xy]
        ax.scatter(xs, ys, s=25)
        for (x, y), lab in zip(xy, labels, strict=False):
            ax.text(x, y, lab, fontsize=8)

    ax.set_xlabel("X meters")
    ax.set_ylabel("Y meters")
    ax.set_title("Court BEV with kp_index and name")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    args = parse_args()
    images, anns_per_img, kp_names_coco = load_coco(args.ann)
    name2xyz, dims = load_court_spec(args.court_spec)
    resolved_names = resolve_names(kp_names_coco, args.name_source)

    out_dir = Path(args.out)
    out_ann = out_dir / "ann_ids"
    out_bev = out_dir / "bev"
    out_ann.mkdir(parents=True, exist_ok=True)
    out_bev.mkdir(parents=True, exist_ok=True)

    # Mapping CSV
    rows = []
    for i, nm in enumerate(resolved_names):
        X = Y = Z = None
        in_spec = nm in name2xyz
        if in_spec:
            X, Y, Z = name2xyz[nm]
        rows.append({"kp_index": i, "name": nm, "in_spec": in_spec, "X": X, "Y": Y, "Z": Z})
    pd.DataFrame(rows).to_csv(out_dir / "id_name_mapping.csv", index=False, encoding="utf-8")

    # Per image overlays
    processed = 0
    for img_id, ann_list in anns_per_img.items():
        # filtering by image id or filename when specified
        if args.image_id is not None and int(img_id) != int(args.image_id):
            continue
        ann = ann_list[0]
        iminfo = images[img_id]
        if args.image_file is not None and str(iminfo["file_name"]) != str(args.image_file):
            continue
        img_path = os.path.join(args.img_root, iminfo["file_name"])
        out_path = out_ann / f"{Path(iminfo['file_name']).stem}_ids.png"
        try:
            draw_annotation_ids(img_path, ann["keypoints"], resolved_names, args.draw_names, str(out_path))
            if args.export_single:
                # Save a fixed-name copy in out root for quick reference
                import shutil

                shutil.copyfile(str(out_path), str(Path(args.out) / "annotation_ids.png"))
        except Exception as e:
            print(f"Skip overlay for {iminfo['file_name']}: {e}")
        processed += 1
        if args.max_images > 0 and processed >= args.max_images:
            break
        break

    # One BEV for the spec
    plot_court_bev(name2xyz, resolved_names, dims, str(out_bev / "court_bev_ids.png"))
    print("Wrote overlays to:", str(out_ann))
    print("Wrote BEV to:", str(out_bev / "court_bev_ids.png"))
    print("Wrote mapping CSV to:", str(out_dir / "id_name_mapping.csv"))


if __name__ == "__main__":
    main()
