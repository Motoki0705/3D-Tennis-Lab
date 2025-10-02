import torch
from typing import Dict


def render_pred_gt_bboxes_overlay_dict(
    *,
    tag: str = "PredGToverlay",
    image_key: str = "images",
    pred_bboxes_key: str = "pred_bboxes",
    gt_bboxes_key: str = "target_bboxes",
    pred_labels_key: str | None = "class_labels",  # 予測ラベル（任意）
    gt_labels_key: str | None = None,  # GTラベル（任意）
    pred_scores_key: str | None = None,  # 予測スコア（任意）
    assume_xyxy: bool = True,  # False の場合、[x,y,w,h] を xyxy に変換
    to_uint8: bool = True,  # 描画前に [0..1] -> uint8 へ
    pred_color: str = "red",
    gt_color: str = "blue",
    linewidth: int = 2,
):
    """
    1つのカテゴリー(tag)の下に、pred と GT の bbox を色分けして重ね描きした画像を返すレンダラー。
    返り値: { tag: Tensor([N,3,H,W]) } (0..1 float)

    期待する buf のテンソル:
      - images: [N, C, H, W] (C=1/3, 値は 0..1 推奨)
      - pred_bboxes: [N, Kp, 4] (xyxy or xywh)
      - target_bboxes: [N, Kg, 4] (xyxy or xywh)
      - class_labels (任意): [N, Kp]
      - (任意) gt_labels_key: [N, Kg]
      - (任意) pred_scores_key: [N, Kp] (0..1 など)
    """
    from torchvision.utils import draw_bounding_boxes

    def _to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
        if boxes.numel() == 0:
            return boxes
        if assume_xyxy:
            return boxes
        # xywh -> xyxy
        x, y, w, h = boxes.unbind(-1)
        return torch.stack([x, y, x + w, y + h], dim=-1)

    def _valid_mask_xyxy(boxes_xyxy: torch.Tensor) -> torch.Tensor:
        if boxes_xyxy.numel() == 0:
            return boxes_xyxy.new_zeros((0,), dtype=torch.bool)
        x1, y1, x2, y2 = boxes_xyxy.unbind(-1)
        # 幅・高さが正のものだけ描画
        return (x2 > x1) & (y2 > y1)

    def _fmt_labels(lbls: torch.Tensor | None, scrs: torch.Tensor | None) -> list[str] | None:
        if lbls is None and scrs is None:
            return None
        if lbls is not None:
            lbls_list = lbls.tolist()
        else:
            lbls_list = [None] * (scrs.size(0) if scrs is not None else 0)
        if scrs is not None:
            scrs_list = scrs.tolist()
        else:
            scrs_list = [None] * len(lbls_list)
        out = []
        for li, si in zip(lbls_list, scrs_list):
            if li is None and si is None:
                out.append("")
            elif li is None:
                out.append(f"{si:.2f}")
            elif si is None or (isinstance(si, float) and (si != si)):  # NaN guard
                out.append(str(int(li)))
            else:
                out.append(f"{int(li)}:{si:.2f}")
        return out

    def _renderer(buf: Dict[str, torch.Tensor], step: int, stage: str):
        out: Dict[str, torch.Tensor] = {}

        imgs = buf.get(image_key)
        pred_b = buf.get(pred_bboxes_key)
        gt_b = buf.get(gt_bboxes_key)
        if (imgs is None) or (pred_b is None) or (gt_b is None):
            return out

        # オプションのラベル/スコア
        pred_lbl = buf.get(pred_labels_key) if pred_labels_key and pred_labels_key in buf else None
        gt_lbl = buf.get(gt_labels_key) if gt_labels_key and gt_labels_key in buf else None
        pred_scr = buf.get(pred_scores_key) if pred_scores_key and pred_scores_key in buf else None

        N = imgs.size(0)
        drawn: list[torch.Tensor] = []

        for i in range(N):
            img = imgs[i]
            # 1ch -> 3ch
            if img.size(0) == 1:
                img = img.repeat(3, 1, 1)
            # uint8 変換
            canvas = (img.clamp(0, 1) * 255).to(torch.uint8) if to_uint8 else img.clone()

            # ---- GT を先に描画 ----
            gt_i = _to_xyxy(gt_b[i])
            mask_gt = _valid_mask_xyxy(gt_i)
            if mask_gt.any():
                gt_boxes = gt_i[mask_gt]
                gt_labels = gt_lbl[i][mask_gt] if (gt_lbl is not None) else None
                gt_labels_str = _fmt_labels(gt_labels, None)
                canvas = draw_bounding_boxes(
                    canvas,
                    gt_boxes,
                    labels=gt_labels_str,
                    colors=gt_color,
                    width=linewidth,
                )

            # ---- Pred を後から上書き描画 ----
            pred_i = _to_xyxy(pred_b[i])
            mask_pr = _valid_mask_xyxy(pred_i)
            if mask_pr.any():
                pr_boxes = pred_i[mask_pr]
                pr_labels = pred_lbl[i][mask_pr] if (pred_lbl is not None) else None
                pr_scores = pred_scr[i][mask_pr] if (pred_scr is not None) else None
                pr_labels_str = _fmt_labels(pr_labels, pr_scores)
                canvas = draw_bounding_boxes(
                    canvas,
                    pr_boxes,
                    labels=pr_labels_str,
                    colors=pred_color,
                    width=linewidth,
                )

            # 後段のロガー規約に合わせて 0..1 float へ
            drawn.append((canvas.float() / 255.0) if to_uint8 else canvas.clamp(0, 1))

        out[tag] = torch.stack(drawn, dim=0)  # [N,3,H,W]
        return out

    return _renderer


__all__ = ["render_pred_gt_bboxes_overlay_dict"]
