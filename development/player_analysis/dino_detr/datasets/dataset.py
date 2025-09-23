from __future__ import annotations

from typing import Any, Optional, Sequence, Mapping


import torch

from development.core.datasets.player import PlayerSequenceDataset


class CocoDetectionDataset(PlayerSequenceDataset):
    """
    COCO detection dataset based on :class:`PlayerSequenceDataset`.

    入力 (annotation_file / image_dir):
        - COCO 形式のアノテーション (images / annotations / categories)
        - 画像ディレクトリ

    出力 (__getitem__):
        {
            "inputs":  Tensor[T, C, H, W]     # 正規化済み画像 (通常 T=1)
            "targets": {
                "boxes":  List[Tensor[Mi, 4]] # 各フレームの bbox (xyxy, ピクセル座標)
                "labels": List[Tensor[Mi]]    # 各フレームのクラスID
            },
            "metadata": dict                  # 画像パス, frame_id など補助情報
        }

    備考:
        - Albumentations の transform は "coco"(xywh) 形式で実行されるが、
          本クラスが最終的に xyxy Tensor に変換して返す。
        - DETR 系モデルで使う場合は、collate_fn で T=1 を潰し、
          (images: [N,C,H,W], targets: List[dict]) に整形して渡す。
    """

    def __init__(
        self,
        *,
        annotation_file: Optional[str | bytes] = None,
        image_dir: Optional[str] = None,
        coco: Optional[Mapping[str, Any]] = None,
        sequence_length: int = 1,
        frame_stride: int = 1,
        target_category: str = "player",
        min_box_size: float = 1.0,
        image_size: Optional[Sequence[int]] = None,
        drop_short_clips: bool = False,
        allow_partial_last: bool = False,
        transform: Optional[Any] = None,
        normalize_mean: Sequence[float] = (0.485, 0.456, 0.406),
        normalize_std: Sequence[float] = (0.229, 0.224, 0.225),
    ) -> None:
        super().__init__(
            annotation_file=annotation_file,
            image_dir=image_dir,
            coco=coco,
            sequence_length=sequence_length,
            frame_stride=frame_stride,
            target_category=target_category,
            min_box_size=min_box_size,
            image_size=image_size,
            drop_short_clips=drop_short_clips,
            allow_partial_last=allow_partial_last,
            transform=transform,
            normalize_mean=normalize_mean,
            normalize_std=normalize_std,
        )

    # 置き換え版 _finalize_sample（transform後に xywh -> xyxy に統一）
    def _finalize_sample(self, sample, *, payloads, metadata):
        """
        sample["targets"]["bboxes"]: list[T][Mi,4] in xywh (Albumentations "coco" 出力)
        sample["inputs"]: Tensor[T,C,H,W] もしくは numpy -> transform側の実装に依存
        ここで xyxy Tensor に変換し、labels も Tensor 化して返す。
        """
        TCHW = sample["inputs"].shape
        assert len(TCHW) == 4, "inputs must be [T,C,H,W]"
        T, _, H, W = TCHW

        tgt_in = sample.get("targets", {})
        bboxes_seq = tgt_in.get("bboxes", [])  # list[T][Mi,4] (xywh)
        classes_seq = tgt_in.get("classes", [])  # list[T][Mi]

        # 長さを T に合わせて安全化
        def _ensure(seq, filler):
            seq = [] if seq is None else list(seq)
            if len(seq) < T:
                seq.extend(filler for _ in range(T - len(seq)))
            return seq[:T]

        bboxes_seq = _ensure(bboxes_seq, [])
        classes_seq = _ensure(classes_seq, [])

        # フレームごとに xywh -> xyxy 変換 + Tensor 化
        boxes_xyxy_T = []
        labels_T = []
        for t in range(T):
            boxes_xywh = bboxes_seq[t] or []
            labels = classes_seq[t] or []
            boxes_xyxy_T.append(self._xywh_list_to_xyxy_tensor(boxes_xywh, H=H, W=W, clamp=True))
            labels_T.append(self._labels_list_to_tensor(labels))

        targets = {
            # list[T] of Tensor[Mi,4] / Tensor[Mi]
            "boxes": boxes_xyxy_T,
            "labels": labels_T,
        }

        return {
            "inputs": sample["inputs"],  # [T,C,H,W]
            "targets": targets,  # xyxy に統一済み
            "metadata": sample.get("metadata", metadata),
        }

    @staticmethod
    def _xywh_to_xyxy_np(box):
        # box: [x, y, w, h] (float)
        x, y, w, h = float(box[0]), float(box[1]), float(box[2]), float(box[3])
        return [x, y, x + w, y + h]

    @staticmethod
    def _xywh_list_to_xyxy_tensor(frame_boxes_xywh, H=None, W=None, clamp=True):
        """
        frame_boxes_xywh: list[list[4]] (xywh)  -> Tensor[M,4] (xyxy)
        H,W: 画像サイズ（任意）。指定時は [0..W],[0..H] にクリップ。
        """
        if not frame_boxes_xywh:
            return torch.zeros((0, 4), dtype=torch.float32)
        xyxy = [PlayerSequenceDataset._xywh_to_xyxy_np(b) for b in frame_boxes_xywh]
        t = torch.tensor(xyxy, dtype=torch.float32)
        if clamp and (H is not None) and (W is not None):
            # [x1,y1,x2,y2] を画像境界にクリップ
            t[:, 0] = t[:, 0].clamp_(min=0.0, max=W - 1)
            t[:, 1] = t[:, 1].clamp_(min=0.0, max=H - 1)
            t[:, 2] = t[:, 2].clamp_(min=0.0, max=W)
            t[:, 3] = t[:, 3].clamp_(min=0.0, max=H)
        return t

    @staticmethod
    def _labels_list_to_tensor(frame_labels):
        if not frame_labels:
            return torch.zeros((0,), dtype=torch.int64)
        return torch.tensor([int(x) for x in frame_labels], dtype=torch.int64)
