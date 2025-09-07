import json
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from hydra.utils import to_absolute_path
from torch.utils.data import Dataset


class CourtKeypointDataset(Dataset):
    """
    テニスコートキーポイントデータセット。

    - 画像とアノテーションを読み込む。
    - キーポイント座標からガウシアンヒートマップを生成する。
    - データ拡張を適用する。
    - deep_supervision=True の場合、OS=8の補助ターゲットも返す。
    """

    def __init__(
        self,
        *,
        img_dir: str | Path,
        annotation_file: str | Path,
        img_size: tuple[int, int],
        output_stride: int = 4,
        sigma: float = 2.0,
        deep_supervision: bool = False,
        use_offset: bool = False,
        transform=None,
        num_keypoints: int | None = None,
    ):
        self.img_dir = Path(img_dir)
        self.img_size = tuple(img_size)  # (H, W)
        self.output_stride = int(output_stride)
        self.sigma = float(sigma)
        self.deep_supervision = bool(deep_supervision)
        self.use_offset = bool(use_offset)
        self.transform = transform
        self.num_keypoints = int(num_keypoints) if num_keypoints is not None else 15

        with open(annotation_file) as f:
            coco_data = json.load(f)

        self.image_info = {img["id"]: img for img in coco_data["images"]}
        self.annotations = coco_data["annotations"]

        # keypointsの定義 (デバッグや可視化用)
        try:
            self.keypoint_names = coco_data["categories"][0]["keypoints"]
            self.skeleton = coco_data["categories"][0]["skeleton"]
            if num_keypoints is None and isinstance(self.keypoint_names, list):
                self.num_keypoints = len(self.keypoint_names)
        except Exception:
            self.keypoint_names = None
            self.skeleton = None

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        image_id = annotation["image_id"]
        image_meta = self.image_info[image_id]

        rel_path = self.img_dir / image_meta["file_name"]
        img_path = to_absolute_path(str(rel_path))
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        original_h, original_w, _ = image.shape

        keypoints = np.array(annotation["keypoints"]).reshape(-1, 3)

        # 可視性フラグ v>0 のみ拡張対象
        visible_keypoints = keypoints[keypoints[:, 2] > 0, :2]

        # データ拡張（AlbumentationsのKeypointParamsを前提）
        if self.transform:
            transformed = self.transform(image=image, keypoints=visible_keypoints)
            image = transformed["image"]
            transformed_keypoints = transformed.get("keypoints", [])
        else:
            # transformがない場合でもリサイズとテンソル化は行う
            resize_transform = A.Compose(
                [A.Resize(height=self.img_size[0], width=self.img_size[1]), ToTensorV2()],
                keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
            )
            transformed = resize_transform(image=image, keypoints=visible_keypoints)
            image = transformed["image"]
            # キーポイントも手動でスケール（Resizeが行うのと等価）
            scale_x = self.img_size[1] / original_w
            scale_y = self.img_size[0] / original_h
            visible_keypoints = visible_keypoints.astype(np.float32)
            visible_keypoints[:, 0] *= scale_x
            visible_keypoints[:, 1] *= scale_y
            transformed_keypoints = visible_keypoints

        # ヒートマップ生成（メイン: OS=out_stride）
        H, W = self.img_size
        h_main, w_main = H // self.output_stride, W // self.output_stride
        target_main = self._generate_heatmap_for_size(
            transformed_keypoints, keypoints, (h_main, w_main), sigma=self.sigma
        )

        out = {
            "inputs": image.float() / 255.0 if image.dtype == torch.uint8 else image,
            "targets": torch.from_numpy(target_main).float(),
        }

        # Deep supervision: OS=8 の補助ターゲット
        if self.deep_supervision:
            h_aux, w_aux = H // 8, W // 8
            target_aux = self._generate_heatmap_for_size(
                transformed_keypoints, keypoints, (h_aux, w_aux), sigma=self.sigma
            )
            out["aux"] = {"os8": torch.from_numpy(target_aux).float()}

        # Offset head targets at main scale
        if self.use_offset:
            off_map, off_mask = self._generate_offset_for_size(transformed_keypoints, keypoints, (h_main, w_main))
            out["offsets"] = torch.from_numpy(off_map).float()
            out["offset_mask"] = torch.from_numpy(off_mask).float()

        return out

    def _generate_heatmap_for_size(self, transformed_kps, original_kps, size_hw: tuple[int, int], sigma: float):
        """指定サイズのガウシアンヒートマップを生成する。

        transformed_kps: 変換後の可視キーポイント配列 (N_vis, 2)
        original_kps: 元の全キーポイント配列 (K, 3) 可視性でマッピング
        size_hw: (H_out, W_out)
        sigma: ガウシアンのσ（ヒートマップ座標系のピクセル単位）
        """
        h_out, w_out = int(size_hw[0]), int(size_hw[1])
        heatmap = np.zeros((self.num_keypoints, h_out, w_out), dtype=np.float32)

        # transformed_kpsは可視キーポイントのみなので、元インデックスに戻す
        visible_indices = np.where(original_kps[:, 2] > 0)[0]
        if len(visible_indices) != len(transformed_kps):
            # 安全のため長さを合わせる
            n = min(len(visible_indices), len(transformed_kps))
            visible_indices = visible_indices[:n]
            transformed_kps = transformed_kps[:n]

        # 2Dガウシアン生成のためにグリッドを事前計算
        xs = np.arange(w_out)[None, :]
        ys = np.arange(h_out)[:, None]

        for i, kp_idx in enumerate(visible_indices):
            x, y = transformed_kps[i]
            # リサイズ後の座標系をヒートマップ座標系へ（整数に丸めず浮動で計算）
            center_x = x * (w_out / self.img_size[1])
            center_y = y * (h_out / self.img_size[0])

            # 範囲外はスキップ
            if not (0 <= center_x < w_out and 0 <= center_y < h_out):
                continue

            dist_sq = (xs - center_x) ** 2 + (ys - center_y) ** 2
            exponent = dist_sq / (2 * sigma**2)
            heatmap[kp_idx] = np.exp(-exponent)

        return heatmap

    def _generate_offset_for_size(self, transformed_kps, original_kps, size_hw: tuple[int, int]):
        """オフセットターゲットを生成する。

        - 出力: (2K, H, W) のオフセットマップ（dx, dy）
        - 併せて (K, H, W) のマスクを返す（そのキーポイントの存在位置のみ1）
        """
        H, W = int(size_hw[0]), int(size_hw[1])
        K = self.num_keypoints
        off = np.zeros((2 * K, H, W), dtype=np.float32)
        mask = np.zeros((K, H, W), dtype=np.float32)

        visible_indices = np.where(original_kps[:, 2] > 0)[0]
        if len(visible_indices) != len(transformed_kps):
            n = min(len(visible_indices), len(transformed_kps))
            visible_indices = visible_indices[:n]
            transformed_kps = transformed_kps[:n]

        for i, kp_idx in enumerate(visible_indices):
            x_img, y_img = transformed_kps[i]
            # 画像座標系から出力グリッド座標系へ
            gx = x_img * (W / self.img_size[1])
            gy = y_img * (H / self.img_size[0])
            ix = int(round(gx))
            iy = int(round(gy))
            if ix < 0 or ix >= W or iy < 0 or iy >= H:
                continue
            dx = float(gx - ix)
            dy = float(gy - iy)
            off[2 * kp_idx + 0, iy, ix] = dx
            off[2 * kp_idx + 1, iy, ix] = dy
            mask[kp_idx, iy, ix] = 1.0

        return off, mask
