# filename: development/court_pose/01_vit_heatmap/dataset.py
import json
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset


class CourtKeypointDataset(Dataset):
    """
    テニスコートキーポイントデータセット。

    - 画像とアノテーションを読み込む。
    - キーポイント座標からガウシアンヒートマップを生成する。
    - データ拡張を適用する。
    """

    def __init__(self, img_dir, annotation_file, img_size, heatmap_size, sigma, transform=None):
        self.img_dir = Path(img_dir)
        self.img_size = tuple(img_size)
        self.heatmap_size = tuple(heatmap_size)
        self.sigma = sigma
        self.transform = transform
        self.num_keypoints = 15

        with open(annotation_file) as f:
            coco_data = json.load(f)

        self.image_info = {img["id"]: img for img in coco_data["images"]}
        self.annotations = coco_data["annotations"]

        # keypointsの定義 (デバッグや可視化用)
        self.keypoint_names = coco_data["categories"][0]["keypoints"]
        self.skeleton = coco_data["categories"][0]["skeleton"]

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        image_id = annotation["image_id"]
        image_meta = self.image_info[image_id]

        img_path = self.img_dir / image_meta["file_name"]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        original_h, original_w, _ = image.shape

        keypoints = np.array(annotation["keypoints"]).reshape(-1, 3)

        # 可視性フラグ v=0 のキーポイントは(0,0)とし、データ拡張の対象外にする
        visible_keypoints = keypoints[keypoints[:, 2] > 0, :2]

        # データ拡張
        if self.transform:
            transformed = self.transform(image=image, keypoints=visible_keypoints)
            image = transformed["image"]
            transformed_keypoints = transformed["keypoints"]
        else:
            # transformがない場合でもリサイズとテンソル化は行う
            resize_transform = A.Compose([
                A.Resize(height=self.img_size[0], width=self.img_size[1]),
                ToTensorV2(),
            ])
            transformed = resize_transform(image=image)
            image = transformed["image"]

            # キーポイントも手動でスケール
            scale_x = self.img_size[1] / original_w
            scale_y = self.img_size[0] / original_h
            visible_keypoints = visible_keypoints.astype(np.float32)  # または np.float32
            visible_keypoints[:, 0] *= scale_x
            visible_keypoints[:, 1] *= scale_y
            transformed_keypoints = visible_keypoints

        # ヒートマップ生成
        target_heatmap = self.generate_heatmap(transformed_keypoints, keypoints)

        # imageは float32 に、heatmapは float32 に
        image = image.float() / 255.0 if image.dtype == torch.uint8 else image
        target_heatmap = torch.from_numpy(target_heatmap).float()

        return image, target_heatmap

    def generate_heatmap(self, transformed_kps, original_kps):
        """キーポイント座標からガウシアンヒートマップを生成する"""
        heatmap = np.zeros((self.num_keypoints, self.heatmap_size[0], self.heatmap_size[1]), dtype=np.float32)

        scale_x = self.heatmap_size[1] / self.img_size[1]
        scale_y = self.heatmap_size[0] / self.img_size[0]

        # transformed_kpsは可視キーポイントのみなので、元のインデックスに戻す
        visible_indices = np.where(original_kps[:, 2] > 0)[0]

        for i, kp_idx in enumerate(visible_indices):
            kp = transformed_kps[i]
            x, y = kp

            # ヒートマップ座標系に変換
            center_x = int(x * scale_x)
            center_y = int(y * scale_y)

            # 範囲外のキーポイントはスキップ
            if not (0 <= center_x < self.heatmap_size[1] and 0 <= center_y < self.heatmap_size[0]):
                continue

            # ガウシアンカーネル生成
            x, y = np.meshgrid(np.arange(self.heatmap_size[1]), np.arange(self.heatmap_size[0]))
            dist_sq = (x - center_x) ** 2 + (y - center_y) ** 2
            exponent = dist_sq / (2 * self.sigma**2)
            heatmap[kp_idx] = np.exp(-exponent)

        return heatmap
