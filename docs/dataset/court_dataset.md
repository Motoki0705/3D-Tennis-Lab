# コートデータセット仕様書

このドキュメントは、テニスコートのキーポイント検出タスクに使用されるデータセットの構造と仕様について詳述します。

## 1. データセット概要

本データセットは、テニスの試合映像から切り出されたフレーム画像と、それに対応するコートのキーポイント情報から構成されます。アノテーションはCOCO (Common Objects in Context) 形式に準拠しています。

- **画像数**: 8,841枚
- **アノテーション数**: 8,841件
- **データ格納先**:
  - **画像**: `data/processed/court/images/`
  - **アノテーションファイル**: `data/processed/court/annotation.json`

## 2. アノテーション形式 (COCO準拠)

アノテーションファイル (`annotation.json`) は、以下の主要なキーを持つJSONオブジェクトです。

- `images`: 画像に関する情報リスト
- `annotations`: アノテーション情報リスト
- `categories`: カテゴリ定義リスト

### 2.1. `images`

各要素は単一の画像を表すオブジェクトです。

- `id` (int): 画像のユニークID
- `file_name` (str): 画像ファイル名 (例: `PuXlxKdUIes_2450.png`)
- `width` (int): 画像の幅 (ピクセル)
- `height` (int): 画像の高さ (ピクセル)

### 2.2. `annotations`

各要素は単一のオブジェクト（このデータセットではコート）に対するアノテーションです。

- `id` (int): アノテーションのユニークID
- `image_id` (int): 対応する画像のID
- `category_id` (int): 対応するカテゴリのID (本データセットでは常に `1`)
- `keypoints` (list[int]): キーポイントのリスト `[x1, y1, v1, x2, y2, v2, ...]`
- `num_keypoints` (int): 可視（`v > 0`）なキーポイントの数

#### キーポイントの可視性 (`v`)

- `v = 0`: アノテーションなし（画像内に存在しない、またはラベル付けされていない）
- `v = 1`: 隠れている（Occluded）
- `v = 2`: 見えている（Visible）

### 2.3. `categories`

オブジェクトのカテゴリを定義します。本データセットには`court`という単一のカテゴリのみ存在します。

- `id` (int): カテゴリID (常に `1`)
- `name` (str): カテゴリ名 (`court`)
- `keypoints` (list[str]): 15個のキーポイントの名前リスト。インデックス順に定義されています。
- `skeleton` (list[list[int]]): キーポイント間の接続情報。線描画に使用します。

#### キーポイント定義

| インデックス | キーポイント名 |
| :--- | :--- |
| 0 | `Net-Post_Left` |
| 1 | `Net-Post_Right` |
| 2 | `Net-Top_Left` |
| 3 | `Net-Top_Right` |
| 4 | `T-Service_Left` |
| 5 | `T-Service_Right` |
| 6 | `Center-Service_Line` |
| 7 | `Sideline-Service_Left` |
| 8 | `Sideline-Service_Right` |
| 9 | `Baseline-T_Left` |
| 10 | `Baseline-T_Right` |
| 11 | `Baseline-Corner_Left` |
| 12 | `Baseline-Corner_Right` |
| 13 | `Sideline-Baseline_Corner_Left` |
| 14 | `Sideline-Baseline_Corner_Right` |

#### スケルトン定義

以下のキーポイントペアが接続され、コートのラインを形成します。
(例: `[1, 2]` は `Net-Post_Left` と `Net-Post_Right` を接続)

```json
[
  [1, 2], [1, 3], [2, 4], [3, 4], [5, 6], 
  [7, 8], [9, 10], [11, 12], [13, 14]
]
```

## 3. データセットの分析（参考）

`notebooks/court_data_analysis.ipynb` で実施された分析により、以下の点が明らかになっています。

- **キーポイントの欠損**: 全てのアノテーションが15個全てのキーポイントを持っているわけではありません。ヒストグラム分析により、アノテーションごとの有効なキーポイント数の分布が確認できます。
- **キーポイントの可視性**: 特定のキーポイント（例: 奥のコーナー）は、他のキーポイントに比べて「隠れている(occluded)」とラベル付けされる傾向があります。これはデータセットの撮影角度の偏りを示唆している可能性があります。
