# PnP Demo (Tennis Court Keypoints, COCO)

このスターターは、提示いただいた **COCO形式のコート・キーポイント** データから、
**OpenCVのPnP**（`solvePnPRansac`）でカメラ外部姿勢 \(R,t\) を推定し、
**再投影オーバレイ**（予測投影 vs GT 2D）と**誤差CSV**を生成します。

## 1. 依存関係

ローカル環境で以下をインストールしてください。

```bash
pip install opencv-python-headless numpy pyyaml pandas
```

## 2. 3D基準点（`court_spec.yaml`）

- ITF準拠の寸法をベースに、15キーポイントへ**推定対応**を与えています。
- **Left/Right = Y負/正** としています。
- 命名や対応がデータセットと異なる場合は、このファイルを編集して合わせてください。
  **ここが一致しないとPnPは正しく動作しません。**

## 3. 実行例

```bash
python experiments/pnp/pnp_demo.py \
  --ann data/processed/court/annotation_pruned.json \
  --img-root data/processed/court/images \
  --court-spec experiments/pnp/court_spec.yaml \
  --out experiments/pnp/out_pnp_demo \
  --max-images 10 \
  --name-source coco \
  --fx 1600 --fy 1600 --cx auto --cy auto \
  --ransac-reproj-thresh 5.0 \
  --method IPPE
```

- `--cx --cy` を `auto` にすると、画像中心を主点に設定します。
- `--method` は `EPNP | ITERATIVE | P3P | AP3P | IPPE` から選択可能です。
  - **平面点のみ**を使う場合は `--method IPPE` が有利ですが、
    実装上 `solvePnPGeneric(SOLVEPNP_IPPE)` を内部で自動切替します。
  - 平面でない点（例：Net-Top）も含めるなら `EPNP` が安定です。
- 出力：
  - `out_pnp_demo/overlays/*.png`：入力画像に **GT(黄)・投影(緑)** を重ねた可視化
  - `out_pnp_demo/results.csv`：画像ごとの誤差・姿勢サマリ

## 4. CSVのカラム

- `num_points`：使用できた2D-3D対応点（v>0 かつ specに存在）
- `num_inliers`：RANSACインライア数
- `mean_err_px / median_err_px / mean_err_inliers_px`：再投影誤差（px）
- `rot_mag_deg`：ロドリゲスベクトルのノルム（角度）
- `tvec_*_m`：並進ベクトル（メートル）。**Kに依存**してスケール感は変わります。

## 5. よくある調整ポイント

1. **court_spec.yamlの対応**：Left/Rightや「Baseline-T」の定義がズレると、
   重ね図がずれます。`overlays` を見ながら修正してください。
2. **内部パラメータK**：`fx, fy, cx, cy` を調整（画像の実焦点距離に近づける）。
3. **RANSAC閾値**：`--ransac-reproj-thresh` を 2–8px 程度で試行。
4. **方法**：平面点が多い場合は `--method IPPE` を試す。

## 6. 次のステップ（学習への接続）

- 本PnP結果を **初期姿勢**として、前段で議論した **ΔPose学習（微小オフセット学習）** に接続できます。
- 本スクリプトを前処理として走らせ、画像ごとに \((R*{\mathrm{pnp}}, t*{\mathrm{pnp}})\) と
  インライア・マスクをキャッシュ → 学習時の**初期値**や**重み付け**に利用します。

python experiments/pnp/make_id_visuals.py \
 --ann data/processed/court/annotation_pruned.json \
 --img-root data/processed/court/images \
 --court-spec experiments/pnp/court_spec.yaml \
 --out experiments/pnp/out_id_visuals \
 --name-source spec \
 --draw-names \
 --max-images 0

python experiments/pnp/pnp_ui_trackbar.py --court-spec experiments/pnp/court_spec.yaml --size 1920,1080
