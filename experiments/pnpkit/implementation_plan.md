# 0. ゴールとスコープ

- **ゴール**：`experiment/pnpkit/` を **Hydra ベース**で安定運用できる PnP 専用モジュールとして完成させ、
  コートの 2D–3D 対応から **\[R,t]（世界→カメラ）** と（必要に応じて）**K**をロバストに推定し、**JSONL** 出力を提供。
- **非ゴール**：HMR 側の前処理・後処理・可視化（将来接続は前提）。

---

# 1. ディレクトリと初期整備（P0）

## 1.1 構成・初期ファイル

- ツリー（前計画どおり）に **`__init__.py`（空）** を `experiment/` と `experiment/pnpkit/` 直下へ追加（import 安定化）。
- `cli.py` の先頭コメントを **「最小CLI（Hydra）」** に修正。
- `core/types.py`：

  - `PnPConfig.irls` を `field(default_factory=IRLSConfig)` に変更（可変既定値共有の回避）。

- `configs/intrinsics/default.yaml` を **インライン定義に統一**（`file:` を削除。`width/height/K/dist` のみ）。

## 1.2 パス解決・作業ディレクトリ

- `configs/hydra/default.yaml` で
  `hydra.run.dir = experiment/pnpkit/outputs/${now:%Y%m%d_%H%M%S}`, `hydra.job.chdir=true` を既定化。
- すべての外部パスは `to_absolute_path()` で解決することを **コーディング規約**として明記。

**受け入れ基準**

- `python -m experiment.pnpkit.cli` が空実装でも起動し、Hydra の run.dir に `resolved.yaml`（後述）を保存。

---

# 2. コンフィグ体系と I/O ポリシー（P0）

## 2.1 Hydra defaults の確定

- `configs/config.yaml` の `defaults` を確定：`pnp/base`, `irls/hard`, `autotune/global_small`, `intrinsics/default`, `court/itf`, `input/csv_example`, `hydra/default`。
- **`io.*` キーは廃止**し、読み込みは **`intrinsics/*`, `court/*`, `input/*`** の各グループキーを参照（重複を排除）。

## 2.2 Resolved 設定の保存

- `cli.main()` 冒頭で `OmegaConf.save(config=cfg, f="resolved.yaml")` を **run.dir に保存**（再現性担保）。

## 2.3 ログ

- `configs/hydra/job_logging.yaml` を追加（INFO 既定／RMSE・候補数・IRLS 反復回数をログ出力）。

**受け入れ基準**

- `resolved.yaml` が保存され、ログに **最終構成**と **主要メトリクス**が出力される。

---

# 3. ドメイン／I/O 層（P0）

## 3.1 Court モデル

- `domain/court.py`：

  - `load_court_3d(path_or_cfg)`：**数値ベタ書きのみ対応**（初期は式展開をしない）。`(N,3)` と `name→idx` を返す。
  - **平面前提**の 2D 射影補助：`to_plane_xy(points3d) -> (N,2)` を提供（Z を破棄して XY 抽出）。

## 3.2 Pairing

- `domain/pairing.py`：

  - `build_pairs(names2d, court_model)`：**並び対応**を構築し、`pts2d(N,2)/pts3d(N,3)` を返す。
  - 点数不足・重複は例外化。

## 3.3 Readers/Writers

- `io/readers.py`：

  - `load_intrinsics_yaml(path)`（将来の外部 K 読み込み用。現状は未使用）。
  - `load_court_yaml(path)`（`court_3d.yaml` の単純ローダ）。
  - `load_points2d_csv(path)`：`name,x,y[,score]` を `(N,2)`＋`names` で返す。
  - `load_listfile(path)`：1 行 1 パス。

- `io/writers.py`：

  - `append_jsonl(out_path, result)`・`write_jsonl(out_path, results)`。

- `io/viz.py`：後回し（P2）。

**受け入れ基準**

- CSV（2D）、YAML（court/intrinsics）、JSONL（出力）の往復が最小テストで成功。

---

# 4. 幾何・カメラ・誤差（P0）

## 4.1 Core/Geometry

- `project_points(rvec,tvec,K,dist,pts3d)` は `cv.projectPoints` に一本化。
- `reproj_errors(...)` と `reproj_stats(...)` を提供（mean/median/p90/p95/max）。

## 4.2 Core/Camera

- `invert_extrinsics(extr: Extrinsics) -> (R_wc, t_wc)` を実装（HMR 接続用ユーティリティ）。
- 将来の `undistort_points` は TODO コメントで留置。

**受け入れ基準**

- 既知の合成データで投影→逆投影の数値整合がテストで確認できる。

---

# 5. 推定モジュール（IPPE→LM→IRLS）とフォールバック（P0）

## 5.1 IPPE

- `calib/ippe.py`：

  - `solve_candidates(pts2d, pts3d, intr)`：`cv.solvePnPGeneric(..., SOLVEPNP_IPPE)`。
  - `score_candidate(...)`：RMSE ＋ ケイラリティ ＋ 事前（俯角など）。
  - `choose_best(...)`：もっとも良い候補を返す。

## 5.2 ホモグラフィ分解（重要：2D–2D）

- `calib/homography.py`：

  - `to_plane_xy(pts3d) -> ptsXY` を **必ず**使用し、`findHomography(ptsXY, pts2d)` → `decomposeHomographyMat(...)`。
  - 得られた候補に `score_candidate` を適用して選別。

## 5.3 LM

- `calib/refine_lm.py`：

  - `solve(init_pose, pts2d, pts3d, intr)`：`cv.solvePnPRefineLM` を安全ラップ（NaN/非収束防御）。

## 5.4 IRLS

- `calib/irls.py`：

  - `refine(pose, pts2d, pts3d, intr, IRLSConfig)`：
    `tau0 = tau0_ratio * diag_len` → 重み更新（hard/huber/tukey）→ 重み>0 の点で LM → `tau *= decay`（下限あり）× `max_iter`。
  - 最終 `(pose, inlier_mask)` を返す。

**受け入れ基準**

- 合成データで **IPPE 初期化→LM→IRLS** により RMSE が単調非増加、インライアが非拡大、最終 RMSE 改善。

---

# 6. オートチューン（K の粗探索）（P1）

## 6.1 Global / Per-image

- `calib/autotune.py`：

  - `global_search(sample_frames, intr_init, grid) -> Intrinsics`
    代表フレーム集合（下記 6.2）で `(fx,cx,cy)` を格子探索、平均 RMSE 最小の K を採用。
  - `per_image_search(pts2d, pts3d, intr_prev, grid) -> Intrinsics`
    そのフレームの小範囲探索で最良 K を採用。

## 6.2 サンプル選定

- `configs/autotune/*` に `sample` を追加（例：`{mode: "stride", step: 20}` または `{mode: "random", n: 30, seed: 42}`）。
- `api.estimate_pose_batch` 内でサンプル抽出→`global_search()`。

**受け入れ基準**

- 固定ズーム映像で global により K が安定、per-image でスケール揺れが抑制される（統計ログで確認）。

---

# 7. RANSAC 前段（任意・推奨）（P1）

- **選択肢A**：`calib/homography.py` に `ransac_filter(ptsXY, pts2d, ransac_px)` を実装し、2D–2D の H で外れ値を事前除去 → IPPE。
- **選択肢B**：`calib/ippe.py` 内で前処理として呼ぶフックを設置。

**受け入れ基準**

- ノイズ点を 10–20% 混入させた合成データで、RANSAC ありの方が RMSE/n_inliers が改善。

---

# 8. API／CLI とバッチ運用（P0）

## 8.1 API

- `api.estimate_pose_single(...)`：前述パイプライン（IPPE→LM→IRLS→統計）。
- `api.estimate_pose_batch(...)`：`autotune` を解釈し、global/per-image を実行。

## 8.2 CLI（Hydra）

- `@hydra.main(config_path="configs", config_name="config")`。
- `resolved.yaml` を run.dir に保存。
- シングル／バッチ実行、JSONL 書き出し。

**受け入れ基準**

- 代表的な Hydra オーバーライドで実行可能：
  `python -m experiment.pnpkit.cli input.csv_example=... irls=huber autotune.mode=global`

---

# 9. 失敗時の扱いとメタ情報（P0）

- 候補ゼロ／非収束／IRLS で点が尽きた場合：

  - `PnPResult.method = "FAIL"`、`meta.reason` に理由、`n_inliers=0`、RMSE は `null`。
  - バッチ処理は継続し、JSONL にも 1 行として記録。

**受け入れ基準**

- 異常入力でプロセスが落ちず、失敗行が JSONL に蓄積される。

---

# 10. テスト計画（P0→P1）

- **P0（最小）**

  - `test_ippe.py`：合成データで真値近傍の候補選択。
  - `test_refine_lm.py`：LM 後に誤差非増加。
  - `test_irls.py`：RMSE 改善、インライア非拡大。
  - `test_io_api.py`：CSV/YAML/JSONL の往復・不足項目例外。

- **P1（拡張）**

  - `test_autotune.py`：global/per-image の格子端・異常値。
  - `test_h_decomp.py`：2D–2D 前提での H 分解が動作。

**受け入れ基準**

- P0 テストがローカルでグリーン。P1 は実装後に追加でグリーン。

---

# 11. 可視化・運用（P2）

- `io/viz.py`：再投影点・残差ヒスト・インライア可視化。
- `visualize.py` スクリプト（任意）：run.dir の JSONL を読んで画像に描画。

---

# 12. リスクと対策

- **平面退化（ほぼ一直線）**：ペアリングで退化検出。必要数不足は即エラー。
- **歪み未補正**：`dist` を確実に `projectPoints` に渡す（未指定なら None）。
- **Hydra の chdir 迷子**：すべて `to_absolute_path()` を強制。
- **per-image K の振れ**：既定は `autotune.mode=global`。per-image は小範囲＋記録。

---

# 13. Definition of Done（完成条件）

1. `python -m experiment.pnpkit.cli` が **Hydra** で起動し、`resolved.yaml` を保存。
2. シングル／バッチで **JSONL 出力**（成功・失敗ともに記録）。
3. IPPE→LM→IRLS→H 分解フォールバックが実装・テスト済み。
4. global オートチューンが動作（代表フレームの選抜仕様を満たす）。
5. 主要ログ（RMSE、候補数、IRLS 反復、K 採用値）が出力。
6. P0 テストが全てグリーン。
