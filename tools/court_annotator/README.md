# Court Annotator (Current Implementation)

最終更新: 2025-08-24 / バージョン 0.1（実装準拠）

このツールはテニスコートの15キーポイントを半自動で注釈します。OpenCV ベースのインタラクティブ UI、ホモグラフィ推定による自動補完、COCO 形式エクスポート、最小限のロギングに対応しています。複数動画の連続アノテーションと単一出力（集約）に対応しています。

## 機能

- ホモグラフィ（RANSAC）推定と自動補完（skip/lock を尊重）
- インタラクティブ UI（配置・ドラッグ、HUD 表示、Undo/Redo、フレームリセット）
- 画像保存（そのフレームで初回注釈時のみ）
- COCO 形式エクスポート（単一 `out_dir/ann.json` に集約、`out_dir/meta.json`）
- ログ出力（`out_dir/annotator.log`）

## セットアップ

- 主要依存: `opencv-python`, `numpy`, `hydra-core`, `omegaconf`
- 実行: `python -m tools.court_annotator`

## 設定

`tools/court_annotator/configs/annotator.yaml`（主要）と `tools/court_annotator/configs/ui.yaml`（UI バインドと色）、`tools/court_annotator/configs/thresholds.yaml`（幾何閾値）を使用します。

主な項目（annotator.yaml）

- `video_dir`: 入力動画のあるディレクトリ
- `out_dir`: 出力ベースディレクトリ（単一の `images/`, `ann.json`, `meta.json`, `resume.json` を生成）
- `court`: コート仕様 YAML
- `thresholds`, `ui`: 参照 YAML
- `start`, `end`, `stride`: 範囲と既定ステップ
- `nav.arrow_step`, `nav.confirm_step`: フレーム移動のスキップ幅（矢印、confirm 用）
- `discover.exts`, `discover.sort`: 対象拡張子と並び順（name|mtime）
- `preprocess.display_scale`: 表示縮尺（表示とヒットテストに同時適用）
- `logging.level`, `logging.file`: ログ出力設定

## キーバインド（既定、`ui.yaml` で上書き可）

- 保存: `s`, `enter`（`save_annotations`）
- 終了（保存付）: `q`（`quit`）
- 前/次フレーム: `←`/`→`（`prev_frame`/`next_frame`）
- フォーカス移動: `a`/`d`（`focus_prev`/`focus_next`）
- 前フレームからコピー: `space`（`copy_prev`）
- 再計算: `r`（`recompute`）
- Undo/Redo: `Ctrl+Z`/`Ctrl+Y`（`undo`/`redo`）
- フレームリセット: `Ctrl+R`（`reset_frame`）
- 動画完了: `Ctrl+D`（`mark_done`。現在の動画を完了として次の動画へ）

備考:

- 矢印キーは OS 差異に強い複数コードを内部で対応。
- `confirm` を使う場合は `ui.yaml` の `binds.confirm` にキーを割当（READY の時に `nav.confirm_step` 分進む）。

## 操作

- 左クリックでフォーカス中の点を配置、ドラッグで移動。初回注釈時のみ画像を保存します。
- `copy_prev` は未入力スロットのみ直前フレームから座標/可視/skip/lock をコピー（上書きしない）。
- `recompute` は現在入力から推定を再実行（補完は毎フレーム実行）。
- `undo`/`redo` はフレーム内の編集に対して動作。`reset_frame` はフレーム状態を完全クリア（Undo 可）。
- フレーム移動: `prev_frame`/`next_frame` は `nav.arrow_step`、`confirm` は `nav.confirm_step` を使用。

## 出力

- 画像: `out_dir/images/<token>_<frame:06d>(-<video_id>).jpg`（初回注釈時）
- COCO: 単一 `out_dir/ann.json`（全動画の `images`/`annotations` を集約、image_id はグローバル一意）
- メタ: 単一 `out_dir/meta.json`（動画ごとの fps/nframes/サイズなどを列挙）
- レジューム: `out_dir/resume.json`（進捗と image_id 対応表）

## 既知の制限

- ミニマップ、線スナップ、アウトライア強調は未実装
- ヘッドレス一括処理は未実装
