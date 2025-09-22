# DetectRunner 再実装ガイド

本ドキュメントは、新規メンバーが `third_party` 以下の実装を参照せずとも、`development/ball_tracking/hrnet/runners/detect.py` に相当する推論ランナーを構築できるようにすることを目的としています。DetectRunner が担う責務・入出力・依存モジュール・制御フローを粒度高く整理し、同等機能を別プロジェクトへ移植する際の手順を示します。

---

## 1. DetectRunner が解決する課題

1. **外部推論モジュールとのブリッジ**: Hydra 設定で選ばれる WASB 系ボール検出器（`build_detector`）を初期化し、正しいデバイス／チェックポイントで実行する。
2. **動画処理パイプラインの統括**: OpenCV を用いたフレーム取得、サイズ正規化、テンソル化、モデル入力整形（連続フレームのチャネル結合）を行い、推論可能なバッチを生成する。
3. **後段処理のオーケストレーション**: 推論結果（バッチ毎・フレーム毎のボール候補）をトラッカーへ渡し、座標を元解像度へマッピングし、描画付き動画として出力する。
4. **可搬性と再設定**: Hydra 設定（デバイス、スケール、チェックポイントなど）を尊重しつつ、GPU が無い環境では自動的に CPU フォールバックする。

---

## 2. 必要な外部インターフェース

DetectRunner は以下のコンポーネントを前提にします。再実装時はこれらと同等のインターフェースを用意してください。

| 役割       | 入口                                                             | 主な責務                                                                                       |
| ---------- | ---------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| Detector   | `from detectors import build_detector`                           | `run_tensor(input_tensor, trans_outputs)` を持つオブジェクトを返す。モデル初期化・推論を担当。 |
| Tracker    | `from trackers import build_tracker`                             | `update(predictions)` で最新フレームのボール位置を確定し、連続追跡する。                       |
| 画像変換   | `from utils.image import get_affine_transform, affine_transform` | アフィン行列の算出と座標変換を提供。                                                           |
| Hydra 設定 | `cfg.runner`, `cfg.model`, `cfg.detector`                        | 入力解像度・出力解像度・バッファサイズなど全パラメータを提供。                                 |

これらは DI（依存性注入）でもよく、モジュール配置はチーム規約に合わせて構いません。重要なのは、DetectRunner から呼び出す関数シグネチャを一致させることです。

---

## 3. DetectRunner のライフサイクル

### 3.1 初期化 (`__init__`)

1. `BaseRunner` の初期化を呼ぶ（ログ設定や共通プロパティを保持するため）。
2. Hydra 設定から希望デバイスを取得し、GPU が無い場合は警告を出しつつ `torch.device('cpu')` を採用する。
3. `build_detector(cfg)` / `build_tracker(cfg)` を呼び、インスタンスを生成する。
4. `torchvision.transforms.Compose` などで画像の正規化パイプラインを定義する（例: `ToTensor` と `Normalize(mean, std)`）。

### 3.2 実行 (`run(video_path, output_path)`)

1. **映像入力**: `cv2.VideoCapture` でフレーム幅・高さ・FPS・総フレーム数を取得し、`cv2.VideoWriter` で出力動画を準備する。
2. **前処理行列の算出**:
   - `get_transform((H, W), (inp_w, inp_h))` でモデル入力サイズへのアフィン行列 `trans_input` を求める。
   - `get_transform((H, W), (inp_w, inp_h), inv=1)` で逆行列 `trans_input_inv` を取得し、元解像度への逆変換に使う。
   - モデル出力ヒートマップのサイズ（例: `out_w`, `out_h`）についても `get_transform(..., inv=1)` を求め、ポストプロセッサへ渡す。
3. **フレームバッファ**: `collections.deque(maxlen=frames_in)` を用意し、推論に必要な連続フレーム数が揃うまで原画像をそのまま書き出す。
4. **前処理**:
   - 各フレームを `cv2.warpAffine(frame, trans_input, (inp_w, inp_h))` で正規化。
   - `self.img_transforms` へ通し、`torch.Tensor` 化と正規化を行う。
   - 直近 `frames_in` 枚を `torch.cat` でチャネル方向に結合し、`[1, frames_in * 3, inp_h, inp_w]` のテンソルを作る。
5. **推論**: `detector.run_tensor(input_tensor, trans_outputs)` を呼ぶ。`trans_outputs` は `{scale_name: torch.tensor(affine_matrix).unsqueeze(0)}` の辞書。
6. **トラッキング**: `batch_results` の最後のフレーム結果を取り出し、`tracker.update(preds_for_last_frame)` へ渡す。
7. **描画**: トラッカーから `{"x", "y", "visi"}` を受け取り、元解像度座標に対応する（WASB の出力は既に逆変換済み）。表示用に `int(round(x))` など整数化し、`cv2.circle` と `cv2.putText` で書き込む。
8. **後片付け**: ループ終了後に `cap.release()` / `out_writer.release()` を呼び、ログで出力パスを伝える。

### 3.3 エラーハンドリング

- 動画が開けない場合は `IOError` を送出。
- トラッカーが検出できない場合でも動画出力は継続し、`visi=False` のフレームはオーバーレイ無しで書き出す。
- CUDA が無い環境で `device='cuda'` が設定されている場合は、警告ログを出した上で CPU へ切り替える。

---

## 4. 再実装のためのステップバイステップ

1. **設定読み込み**: Hydra もしくは別の設定管理ツールで、以下の値を取得できるようにする。
   - `model.frames_in`, `model.inp_width`, `model.inp_height`, `model.out_width`, `model.out_height`, `model.out_scales`
   - `runner.device`, `detector.model_path`
2. **依存モジュールのインターフェース確立**: 上記の Detector / Tracker / 画像変換関数を用意する。
3. **DetectRunner クラスの骨組み**: `__init__`, `run`, `get_transform`（必要ならユーティリティとして外だし）を定義する。
4. **テスト実装**: 短い動画と既知のチェックポイントを用い、
   - フレームバッファが正しく維持されるか
   - 推論結果が返ってくるか
   - 座標描画がずれていないか
     を確認する。
5. **例外系の確認**: 入力動画が存在しない・CUDA 非対応環境・トラッキング対象が不在など、主要な失敗パターンを確認する。

---

## 5. 追加の設計判断

- **スケール追加**: 将来的に複数スケールの出力を扱う場合は、`trans_outputs` を複数エントリの辞書で渡し、ポストプロセッサ側の集約ロジックを更新する。
- **マルチランナー共存**: DetectRunner 以外のランナー（学習用など）と共通化できる部分は、親クラス `BaseRunner` やユーティリティモジュールへ切り出す。
- **描画ポリシー**: 速度の速いワークフローでは、描画を非同期に回し、解析メタデータのみ先に保存する設計も有効。

---

## 6. チェックリスト

DetectRunner と同等の機能を備えたら、以下を満たしているか確認してください。

- [ ] 入力動画→出力動画の変換がエラー無く完了する。
- [ ] フレームバッファが `frames_in` に達するまで原画像を素通しし、その後連続フレームを推論する。
- [ ] モデル結果が `tracker.update` へ渡され、可視フラグに応じた描画が行われる。
- [ ] CUDA が無い環境でも CPU に自動フォールバックし、推論が継続する。
- [ ] 構成値（パス、閾値など）が Hydra 設定のみで変更できる。

---

このガイドを参照すれば、`third_party` 以下の具体的な実装に触れなくても DetectRunner と同等の推論パイプラインを再構築できるはずです。必要に応じて、プロジェクト固有のログ体系や監視ツールとの連携を追加して運用に備えてください。
