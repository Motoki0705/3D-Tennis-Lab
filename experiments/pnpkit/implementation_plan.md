# 1) 座標系の取り決め（World / Camera / Image）

- **World（コート）座標系**：右手系。
  原点＝**コート中心**、+X＝向かって右のダブルスサイドライン方向、+Y＝相手コート方向（ネット直交）、+Z＝上向き。単位は **m**（ITF寸法）。
- **Camera座標系**：右手系。+Z＝**カメラ光軸前方**、+X＝右、+Y＝下（OpenCV準拠）。
- **Extrinsics の向き**：最適化内部は **World→Camera（R_cw, t_cw）** を採用。
  HMR/可視化等で **Camera→World（R_wc, t_wc）** が必要なら、`R_wc = R_cw^T`、`t_wc = -R_cw^T t_cw` で提供（変換ユーティリティを用意）。

# 2) コート幾何（既知3D）

- コートは **Z=0 の平面**。各キーポイントは（X, Y, 0）。
- 寸法は ITF 既知値を Yaml で一元管理（将来バリエーション対応可）。

# 3) カメラモデル（Intrinsics / Distortion）

- **投影**：ピンホール
  $K=\begin{bmatrix} f_x & s & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}$ ，**基本は skew s=0**。
- **歪み**：OpenCV Brown–Conrady（rational）
  $\mathbf{d} = (k_1,k_2,p_1,p_2,k_3[,k_4,k_5,k_6])$。
  単発推定では $k_1,k_2,p_1,p_2,k_3$ の5つを推奨。最初は **固定 or 0**、マルチフレーム束調整で解放。
- **最適化で動かす既定**

  - 既知Kあり：$(f_x,f_y,c_x,c_y)$ は固定 or 弱正則化で微調整。
  - 既知Kなし：$(f_x=f_y, c_x,c_y)$ を推定（s=0）。初期値は画像中心＆EXIF由来の見込みf。

- **単位**：画素（fx, fy）・画素座標（cx, cy）。

# 4) 観測モデルと誤差

- **観測**：画像上のキーポイント $\hat{\mathbf{u}}_i$（px）。可視フラグ/重みあり。
- **予測**：3D点 $\mathbf{X}_i=(X_i,Y_i,0)$ を

  1. 座標変換（World→Camera）、
  2. 正規化投影、
  3. **歪み適用**、
  4. 内部パラメタで画像座標へ。

- **損失**：**画素の再投影誤差**に **ロバスト損失（Huber/Tukey）**、点ごとの重み（信頼度）を掛ける。
- **品質指標**：RMSE(px)、外れ値率、推定カメラ高さ（t_wc.z）の物理妥当性、コート法線と光軸角度。

# 5) 初期化戦略（Planar前提）

- **Kがある程度既知**：

  - **IPPE**（planar PnP）で $(R_cw,t_cw)$ 初期化 → LMで微分最適化。

- **K未知/不確か**：

  - まず平面対応から **DLTでホモグラフィH** 推定 → **vanishing constraints** で $f$ と $(c_x,c_y)$ を初期推定（少なくとも $f$ と中心近傍仮定）。
  - そのKで **IPPE分解** → LMで $(R,t,K)$ 同時最適化。

- **マルチフレーム**：K共有、フレームごとに $(R,t)$。粗→精の順でパラメタ解放（K固定→中心固定→f解放→歪み解放）。

# 6) 退化と対策

- **平面＋未知K**はスケール/姿勢の曖昧性が出やすい →

  - 事前分布（例：カメラ高さ 1–20 m、ピッチ/ロールの範囲）を**弱正則化**で付与。
  - **fの下限・上限**を設定（例：0.5〜3×画像幅）。
  - 画面内のキーポイントは**広い分布**で（片側に偏らない）。
  - 初期は **歪み0固定** で安定化、のち解放。

# 7) データ記述（ファイル/I/O）

- **コート定義**: `configs/courts/tennis_itf.yaml`（キーポ名→(X,Y,0)m、スケルトンもここ）
- **検出2D**: `annotations/frames.jsonl`（1行1フレーム: frame_idx, image_path, {kp_name: \[u,v,vis]}）
- **カメラK**: `calib/intrinsics_<camera_id>.yaml`（fx,fy,cx,cy,dist, skew省略）

  - 画像名の **prefix** → camera_id にマップ（`cameras/index.yaml`）。

- **推定結果**:

  - 単発：`results/<video>/<frame>.yaml`（R_cw,t_cw,K,RMSE,meta）
  - 束調整：`results/<video>/bundle.yaml`（K共有、各frameの(R,t), 指標）

- **パス**はすべて **POSIXスラッシュ**で保存（Windowsでもinternally変換）。

# 8) 実装上の表現（クラスと最適化変数）

- **回転の内部表現**：最適化は **so(3)（3パラの指数写像）**。
  I/O は **クォータニオン** or **R行列**（数値安定＆直感の両立）。
- **状態ベクトルの既定**

  - 単発：$\theta = [\omega_x,\omega_y,\omega_z,\, t_x,t_y,t_z,\, f_x,f_y,c_x,c_y,\, k_1,k_2,p_1,p_2,k_3]$（必要に応じて固定/解放）。
  - 束調整：$\Theta = K \;\cup\; \{(\omega^{(f)},t^{(f)})\}_{f\in \mathcal{F}}$

- **最適化**：LM（SciPy `least_squares` or Ceres）。自動微分が使えるならCeres推奨。
  IRLSで外れ値抑制、点ごと重みは可視フラグや検出信頼度から。

# 9) ライブラリ選定

- **幾何/初期化**：OpenCV（`findHomography`, `solvePnPGeneric(IPPE)`）
- **最適化**：SciPy（手軽） or Ceres（スケール・自動微分・高速）。
- **設定管理**：Hydra + YAML（既存構成に合わせる）。
- **検証**：ユニットテスト（投影・歪み・Exp/Logmap・変換の往復一致、数値微分一致）。

---

## 最小コード骨子（型とユーティリティだけ）

```python
from dataclasses import dataclass
import numpy as np
import cv2

@dataclass
class Distortion:
    k1: float=0; k2: float=0; p1: float=0; p2: float=0; k3: float=0

@dataclass
class Intrinsics:
    fx: float; fy: float; cx: float; cy: float
    skew: float=0.0
    dist: Distortion=Distortion()

@dataclass
class PoseCW:  # World->Camera
    R: np.ndarray  # (3,3)
    t: np.ndarray  # (3,)

def rodrigues_exp(omega: np.ndarray) -> np.ndarray:
    R, _ = cv2.Rodrigues(omega.astype(np.float64))
    return R

def pose_from_omega_t(omega: np.ndarray, t: np.ndarray) -> PoseCW:
    return PoseCW(R=rodrigues_exp(omega), t=t)

def project_points_W(Xw: np.ndarray, K: Intrinsics, P: PoseCW) -> np.ndarray:
    # Xw: (N,3). Apply world->camera
    Xc = (P.R @ Xw.T + P.t.reshape(3,1)).T
    xn = Xc[:, :2] / Xc[:, 2:3]  # normalized
    x, y = xn[:,0], xn[:,1]
    r2 = x*x + y*y
    k1,k2,p1,p2,k3 = K.dist.k1, K.dist.k2, K.dist.p1, K.dist.p2, K.dist.k3
    x_d = x*(1+k1*r2+k2*r2*r2+k3*r2*r2*r2) + 2*p1*x*y + p2*(r2+2*x*x)
    y_d = y*(1+k1*r2+k2*r2*r2+k3*r2*r2*r2) + p1*(r2+2*y*y) + 2*p2*x*y
    u = K.fx*x_d + K.skew*y_d + K.cx
    v = K.fy*y_d + K.cy
    return np.stack([u,v], axis=1)

def invert_pose(P: PoseCW):
    Rwc = P.R.T
    twc = -Rwc @ P.t
    return Rwc, twc
```

---

## まとめ：今回の“選定”の要点

- **平面＋既知寸法**を最大限活用：DLT→IPPE→LM、複数フレームは束調整。
- **回転は so(3)**、**Kは最初は保守的に**（s=0、中心は画像中心）→段階的に解放。
- **歪みは後回し**、ロバスト誤差＋弱正則化で安定化。
- **I/OはYAML/JSONで固定スキーマ**、prefix→camera_id で K を引く。
- **内部は World→Camera、I/Oで双方向提供**。

いいですね。拡張・保守・再利用を最優先に、\*\*パイプライン（stageをつないで束ねる）\*\*前提のディレクトリと最小骨子を示します。
（Hydra / YAML 設定、OpenCV/SciPy 最適化、既存の `tools/court_annotator` とも親和的な形）

# 目次

- 全体ディレクトリ構成（木）
- 各ディレクトリの役割
- パイプライン設計（Stage I/F、Bundle契約）
- 最小ランナ例（Hydra + Registry）
- 代表コンフィグ例（単発 / 束調整）
- I/Oスキーマ（ファイル形式）
- テスト方針

---

# 全体ディレクトリ構成（提案）

```
pnpkit/
├─ apps/
│  ├─ single_frame_runner.py        # 1フレーム推定エントリ
│  └─ bundle_adjust_runner.py       # マルチフレーム束調整エントリ
│
├─ configs/
│  ├─ defaults.yaml                 # 共通デフォルト
│  ├─ pipeline/
│  │  ├─ single_frame.yaml          # 単発パイプライン構成
│  │  └─ bundle_adjust.yaml         # 束調整パイプライン構成
│  ├─ cameras/
│  │  ├─ index.yaml                 # prefix→camera_id 対応
│  │  └─ intrinsics_demo.yaml       # fx,fy,cx,cy,dist...
│  ├─ courts/
│  │  └─ tennis_itf.yaml            # 3Dコート定義（Z=0）
│  ├─ data/
│  │  ├─ demo_single.yaml           # 2D観測の参照（jsonlなど）
│  │  └─ demo_sequence.yaml         # 連番/複数フレーム
│  └─ optim/
│     ├─ lm_robust.yaml             # LM/ロバスト損失/正則化
│     └─ ceres_like.yaml            # 将来用の別プリセット
│
├─ data/                            # 入力データ（git外管理推奨）
│  ├─ annotations/                  # 2D観測 (jsonl等)
│  └─ images/
│
├─ results/                         # 出力（推定・ログ・可視化）
│
├─ src/
│  ├─ core/
│  │  ├─ camera.py                  # Intrinsics/Distortion/Pose 型
│  │  ├─ geometry.py                # 投影/歪み/Exp-Log/変換
│  │  ├─ court.py                   # コート3Dモデル管理
│  │  ├─ losses.py                  # Huber/Tukey、正則化
│  │  └─ io_schema.py               # I/Oバリデーション（pydantic等）
│  │
│  ├─ io/
│  │  ├─ reader.py                  # 2D観測/画像/メタの読込
│  │  ├─ writer.py                  # 推定結果/中間の保存
│  │  └─ viz.py                     # 画像への投影描画・可視化
│  │
│  ├─ pipeline/
│  │  ├─ base.py                    # Stage抽象 & Registry & Bundle
│  │  ├─ stages/
│  │  │  ├─ s00_load_inputs.py      # 観測/コート/初期Kをロード
│  │  │  ├─ s10_init_homography.py  # 平面H初期化（DLT, RANSAC）
│  │  │  ├─ s20_estimate_k.py       # K推定/微調整（必要時）
│  │  │  ├─ s30_ippe_init.py        # IPPEで(R,t)初期化
│  │  │  ├─ s40_refine_lm.py        # LMで(R,t,K,dist)同時最適化
│  │  │  ├─ s50_evaluate.py         # RMSE/角度/外れ値率評価
│  │  │  ├─ s60_export.py           # 結果保存（YAML/NPZ/可視化）
│  │  │  └─ s70_bundle_adjust.py    # マルチフレーム束調整
│  │  └─ utils.py                   # 共有ヘルパ（重み付け等）
│  │
│  └─ utils/
│     ├─ logging.py                 # 構造化ログ設定
│     ├─ paths.py                   # POSIXパスユーティリティ
│     └─ hydra_tools.py             # instantiate/シード等
│
├─ tests/
│  ├─ test_geometry.py              # 投影/逆投影/数値微分一致
│  ├─ test_pipeline_contract.py     # Bundle契約の整合
│  └─ test_stages.py                # 各stageの単体テスト
│
├─ scripts/
│  ├─ export_intrinsics_template.py
│  └─ draw_projection_demo.py
│
└─ third_party/                     # 外部コード（submodule推奨）
```

---

# 各ディレクトリの役割（要点）

- **apps/**：実行エントリ。Hydraの`@hydra.main`で設定を受け取り、`pipeline/base.py`の`build_and_run()`に委譲。
- **configs/**：Hydraコンフィグの分割（パイプライン、カメラ、データ、最適化）。組合せで再現性を担保。
- **src/core/**：幾何の純粋関数群と型（`dataclass|pydantic`）。副作用なし。最も再利用される層。
- **src/io/**：読み書き・可視化。ストレージ形式の変更はここに閉じ込める。
- **src/pipeline/**：Stage実装・Bundle契約・Registry管理。処理順を設定で切替可能。
- **tests/**：ユニット＋契約テスト。幾何は数値微分で検証、パイプラインは最小ダミーデータで回す。
- **results/**：日時/実験名で階層化。YAML/JSON/NPZと可視化画像、ログをまとめて保存。

---

# パイプライン設計（Stage I/F、Bundle契約）

### Bundle 契約（最低限のキー）

```python
# src/pipeline/base.py（一部）
from dataclasses import dataclass, field
import numpy as np
from typing import Dict, Any, List, Optional

@dataclass
class FrameObs:
    frame_idx: int
    image_path: str
    # 観測: 名前→(u,v,vis,weight)
    keypoints: Dict[str, tuple[float, float, int, float]]

@dataclass
class Bundle:
    court3d: Dict[str, np.ndarray]           # name->(3,), Z=0
    K: Optional[Dict[str, Any]] = None       # fx,fy,cx,cy,dist...
    pose_cw: Optional[Dict[str, Any]] = None # R,t  (world→camera)
    frames: List[FrameObs] = field(default_factory=list)
    matches: Optional[List[tuple[str,str]]] = None  # (kp_name, kp_name) 等
    H: Optional[np.ndarray] = None           # 平面ホモグラフィ初期値
    report: Dict[str, Any] = field(default_factory=dict)  # 指標、履歴
```

### Stage 抽象 & Registry

```python
class Stage:
    def __init__(self, **cfg): self.cfg = cfg
    def __call__(self, B: Bundle) -> Bundle: return self.run(B)
    def run(self, B: Bundle) -> Bundle: raise NotImplementedError

REGISTRY = {}
def register(name):
    def _wrap(cls):
        REGISTRY[name] = cls
        return cls
    return _wrap
```

- すべての Stage は **`Bundle`を入力→`Bundle`を返す**。
- **副作用なし**が理想（保存は `s60_export` に集約）。
- **キー名は固定**（契約）。ステージは不足キーがあれば明示エラー。

---

# 最小ランナ例（Hydra + Registry）

```python
# apps/single_frame_runner.py
import hydra
from omegaconf import DictConfig, OmegaConf
from src.pipeline.base import Bundle, REGISTRY
from src.utils.hydra_tools import instantiate_stages

@hydra.main(config_path="../configs", config_name="pipeline/single_frame.yaml", version_base=None)
def main(cfg: DictConfig):
    stages = instantiate_stages(cfg.pipeline.stages)  # list[Stage]
    B = Bundle(court3d={}, frames=[])
    for s in stages:
        B = s(B)
    # ここでB.reportを標準出力
    print(OmegaConf.to_yaml(cfg))
    print(B.report)
if __name__ == "__main__":
    main()
```

```python
# src/utils/hydra_tools.py
from hydra.utils import instantiate

def instantiate_stages(stage_cfg_list):
    stages = []
    for sc in stage_cfg_list:
        # sc: {_target_: "module.Class", param1: ..., name: "s10_init_homography"}
        stages.append(instantiate(sc))
    return stages
```

---

# 代表コンフィグ例

### 単発推定（`configs/pipeline/single_frame.yaml`）

```yaml
defaults:
  - /courts: tennis_itf
  - /cameras: intrinsics_demo
  - /data: demo_single
  - /optim: lm_robust
  - _self_

pipeline:
  stages:
    - _target_: src.pipeline.stages.s00_load_inputs.LoadInputs
      court_cfg: ${courts}
      data_cfg: ${data}
      camera_cfg: ${cameras}
    - _target_: src.pipeline.stages.s10_init_homography.InitHomography
      ransac: { reproj_px: 2.0, iters: 2000, conf: 0.999 }
    - _target_: src.pipeline.stages.s20_estimate_k.EstimateK
      enable: false # 既知Kならfalse
      principal_lock: true
    - _target_: src.pipeline.stages.s30_ippe_init.IPPEInit
    - _target_: src.pipeline.stages.s40_refine_lm.RefineLM
      refine:
        {
          R: true,
          t: true,
          fx: false,
          fy: false,
          cx: false,
          cy: false,
          dist: false,
        }
      robust: ${optim.robust}
      reg: ${optim.reg}
    - _target_: src.pipeline.stages.s50_evaluate.Evaluate
    - _target_: src.pipeline.stages.s60_export.Export
      out_dir: ${oc.env:RUN_DIR,results}/${now:%Y%m%d_%H%M%S}
```

### 束調整（`configs/pipeline/bundle_adjust.yaml`）

```yaml
defaults:
  - pipeline/single_frame@base: # 単発を継承
  - _self_

pipeline:
  stages:
    - ${base.pipeline.stages[0]} # LoadInputs
    - ${base.pipeline.stages[1]} # InitHomography (各フレーム初期化)
    - ${base.pipeline.stages[3]} # IPPEInit
    - _target_: src.pipeline.stages.s70_bundle_adjust.BundleAdjust
      refine: { R: true, t: true, K_shared: true, dist: true }
      schedule:
        - { unlock: fx, iter: 50 }
        - { unlock: dist, iter: 100 }
    - ${base.pipeline.stages[5]} # Evaluate
    - ${base.pipeline.stages[6]} # Export
```

---

# I/Oスキーマ（ファイル形式）

- **コート定義** `configs/courts/tennis_itf.yaml`

  - `keypoints: {name: [X,Y,0]}`、`skeleton: [[i,j],...]`、単位m

- **2D観測** `data/annotations/*.jsonl`

  - 1行/1フレーム：`{"frame_idx": 123, "image_path": "...", "keypoints": {"far_doubles_corner_left":[u,v,vis,w], ...}}`
  - パス保存は **POSIXスラッシュ**（Windowsは入出力時に変換）

- **カメラ内部** `configs/cameras/intrinsics_*.yaml`

  - `fx, fy, cx, cy, dist: {k1,k2,p1,p2,k3}`、`skew: 0` 省略可
  - `configs/cameras/index.yaml` に `prefix -> camera_id` 対応

- **結果** `results/.../bundle.yaml`

  - `K`, `poses: [{frame_idx, R, t, rmse_px, inlier_ratio,...}]`, `metrics: {...}`

---

# ステージ実装の要点（例）

- **s00_load_inputs**

  - court YAML→`Bundle.court3d`、2D jsonl→`Bundle.frames`、K YAML→`Bundle.K`
  - 可視/重みを正規化（未検出は `vis=0, w=0`）

- **s10_init_homography**

  - Z=0 平面対応で `findHomography(RANSAC)` → `Bundle.H`
  - Hから **2つのIPPE候補**の初期姿勢に備え、保持

- **s20_estimate_k**（任意）

  - 消失点/直交性から f を粗推定、`cx,cy` は画像中心固定など

- **s30_ippe_init**

  - OpenCV `solvePnPGeneric(..., SOLVEPNP_IPPE)` で (R,t) 初期化

- **s40_refine_lm**

  - 目的関数＝再投影 + ロバスト損失 + 弱正則化（高さ/ロール/ピッチ/f 範囲）
  - 逐次解放（R,t→f→dist）を引数で切替

- **s50_evaluate**

  - RMSE(px), 外れ値率, カメラ高さ（t_wc.z）, コート法線と光軸角

- **s60_export**

  - YAML/NPZ保存、画像に投影可視化（`io/viz.py`）
  - すべて `results/` 以下に集約

- **s70_bundle_adjust**

  - 共有K、各フレーム(R,t) の同時最適化
  - スケジュールでパラメタ解放を段階化

---

# テスト方針（抜粋）

- **幾何**：`core/geometry.py` の投影・逆変換が数値微分で一致（Eps差）。
- **契約**：Stage間で `Bundle` キーが欠けないこと（`test_pipeline_contract.py`）。
- **ステージ**：ダミーデータで `s10→s40` がRMSEを確実に下げること。
- **I/O**：POSIXパスで保存→Windows/Unixで読み戻し一致。

---

## これで得られるメリット

- **Stageの増減・差し替えが設定のみで可能**（Hydra + Registry）。
- 幾何コアは**副作用ゼロ**でユニットテスト容易。
- I/Oや可視化の変更が**上位に波及しない**。
- 単発→束調整まで**同じ契約で拡張**できる。
