# Git Branching Strategy for 3D Tennis Analysis System

このドキュメントは、「3Dテニス自動解析システム」開発プロジェクトにおけるGitのブランチ運用ルールを定義するものです。本戦略の目的は、実験的な開発を効率的かつ安全に進め、成功した成果のみを安定してシステムに統合することです。

## 基本方針

本プロジェクトは、コンポーネントごとに多数の実験的アプローチを試す開発スタイルを取ります。そのため、**一つの実験を一つのブランチで管理する「フィーチャーブランチ戦略」** を採用します。これにより、各実験の独立性を保ち、プロジェクト全体のコードベースを常にクリーンな状態に維持します。

---

## 主要ブランチ

プロジェクトには、常に存在する2つの主要ブランチがあります。

### 🌳 `main`

- **役割**: **本番ブランチ**。常に安定し、実行可能な、統合されたシステムの最新版を保持します。
- **ルール**: このブランチに直接コミットすることは**禁止**です。更新は、後述する`develop`ブランチからのマージによってのみ行われます。

### 🌿 `develop`

- **役割**: **開発の統合ブランチ**。個々の実験で成功し「卒業」した機能が、すべてここに集約されます。
- **ルール**: このブランチも直接のコミットは原則行いません。更新は、後述する実験ブランチからのPull Request (プルリクエスト) を通じて行われます。

---

## 実験ブランチ (`feature/...`)

開発作業の大半は、この実験ブランチで行います。

- **役割**: `development/`ディレクトリ内で行う、**個々の実験（例: 新しいボール追跡モデルの試作）を隔離するためのブランチ**です。

- **作成元**: 必ず`develop`ブランチから作成します。

- **命名規則**: どのプロジェクトの、どんな実験かを明確にするため、以下の規則で命名します。
  `feature/<project>/<experiment>`

  - **`<project>`**: 取り組む課題（例: `ball-tracking`, `court-pose`）。
  - **`<experiment>`**: 具体的な実験内容やアプローチ（例: `vit-heatmap-baseline`, `add-new-loss`）。
  - **フォーマット**: ブランチ名では、慣例的に**`kebab-case`**（ハイフン区切り）を使用します。

  **具体例:**

  - `feature/ball-tracking/vit-heatmap-baseline`
  - `feature/court-pose/hrnet-with-new-loss`
  - `feature/player-analysis/4d-humans-integration`

- **ライフサイクル**: 実験が成功すれば`develop`ブランチにマージされ、その役目を終えます。失敗した場合は、マージせずにそのまま破棄（削除）します。

### Hydra設定との連携

ブランチ名は、Hydraの設定と密接に関連します。これにより、ブランチと実験設定の対応が明確になります。

- ブランチ名の`<project>` (`ball-tracking`)は、`configs/config.yaml`の`project`キー (`ball_tracking`)に対応します。
- ブランチ名の`<experiment>` (`vit-heatmap-baseline`)は、`configs/config.yaml`の`experiment`キー (`vit_heatmap_baseline`)に対応します。
- **フォーマット**: ブランチ名では`kebab-case`、設定ファイル内では`snake_case`と、それぞれの慣例に従い使い分けます。

---

## 開発ワークフロー

新しい開発メンバーでも分かるように、具体的な作業手順を以下に示します。

### Step 1: 実験の開始

新しい実験を始めるには、まず`develop`ブランチを最新の状態にし、そこから命名規則に従った実験ブランチを作成します。

```bash
# 1. developブランチに移動して最新化する
git switch develop
git pull origin develop

# 2. 新しい実験ブランチを作成して移動する
git switch -c feature/player-analysis/4d-humans-integration
```

同時に、`development/player_analysis/`内に、対応する実験用ディレクトリを作成します。

### Step 2: 開発とコミット

作成したブランチで、モデル開発や学習などの試行錯誤を行います。作業の区切りが良い単位で、変更内容をコミットしてください。

```bash
# 作業ファイルをステージング
git add development/player_analysis/4d-humans-integration/

# コミットメッセージを添えてコミット
git commit -m "feat(player-analysis): 4D-Humansの基本骨格を実装"

# 定期的にリモートリポジトリにpushしてバックアップ
git push -u origin feature/player-analysis/4d-humans-integration
```

### Step 3: 成功した実験の統合（モデルの卒業）

実験が成功し、その成果をシステムに正式採用する準備が整ったら、`develop`ブランチへの統合プロセスを開始します。

1.  **Pull Requestの作成**:
    GitHub上で、自分の実験ブランチ (`feature/...`) から`develop`ブランチへの**Pull Request (PR)** を作成します。
2.  **レビュー**:
    PRの概要欄に「どんな実験で、どのような成果が出たか」を明確に記述します。チームメンバーは、このPR上でコードレビューを行い、問題がないかを確認します。

### Step 4: マージと後片付け

PRが承認されたら、`develop`ブランチにマージします。

1.  **マージ**: GitHub上でマージボタンをクリックします。
2.  **ブランチの削除**: `develop`への統合が完了した実験ブランチは不要になるため、削除します。これにより、リポジトリを常に整理された状態に保ちます。

<!-- end list -->

```bash
# ローカルのブランチを削除
git branch -d feature/player-analysis/4d-humans-integration

# リモートのブランチを削除
git push origin --delete feature/player-analysis/4d-humans-integration
```

### 目的別ブランチ命名規則

`feature/` 以外にも、以下のような接頭辞を導入することをお勧めします。

| 接頭辞          | 目的                                                          | 具体例                               |
| :-------------- | :------------------------------------------------------------ | :----------------------------------- |
| **`feature/`**  | 👨‍🔬 新しい機能の開発やモデルの実験                             | `feature/ball-tracking/new-model`    |
| **`analysis/`** | 📈 データ分析、調査、可視化                                   | `analysis/court-dataset/initial-eda` |
| **`docs/`**     | 📚 ドキュメントの追加・修正                                   | `docs/add-branching-strategy`        |
| **`fix/`**      | 🐛 バグの修正                                                 | `fix/player-detection-crash`         |
| **`chore/`**    | 🧹 リファクタリングやライブラリ更新など、機能や修正以外の雑務 | `chore/update-pytorch-version`       |
