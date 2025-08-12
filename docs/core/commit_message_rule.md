# Git Commit Message Guidelines

このドキュメントは、「3Dテニス自動解析システム」プロジェクトにおけるGitのコミットメッセージに関する規約を定めます。本プロジェクトでは、コミット履歴の可読性を高め、変更内容の追跡を容易にするため、**Conventional Commits**の仕様に準拠します。

-----

## 基本構造

すべてのコミットメッセージは、以下の構造を持つ必要があります。

```
<type>(<scope>): <subject>

<body>

<footer>
```

  * **Header**: 必須。1行で変更の概要を示します。
  * **Body**: 任意。変更の背景や詳細な説明を記述します。
  * **Footer**: 任意。破壊的変更や関連するIssue番号などを記述します。

-----

## Headerの構成要素

### `<type>`: 変更の種類 (必須)

コミットがどのような種類の変更なのかを、以下のキーワードで示します。

| type         | 説明                                                              |
| :----------- | :---------------------------------------------------------------- |
| **`feat`** | ✨ 新しい機能の追加（モデルの追加など）                             |
| **`fix`** | 🐛 バグの修正                                                     |
| **`docs`** | 📚 ドキュメントのみの変更                                         |
| **`style`** | 🎨 コードの動作に影響しない、スタイルの修正（空白、フォーマットなど） |
| **`refactor`** | ♻️ バグ修正でも機能追加でもない、コード構造の変更                 |
| **`test`** | ✅ テストコードの追加・修正                                       |
| **`chore`** | 🧹 ビルドプロセスやライブラリ更新など、開発補助に関する変更         |

### `<scope>`: 変更の範囲 (強く推奨)

変更が影響を及ぼすプロジェクトの範囲を示します。これにより、どのコンポーネントに関する変更かが一目で分かります。

  * **主要なscope**:
      * `court_pose`: コート姿勢推定
      * `player_analysis`: プレーヤー解析
      * `ball_tracking`: ボール追跡
      * `event_detection`: イベント検出
      * `system`: システム全体に関わる変更
      * `deps`: 依存ライブラリの更新
      * `ci`: CI/CD設定の変更

### `<subject>`: 変更内容の要約 (必須)

変更内容を50文字以内で簡潔に記述します。

  * **動詞の原形**で始める (例: `add`, `fix`, `change`)
  * 文頭は**小文字**で始める
  * 文末にピリオド (`.`) は付けない

-----

## BodyとFooter (任意)

Headerだけでは説明が不十分な場合に、1行の空白を挟んで詳細を記述します。

  * **Body**: 「なぜ」この変更が必要だったのか、そして「どのように」実装したのかを具体的に説明します。
  * **Footer**: `BREAKING CHANGE:` のように、互換性のない変更を明記したり、`Closes #123` のように関連するIssueを記述したりします。

-----

## プロジェクトにおける具体例

**例1: 新機能の追加**

```
feat(ball_tracking): add transformer-based end-to-end model
```

**例2: バグ修正**

```
fix(court_pose): correct keypoint projection logic for wide-angle video
```

**例3: ドキュメントの更新**

```
docs(project): add git branching and commit message guidelines
```

**例4: 依存ライブラリの更新**

```
chore(deps): update pytorch to version 2.5
```

**例5: 詳細な説明と破壊的変更を含むコミット**

```
refactor(player_analysis): change output format from keypoints to 3d mesh

The previous keypoint-based output was insufficient for advanced
downstream tasks like gait analysis. This change refactors the
player analysis module to output a full 3D mesh, providing richer
information for future development.

BREAKING CHANGE: The output format for player pose has changed.
All downstream modules that consume player data must be updated
to handle the new 3D mesh format instead of the old keypoint array.
```