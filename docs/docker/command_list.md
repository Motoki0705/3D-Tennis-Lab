## Docker Compose コマンド一覧

### **基本的な操作**

| コマンド | 説明 |
| :--- | :--- |
| `docker compose up` | コンテナをビルド、(再)作成、起動し、アタッチします。 `-d` を付けるとバックグラウンドで起動します。 |
| `docker compose down` | コンテナ、ネットワーク、ボリューム、イメージを停止・削除します。 |
| `docker compose start` | 停止しているコンテナを起動します。 |
| `docker compose stop` | 実行中のコンテナを停止します。 |
| `docker compose restart` | 実行中のコンテナを再起動します。 |
| `docker compose pause` | 実行中のコンテナを一時停止します。 |
| `docker compose unpause`| 一時停止中のコンテナを再開します。 |

---

### **状態確認**

| コマンド | 説明 |
| :--- | :--- |
| `docker compose ps` | 実行中のコンテナの一覧を表示します。 |
| `docker compose top` | 実行中のコンテナのプロセスを表示します。 |
| `docker compose logs` | コンテナのログ出力を表示します。 `-f` を付けるとリアルタイムで追跡します。 |
| `docker compose events`| コンテナのリアルタイムイベントを表示します。 |
| `docker compose config`| `docker-compose.yml` の設定を検証し、表示します。 |

---

### **実行と管理**

| コマンド | 説明 |
| :--- | :--- |
| `docker compose exec [service] [command]` | 指定したサービス（コンテナ）内でコマンドを実行します。 (詳細は後述) |
| `docker compose run [service] [command]` | 指定したサービスで一度限りのコマンドを実行します。 |
| `docker compose build` | `docker-compose.yml` に定義されたサービスのイメージをビルドします。 |
| `docker compose pull` | `docker-compose.yml` に定義されたサービスのイメージをプル（ダウンロード）します。 |
| `docker compose push` | `docker-compose.yml` に定義されたサービスのイメージをプッシュ（アップロード）します。 |
| `docker compose create`| コンテナを作成しますが、起動はしません。 |
| `docker compose rm` | 停止しているコンテナを削除します。 |

---

### **クリーンアップ**

| コマンド | 説明 |
| :--- | :--- |
| `docker compose down --volumes` | コンテナとネットワークに加え、データボリュームも削除します。 |
| `docker compose down --rmi all` | コンテナ、ネットワーク、ボリューム、**全てのイメージ**を削除します。 |

---

## コマンドの例と詳細解説

### **例：`docker compose exec -it dev bash`**

このコマンドは、「**現在実行中の `dev` という名前のサービス（コンテナ）の中で、対話的なターミナル（bash）を起動する**」という意味です。コンテナの中に入って、ファイルを確認したり、プログラムを手動で実行したりするデバッグ作業で頻繁に使用されます。

#### **コマンドの分解**

このコマンドは `docker compose exec [options] [service] [command]` の構文に当てはまります。

| 部分 | 該当 | 意味 |
| :--- | :--- | :--- |
| `docker compose` | - | Docker Compose V2に対する命令です。 |
| `exec` | **(サブコマンド)** | 実行中のコンテナ内で、追加のコマンドを実行します。 |
| `-it` | **[options]** | **`-i` (対話的)** と **`-t` (仮想ターミナル割当)** の2つのオプションを組み合わせたものです。これにより、キーボード入力を受け付け、ターミナルでの表示が整うため、通常のターミナルと同じ感覚で操作できます。 |
| `dev` | **[service]** | コマンドを実行する対象のサービス名です。`docker-compose.yml`ファイル内で定義されています。 |
| `bash` | **[command]** | `dev`コンテナ内で実行する具体的なコマンドです。`bash`シェルを起動し、コンテナ内部を操作できるようにします。 |