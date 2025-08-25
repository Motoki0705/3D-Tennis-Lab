import argparse
import filecmp
import os
import shutil


def sync_directories(source_dir, dest_dir):
    """
    2つのディレクトリを同期します。source_dirを正として、dest_dirを更新します。
    この関数は再帰的に呼び出されます。

    :param source_dir: 同期元のディレクトリパス
    :param dest_dir: 同期先のディレクトリパス
    """

    # ディレクトリ比較オブジェクトを作成
    # filecmp.dircmp は2つのディレクトリの差分を効率的に検出します
    comparison = filecmp.dircmp(source_dir, dest_dir)

    # 1. ソースにのみ存在するファイル/ディレクトリ -> コピー
    for name in comparison.left_only:
        source_path = os.path.join(source_dir, name)
        dest_path = os.path.join(dest_dir, name)

        if os.path.isdir(source_path):
            print(f"🔼 新規作成 (フォルダ): {dest_path}")
            shutil.copytree(source_path, dest_path)
        else:
            print(f"🔼 新規作成 (ファイル): {dest_path}")
            shutil.copy2(source_path, dest_path)  # copy2はメタデータもコピー

    # 2. 両方に存在するが内容が異なるファイル -> 上書きコピー
    for name in comparison.diff_files:
        source_path = os.path.join(source_dir, name)
        dest_path = os.path.join(dest_dir, name)
        print(f"🔄 更新: {dest_path}")
        shutil.copy2(source_path, dest_path)

    # 3. 同期先にのみ存在するファイル/ディレクトリ -> 削除
    for name in comparison.right_only:
        dest_path = os.path.join(dest_dir, name)

        if os.path.isdir(dest_path):
            print(f"🗑️ 削除 (フォルダ): {dest_path}")
            shutil.rmtree(dest_path)
        else:
            print(f"🗑️ 削除 (ファイル): {dest_path}")
            os.remove(dest_path)

    # 4. 両方に存在するサブディレクトリ -> 再帰的に同期処理を呼び出す
    for name in comparison.common_dirs:
        # 次の階層のパスを指定して再帰呼び出し
        sync_directories(os.path.join(source_dir, name), os.path.join(dest_dir, name))


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(
        description="ローカルディレクトリを、ファイルシステム上のGoogleドライブパスに差分同期します。",
        formatter_class=argparse.RawTextHelpFormatter,  # ヘルプの改行を維持
    )
    parser.add_argument("source_directory", help="同期元のローカルディレクトリのパス")
    parser.add_argument("drive_directory", help="同期先のGoogleドライブのパス\n" '例: "G:\\マイドライブ\\同期フォルダ"')

    args = parser.parse_args()

    source_path = args.source_directory
    dest_path = args.drive_directory

    # --- パスの存在チェック ---
    if not os.path.isdir(source_path):
        print(f"❌ エラー: 同期元ディレクトリが見つかりません: {source_path}")
        return

    if not os.path.isdir(dest_path):
        try:
            # 同期先が存在しない場合は作成する
            print(f"[INFO] 同期先ディレクトリが存在しないため、作成します: {dest_path}")
            os.makedirs(dest_path)
        except OSError as e:
            print(f"❌ エラー: 同期先ディレクトリの作成に失敗しました: {e}")
            return

    print("\n📁 同期を開始します")
    print(f"  - 同期元: {source_path}")
    print(f"  - 同期先: {dest_path}\n")

    # 同期処理の実行
    sync_directories(source_path, dest_path)

    print("\n🎉 同期処理が完了しました。")


if __name__ == "__main__":
    main()
