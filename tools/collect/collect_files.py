#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
collect_files.py — 再帰的にファイルを収集し、1つの出力フォルダへ集約するユーティリティ

機能概要
- 複数のソースディレクトリを再帰探索し、ファイルを出力ディレクトリへ「集める」
- フィルタ: 拡張子、fnmatch パターン、除外パターン、サイズ範囲、最大深さ、隠しファイル除外
- 集約方法: フラット(既定) or 元のフォルダ構造を維持 (--preserve-structure)
- 転送モード: copy(既定)/move/symlink/hardlink
- 重複時の挙動 (--dedup): rename(既定)/skip/overwrite/hash
  - hash: 内容ハッシュで重複排除。名称衝突はハッシュ接尾辞で回避
- 乾式実行 (--dry-run)
- マニフェストCSV出力 (--manifest), ログ出力 (--log)
- 並列実行 (--workers)

使用例
  # 1) 画像を *.png と *.jpg のみ収集し、既存名と衝突時は自動リネーム
  python collect_files.py --src data/inputA --src data/inputB --dst outputs/all_assets \
      --ext .png --ext .jpg --dedup rename

  # 2) 研究ログをフラットに集め、内容重複はハッシュでスキップ、CSVマニフェストを保存
  python collect_files.py --src logs --dst outputs/log_dump --pattern "*/*.log" \
      --dedup hash --manifest outputs/manifest.csv

  # 3) 元のディレクトリ階層を維持してコピー
  python collect_files.py --src corpora --dst outputs/corpora_copy --preserve-structure

  # 4) 動画のみを移動 (破壊的)。最大深さ3、除外パターン指定、上書き許可
  python collect_files.py --src videos --dst outputs/vids --ext .mp4 --ext .mov \
      --max-depth 3 --exclude "*temp*" --mode move --dedup overwrite

注意
- デフォルトは安全な copy + rename。
- Windows で symlink は権限が必要です。失敗時はエラーを記録します。
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import hashlib
import logging
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional, Tuple


# -------------------------------
# 型定義
# -------------------------------


@dataclass(frozen=True)
class FileEntry:
    src: Path
    root: Path  # 対応するソースルート
    rel: Path  # root からの相対パス
    size: int
    mtime: float  # unix timestamp


@dataclass
class PlanItem:
    src: Path
    dst: Path
    op: str  # 'copy' | 'move' | 'symlink' | 'hardlink'
    size: int
    mtime: float
    sha256: Optional[str] = None
    skipped_reason: Optional[str] = None


# -------------------------------
# ユーティリティ
# -------------------------------


def is_hidden(path: Path) -> bool:
    name = path.name
    if name.startswith("."):
        return True
    # Windows: ドットで判断しにくい場合もあるが、標準ライブラリのみでの判定は難しい
    return False


def within(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except Exception:
        return False


def sha256_of(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def ensure_parent(dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)


def with_suffix_number(dst: Path, n: int) -> Path:
    # foo/bar/name.ext -> foo/bar/name_001.ext
    stem = dst.stem
    suffix = dst.suffix
    return dst.with_name(f"{stem}_{n:03d}{suffix}")


def with_suffix_hash(dst: Path, digest8: str) -> Path:
    stem, suffix = dst.stem, dst.suffix
    return dst.with_name(f"{stem}_{digest8}{suffix}")


# -------------------------------
# 収集ロジック
# -------------------------------


def iter_files(root: Path, *, max_depth: Optional[int]) -> Iterator[Path]:
    if max_depth is None:
        yield from root.rglob("*")
        return

    # 深さ制限版の walk
    base_parts = len(root.resolve().parts)
    for dirpath, dirnames, filenames in os.walk(root):
        cur_parts = len(Path(dirpath).resolve().parts)
        if max_depth is not None and (cur_parts - base_parts) >= max_depth:
            # これ以上潜らない
            dirnames[:] = []
        for fn in filenames:
            yield Path(dirpath) / fn


def collect_candidates(
    src_roots: Iterable[Path],
    dst_root: Path,
    *,
    include_exts: set[str],
    include_patterns: list[str],
    exclude_patterns: list[str],
    include_hidden: bool,
    max_depth: Optional[int],
    min_size: Optional[int],
    max_size: Optional[int],
) -> list[FileEntry]:
    entries: list[FileEntry] = []
    dst_root_resolved = dst_root.resolve()

    for root in src_roots:
        root = root.resolve()
        if not root.exists():
            logging.warning("ソースが存在しません: %s", root)
            continue
        for p in iter_files(root, max_depth=max_depth):
            if not p.is_file():
                continue

            # 出力ディレクトリ配下は探索しない（自己ループ回避）
            try:
                if within(p, dst_root_resolved):
                    continue
            except Exception:
                pass

            # 隠しファイル
            if not include_hidden and is_hidden(p):
                continue

            # 拡張子フィルタ
            if include_exts:
                if p.suffix.lower() not in include_exts:
                    continue

            # include / exclude パターン
            sp = str(p.as_posix())
            if include_patterns:
                if not any(fnmatch.fnmatch(sp, pat) for pat in include_patterns):
                    continue

            if exclude_patterns:
                if any(fnmatch.fnmatch(sp, pat) for pat in exclude_patterns):
                    continue

            st = p.stat()
            size = st.st_size
            if min_size is not None and size < min_size:
                continue
            if max_size is not None and size > max_size:
                continue

            rel = p.relative_to(root)
            entries.append(FileEntry(src=p, root=root, rel=rel, size=size, mtime=st.st_mtime))

    return entries


def plan_transfers(
    entries: list[FileEntry],
    dst_root: Path,
    *,
    mode: str,  # copy/move/symlink/hardlink
    dedup: str,  # rename/skip/overwrite/hash
    preserve_structure: bool,
    dry_run: bool,
) -> list[PlanItem]:
    dst_root = dst_root.resolve()
    plans: list[PlanItem] = []

    # ハッシュ方式の重複排除は、既出の内容を覚えておく
    seen_hashes: dict[str, Path] = {}  # sha256 -> dst path

    for e in entries:
        dst = dst_root / (e.rel if preserve_structure else e.src.name)

        # 既存衝突処理
        if dst.exists():
            if dedup == "skip":
                plans.append(PlanItem(e.src, dst, mode, e.size, e.mtime, skipped_reason="exists_skip"))
                continue
            elif dedup == "overwrite":
                plans.append(PlanItem(e.src, dst, mode, e.size, e.mtime))
            elif dedup == "rename":
                n = 1
                alt = dst
                while alt.exists():
                    alt = with_suffix_number(dst, n)
                    n += 1
                dst = alt
                plans.append(PlanItem(e.src, dst, mode, e.size, e.mtime))
            elif dedup == "hash":
                # 内容を見て同一ならスキップ、異なるならハッシュ接尾辞で保存
                try:
                    h = sha256_of(e.src)
                except Exception as ex:
                    plans.append(PlanItem(e.src, dst, mode, e.size, e.mtime, skipped_reason=f"hash_error:{ex}"))
                    continue

                if h in seen_hashes:
                    plans.append(
                        PlanItem(e.src, seen_hashes[h], mode, e.size, e.mtime, sha256=h, skipped_reason="dup_by_hash")
                    )
                    continue

                # 既存同名の内容が異なる場合は接尾辞付け
                digest8 = h[:8]
                alt = with_suffix_hash(dst, digest8)
                n = 1
                base = alt
                while alt.exists():
                    alt = with_suffix_number(base, n)
                    n += 1
                dst = alt
                seen_hashes[h] = dst
                plans.append(PlanItem(e.src, dst, mode, e.size, e.mtime, sha256=h))
            else:
                raise ValueError(f"Unknown dedup mode: {dedup}")
        else:
            if dedup == "hash":
                try:
                    h = sha256_of(e.src)
                except Exception as ex:
                    plans.append(PlanItem(e.src, dst, mode, e.size, e.mtime, skipped_reason=f"hash_error:{ex}"))
                    continue
                if h in seen_hashes:
                    plans.append(
                        PlanItem(e.src, seen_hashes[h], mode, e.size, e.mtime, sha256=h, skipped_reason="dup_by_hash")
                    )
                    continue
                seen_hashes[h] = dst
                plans.append(PlanItem(e.src, dst, mode, e.size, e.mtime, sha256=h))
            else:
                plans.append(PlanItem(e.src, dst, mode, e.size, e.mtime))

    return plans


def execute_plan_item(item: PlanItem, dry_run: bool) -> Tuple[PlanItem, Optional[str]]:
    if item.skipped_reason:
        return item, item.skipped_reason

    try:
        if dry_run:
            return item, None

        ensure_parent(item.dst)

        if item.op == "copy":
            shutil.copy2(item.src, item.dst)
        elif item.op == "move":
            shutil.move(str(item.src), str(item.dst))
        elif item.op == "symlink":
            # 相対シンボリックリンク
            rel = os.path.relpath(item.src, start=item.dst.parent)
            os.symlink(rel, item.dst)
        elif item.op == "hardlink":
            try:
                os.link(item.src, item.dst)
            except OSError:
                # 異なるファイルシステム等で失敗したら copy にフォールバック
                shutil.copy2(item.src, item.dst)
        else:
            return item, f"unknown_op:{item.op}"

        return item, None
    except Exception as ex:
        return item, f"error:{ex}"


def write_manifest(path: Path, plans: list[PlanItem]) -> None:
    ensure_parent(path)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["src", "dst", "op", "size", "mtime", "sha256", "skipped_reason"])
        for p in plans:
            w.writerow([str(p.src), str(p.dst), p.op, p.size, f"{p.mtime:.3f}", p.sha256 or "", p.skipped_reason or ""])


# -------------------------------
# メイン
# -------------------------------


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="再帰的にファイルを収集し、出力フォルダに集約します。")
    p.add_argument("--src", action="append", required=True, help="ソースディレクトリ (複数指定可)")
    p.add_argument("--dst", required=True, help="出力ディレクトリ")
    p.add_argument(
        "--mode", choices=["copy", "move", "symlink", "hardlink"], default="copy", help="転送モード (既定: copy)"
    )
    p.add_argument(
        "--dedup", choices=["rename", "skip", "overwrite", "hash"], default="rename", help="重複時の挙動 (既定: rename)"
    )
    p.add_argument("--ext", action="append", default=[], help="含める拡張子 (例: .png) 複数指定可 (小文字判定)")
    p.add_argument("--pattern", action="append", default=[], help="fnmatch パターン (path基準, 例: */*.log)")
    p.add_argument("--exclude", action="append", default=[], help="除外パターン (fnmatch)")
    p.add_argument("--include-hidden", action="store_true", help="隠しファイルを含める")
    p.add_argument("--max-depth", type=int, default=None, help="最大探索深さ (root 直下=0)")
    p.add_argument("--min-size", type=int, default=None, help="最小サイズ (bytes)")
    p.add_argument("--max-size", type=int, default=None, help="最大サイズ (bytes)")
    p.add_argument("--preserve-structure", action="store_true", help="元のディレクトリ構造を維持して出力")
    p.add_argument("--manifest", type=str, default=None, help="マニフェスト CSV 出力先パス")
    p.add_argument("--workers", type=int, default=os.cpu_count() or 4, help="並列ワーカー数")
    p.add_argument("--dry-run", action="store_true", help="コピー/移動を行わず計画のみ実行")
    p.add_argument("--log", type=str, default=None, help="ログ出力パス (INFO)")
    return p.parse_args(argv)


def setup_logging(log_path: Optional[str]) -> None:
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers,
    )


def main(argv: Optional[list[str]] = None) -> int:
    ns = parse_args(argv)
    setup_logging(ns.log)

    src_roots = [Path(s) for s in ns.src]
    dst_root = Path(ns.dst)

    include_exts = set(e.lower() for e in ns.ext)
    include_patterns = list(ns.pattern)
    exclude_patterns = list(ns.exclude)

    logging.info("収集開始: src=%s -> dst=%s", [str(p) for p in src_roots], str(dst_root))
    logging.info("mode=%s dedup=%s preserve=%s dry_run=%s", ns.mode, ns.dedup, ns.preserve_structure, ns.dry_run)

    entries = collect_candidates(
        src_roots,
        dst_root,
        include_exts=include_exts,
        include_patterns=include_patterns,
        exclude_patterns=exclude_patterns,
        include_hidden=ns.include_hidden,
        max_depth=ns.max_depth,
        min_size=ns.min_size,
        max_size=ns.max_size,
    )

    if not entries:
        logging.info("対象ファイルが見つかりませんでした。条件を見直してください。")
        return 0

    logging.info("候補ファイル: %d 件", len(entries))

    plans = plan_transfers(
        entries,
        dst_root,
        mode=ns.mode,
        dedup=ns.dedup,
        preserve_structure=ns.preserve_structure,
        dry_run=ns.dry_run,
    )

    # 並列実行
    ok = 0
    skipped = 0
    failed = 0
    futures = []
    with ThreadPoolExecutor(max_workers=ns.workers) as ex:
        for item in plans:
            futures.append(ex.submit(execute_plan_item, item, ns.dry_run))

        for fut in as_completed(futures):
            item, err = fut.result()
            if err is None:
                if item.skipped_reason:
                    skipped += 1
                    logging.info("SKIP: %s -> %s (%s)", item.src, item.dst, item.skipped_reason)
                else:
                    ok += 1
                    logging.info("OK  : %s -> %s", item.src, item.dst)
            else:
                failed += 1
                logging.error("FAIL: %s -> %s (%s)", item.src, item.dst, err)

    if ns.manifest:
        try:
            write_manifest(Path(ns.manifest), plans)
            logging.info("マニフェストを書き出しました: %s", ns.manifest)
        except Exception as ex:
            logging.error("マニフェスト書き出しに失敗: %s", ex)

    logging.info("完了: 成功=%d, スキップ=%d, 失敗=%d", ok, skipped, failed)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
