#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
YouTube batch downloader (yt-dlp) with duplication guard and YAML config.

Features
- YAMLでURLリストを管理し順次ダウンロード
- yt-dlpのdownload_archiveで動画IDベースの重複取得を自動スキップ
- 追加でURLアーカイブも記録（監査・簡易再確認用）
- 1080pなど解像度上限(max_height)でベスト選択＋フォールバック
- 失敗時にフォーマット一覧を表示してデバッグ支援（任意）

Usage:
  python tools/download/youtube_downloader.py --config configs/youtube_download.yaml
  # 単発追加（設定のurlsに無くてもOK）
  python tools/download/youtube_downloader.py --config configs/youtube_download.yaml \
      --url "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --max-height 720
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    import yaml  # PyYAML
except Exception:  # pragma: no cover
    print("PyYAML が見つかりません。`pip install pyyaml` を実行してください。", file=sys.stderr)
    raise

try:
    from yt_dlp import YoutubeDL
    from yt_dlp.utils import DownloadError
except Exception:  # pragma: no cover
    print("yt-dlp が見つかりません。`pip install yt-dlp` を実行してください。", file=sys.stderr)
    raise


# ------------------------------
# Config load / merge
# ------------------------------

DEFAULTS: Dict[str, Any] = {
    "out_dir": "data/raw/videos",
    "prefer_ext": "mp4",
    "max_height": 1080,  # <= None で上限なし
    "concurrent_fragment_downloads": 4,
    "retries": 5,
    "fragment_retries": 5,
    "cookies": None,
    "archive_file": "data/meta/yt_archive.txt",  # yt-dlp用のIDアーカイブ
    "url_archive": "data/meta/url_archive.txt",  # 追加のURLアーカイブ（任意）
    "filename_template": "%(title)s-%(id)s.%(ext)s",
    "print_formats_on_error": True,
    "quiet": False,
    "no_warnings": True,
    "urls": [],  # YAML側で列挙
}


def load_yaml(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config YAML が見つかりません: {p}")
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError("Config YAML のトップレベルは辞書である必要があります。")
        return data


def merge_config(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if v is None:
            continue
        # リストのときは結合。他は上書き。
        if k == "urls":
            base_list = out.get("urls") or []
            add_list = v if isinstance(v, list) else [v]
            out["urls"] = [*base_list, *add_list]
        else:
            out[k] = v
    return out


# ------------------------------
# Helpers
# ------------------------------


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def now_iso() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def normalize_urls(urls: Iterable[str]) -> List[str]:
    # 簡易正規化（前後空白除去）。本格的な正規化はID抽出に任せる（yt-dlp側が堅牢）
    out = []
    for u in urls:
        if not u:
            continue
        s = str(u).strip()
        if s and not s.startswith("#"):
            out.append(s)
    return out


def build_format_string(max_height: Optional[int]) -> str:
    """max_heightがあればその上限でbestを選択。なければ包括的フォールバック。"""
    if isinstance(max_height, int) and max_height > 0:
        return f"bestvideo[height<={max_height}]+bestaudio/best[height<={max_height}]"
    return "bv*+ba/b"  # 包括フォールバック


def print_format_table(info: Dict[str, Any]) -> None:
    formats = info.get("formats") or []
    print(f"{'id':<8} {'ext':<5} {'vcodec':<16} {'acodec':<12} {'res':<10} {'tbr(Mbps)':<10} {'note'}")
    for f in formats:
        fid = str(f.get("format_id", ""))
        ext = str(f.get("ext", ""))
        vcd = str(f.get("vcodec", ""))
        acd = str(f.get("acodec", ""))
        res = f"{f.get('width', '')}x{f.get('height', '')}" if f.get("height") else ""
        tbr = f.get("tbr")
        tbr_mb = f"{(tbr / 1000):.2f}" if isinstance(tbr, (int, float)) else ""
        note = f.get("format_note", "") or ""
        print(f"{fid:<8} {ext:<5} {vcd:<16} {acd:<12} {res:<10} {tbr_mb:<10} {note}")


def append_url_archive(url_archive: Path, url: str, status: str) -> None:
    ensure_parent(url_archive)
    with url_archive.open("a", encoding="utf-8") as f:
        f.write(f"{now_iso()}\t{status}\t{url}\n")


# ------------------------------
# Core download routine
# ------------------------------


def download_one(url: str, cfg: Dict[str, Any]) -> tuple[bool, str]:
    """
    Returns: (ok, status)
      ok: True=成功 or スキップ、False=失敗
      status: 'downloaded' | 'skipped' | 'failed'
    """
    out_dir = Path(cfg["out_dir"])
    archive_file = Path(cfg["archive_file"])
    url_archive = Path(cfg["url_archive"])
    ensure_parent(out_dir / "dummy")
    ensure_parent(archive_file)
    ensure_parent(url_archive)

    fmt = build_format_string(cfg.get("max_height"))
    ydl_opts: Dict[str, Any] = {
        "outtmpl": str(out_dir / cfg["filename_template"]),
        "format": fmt,
        "merge_output_format": cfg["prefer_ext"],
        "postprocessors": [{"key": "FFmpegVideoRemuxer", "preferedformat": cfg["prefer_ext"]}],
        "download_archive": str(archive_file),  # ★ これが重複防止の中核
        "concurrent_fragment_downloads": int(cfg["concurrent_fragment_downloads"]),
        "retries": int(cfg["retries"]),
        "fragment_retries": int(cfg["fragment_retries"]),
        "noprogress": False,
        "quiet": bool(cfg.get("quiet", False)),
        "no_warnings": bool(cfg.get("no_warnings", True)),
    }
    cookies = cfg.get("cookies")
    if cookies:
        ydl_opts["cookiefile"] = cookies

    # まず情報抽出（失敗時のフォーマット表示・ログに使う）
    info: Optional[Dict[str, Any]] = None
    try:
        with YoutubeDL({"quiet": True, "no_warnings": True, **({"cookiefile": cookies} if cookies else {})}) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception:
        # 情報抽出に失敗しても本ダウンロードで再挑戦する
        info = None

    # 実ダウンロード
    try:
        with YoutubeDL(ydl_opts) as ydl:
            # yt-dlpはdownload_archiveにヒットしたものをスキップ扱いにする
            _ret = ydl.download([url])
            # ret==0 は成功（少なくともエラーはない）。アーカイブヒットの厳密判定は難しいため、
            # ここでは成功=downloaded/もしくはskippedの可能性とし、URLアーカイブには "ok" として記録。
            append_url_archive(url_archive, url, "ok")  # 監査用
            return True, "downloaded_or_skipped"  # 実務上はこれで十分
    except DownloadError as e:
        print(f"\n[Error] ダウンロード失敗: {url}\n{e}", file=sys.stderr)
        if cfg.get("print_formats_on_error", True) and info:
            print("\n[利用可能フォーマット候補]\n")
            print_format_table(info)
            print("\nヒント:")
            print("  - 上の表の format_id を指定して再実行すると解決することがあります。")
            print("  - 例) yt-dlp -f 137+140 <URL>   # 137=mp4 video, 140=m4a audio")
            print("  - 地域/年齢制限がある場合は cookies.txt を指定してください（--cookies PATH）。")
        append_url_archive(url_archive, url, "failed")
        return False, "failed"


# ------------------------------
# CLI
# ------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="YouTube downloader with YAML config & duplication guard (yt-dlp)")
    p.add_argument("--config", type=str, default="configs/youtube_download.yaml", help="YAML設定ファイルのパス")
    p.add_argument("--url", action="append", help="単発URL。複数指定可（設定のurlsに追加される）")
    p.add_argument("--out", dest="out_dir", type=str, help="出力先ディレクトリ")
    p.add_argument("--prefer-ext", type=str, choices=["mp4", "mkv", "webm", "mov"], help="最終拡張子(リマックス先)")
    p.add_argument("--max-height", type=int, help="解像度上限（例: 1080）")
    p.add_argument("--cookies", type=str, help="cookies.txt のパス（地域/年齢制限対策）")
    p.add_argument("--archive-file", type=str, help="yt-dlpのdownload_archiveファイルパス")
    p.add_argument("--url-archive", type=str, help="取得URLの監査用テキスト")
    p.add_argument(
        "--show-formats", action="store_true", help="URLの利用可能フォーマットを表示して終了（ダウンロードなし）"
    )
    p.add_argument("--dry-run", action="store_true", help="情報抽出のみでダウンロードしない")
    p.add_argument("--quiet", action="store_true", help="yt-dlpの進捗出力を抑制")
    p.add_argument("--no-warnings", action="store_true", help="警告を抑制")
    p.add_argument("--no-print-formats-on-error", action="store_true", help="失敗時のフォーマット一覧表示を抑止")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    # 1) 設定ロード
    try:
        file_cfg = load_yaml(args.config)
    except Exception as e:
        print(f"[Config Error] {e}", file=sys.stderr)
        return 2

    # 2) デフォルト → YAML → CLI の順にマージ
    cli_cfg: Dict[str, Any] = {
        "out_dir": args.out_dir,
        "prefer_ext": args.prefer_ext,
        "max_height": args.max_height,
        "cookies": args.cookies,
        "archive_file": args.archive_file,
        "url_archive": args.url_archive,
        "quiet": args.quiet if args.quiet else None,  # Noneなら既定維持
        "no_warnings": args.no_warnings if args.no_warnings else None,
        "print_formats_on_error": False if args.no_print_formats_on_error else None,
        "urls": args.url or [],
    }
    cfg = merge_config(DEFAULTS, file_cfg)
    cfg = merge_config(cfg, cli_cfg)

    # 3) URL集合を決定
    urls = normalize_urls(cfg.get("urls", []))
    if not urls:
        print("処理対象URLがありません。--url または YAMLの urls を指定してください。", file=sys.stderr)
        return 1

    # 4) --show-formats / --dry-run を処理
    if args.show_formats or args.dry_run:
        cookies = cfg.get("cookies")
        with YoutubeDL({"quiet": True, "no_warnings": True, **({"cookiefile": cookies} if cookies else {})}) as ydl:
            for u in urls:
                print(f"\n=== {u} ===")
                try:
                    info = ydl.extract_info(u, download=False)
                    if args.show_formats:
                        print_format_table(info)
                    else:
                        # 概要だけ
                        title = info.get("title")
                        vid = info.get("id")
                        print(f"title: {title} | id: {vid}")
                except Exception as e:
                    print(f"[Error] extract_info 失敗: {u}\n{e}", file=sys.stderr)
        return 0

    # 5) 本ダウンロード
    any_fail = False
    for u in urls:
        ok, status = download_one(u, cfg)
        if ok:
            print(f"[OK] {status}: {u}")
        else:
            any_fail = True
            print(f"[NG] {status}: {u}", file=sys.stderr)

    return 1 if any_fail else 0


if __name__ == "__main__":
    sys.exit(main())
