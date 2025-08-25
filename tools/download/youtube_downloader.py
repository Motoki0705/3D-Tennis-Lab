#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Robust YouTube downloader with auto format selection (yt-dlp).
- Prefers MP4 when available; otherwise falls back gracefully.
- Merges bestvideo+bestaudio via ffmpeg when needed.
- Prints available formats on failure for quick debugging.

Usage (PowerShell):
  # 例: 通常ダウンロード（最適自動選択）
  python tools/download/youtube_downloader.py "https://www.youtube.com/watch?v=xp2mYmNl-lg"

  # 保存先や拡張子希望を指定
  python tools/download/youtube_downloader.py "https://www.youtube.com/watch?v=xp2mYmNl-lg" ^
      --out "data/raw/videos" --prefer-ext mp4
"""

from __future__ import annotations
import pathlib
import re
from typing import Dict, Any, Optional, Tuple, List

from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError


def _pick_progressive(formats: List[Dict[str, Any]], prefer_ext: str) -> Optional[Dict[str, Any]]:
    """音声込み（progressive）の単一トラックを優先拡張子で選ぶ。"""
    progressive = [
        f for f in formats if f.get("acodec") not in (None, "none") and f.get("vcodec") not in (None, "none")
    ]
    # まず prefer_ext のもの
    cand = [f for f in progressive if (f.get("ext") == prefer_ext)]
    if cand:
        # 解像度・ビットレートが高い順
        cand.sort(key=lambda f: (f.get("height") or 0, f.get("tbr") or 0), reverse=True)
        return cand[0]
    if progressive:
        progressive.sort(key=lambda f: (f.get("height") or 0, f.get("tbr") or 0), reverse=True)
        return progressive[0]
    return None


def _pick_best_av_pair(
    formats: List[Dict[str, Any]], prefer_ext: str
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """映像（video only）と音声（audio only）の最適ペアを選ぶ。prefer_ext を映像拡張子に優先。"""
    videos = [f for f in formats if f.get("vcodec") not in (None, "none") and f.get("acodec") in (None, "none")]
    audios = [f for f in formats if f.get("acodec") not in (None, "none") and f.get("vcodec") in (None, "none")]

    # prefer_ext の映像があれば優先
    vids_pref = [v for v in videos if v.get("ext") == prefer_ext]
    vids = vids_pref if vids_pref else videos
    if vids:
        vids.sort(key=lambda f: (f.get("height") or 0, f.get("tbr") or 0), reverse=True)
    if audios:
        audios.sort(key=lambda f: (f.get("abr") or 0, f.get("tbr") or 0), reverse=True)

    best_v = vids[0] if vids else None
    best_a = audios[0] if audios else None
    return best_v, best_a


def _build_format_string(info: Dict[str, Any], prefer_ext: str) -> str:
    """
    1080p を優先的に狙い、なければ下位解像度にフォールバック。
    """
    formats = info.get("formats") or []

    # まず 1080p 以下の bestvideo+bestaudio を狙う
    has_1080 = any(f.get("height") == 1080 for f in formats if f.get("vcodec") not in (None, "none"))
    if has_1080:
        return "bestvideo[height<=1080]+bestaudio/best[height<=1080]"

    # 1080p が無ければ元のロジックにフォールバック
    return "bv*+ba/b"


def _extract_timestamp_from_url(url: str) -> Optional[int]:
    """
    URL の &t=315s・?t=315s・?t=315 のような開始秒を整数にして返す（存在すれば）。
    ※ 注意: 取得しても、そのままではダウンロード区間には適用しません（後述）。
    """
    m = re.search(r"[?&]t=(\d+)(s)?", url)
    if m:
        return int(m.group(1))
    return None


def download_video(
    url: str,
    output_dir: str = "data/raw/videos",
    prefer_ext: str = "mp4",
    filename_template: str = "%(title)s-%(id)s.%(ext)s",
    cookies: Optional[str] = None,
) -> None:
    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # まず情報だけ取得（download=False）
    probe_opts: Dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "ignoreerrors": False,
    }
    if cookies:
        probe_opts["cookiefile"] = cookies

    with YoutubeDL(probe_opts) as ydl:
        info = ydl.extract_info(url, download=False)

    # 最適フォーマットを構築
    fmt = _build_format_string(info, prefer_ext)

    # 実ダウンロード設定
    ydl_opts: Dict[str, Any] = {
        "outtmpl": str(out / filename_template),
        "format": fmt,  # 自動構築した format 式
        "merge_output_format": prefer_ext,  # 結合/リマックスの優先拡張子
        "noprogress": False,
        "ignoreerrors": False,
        "retries": 5,
        "fragment_retries": 5,
        "postprocessors": [
            # 必要に応じて mp4/mkv 等へリマックス（re-mux）
            {"key": "FFmpegVideoRemuxer", "preferedformat": prefer_ext},
        ],
    }
    if cookies:
        ydl_opts["cookiefile"] = cookies

    # HLS/DASH 等の断片でも粘る
    ydl_opts["concurrent_fragment_downloads"] = 4

    try:
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except DownloadError as e:
        # 失敗時はフォーマット一覧を表示して支援
        print("\n[Error] ダウンロードに失敗しました。利用可能フォーマット候補を表示します:\n")
        _print_format_table(info)
        print("\nヒント:")
        print("  - 上の表の format_id を指定して再実行すると解決することがあります。")
        print("  - 例) yt-dlp -f 137+140 <URL>   # 137=mp4 video, 140=m4a audio")
        print("  - 地域/年齢制限がある場合は cookies.txt を指定してください（--cookies PATH）。")
        raise e


def _print_format_table(info: Dict[str, Any]) -> None:
    """取得済み info から簡易フォーマット表を出力。"""
    formats = info.get("formats") or []
    # ヘッダ
    print(f"{'id':<8} {'ext':<5} {'vcodec':<12} {'acodec':<12} {'res':<9} {'tbr(Mbps)':<10} {'note'}")
    for f in formats:
        fid = str(f.get("format_id", ""))
        ext = str(f.get("ext", ""))
        vcd = str(f.get("vcodec", ""))
        acd = str(f.get("acodec", ""))
        res = f"{f.get('width', '')}x{f.get('height', '')}" if f.get("height") else ""
        tbr = f.get("tbr")
        tbr_mb = f"{(tbr / 1000):.2f}" if isinstance(tbr, (int, float)) else ""
        note = f.get("format_note", "") or ""
        print(f"{fid:<8} {ext:<5} {vcd:<12} {acd:<12} {res:<9} {tbr_mb:<10} {note}")


def main():
    url = "https://www.youtube.com/watch?v=F2uJDskg8os"
    output_dir = "data/raw/videos"
    prefer_ext = "mp4"
    cookies = None  # "path/to/cookies.txt"  # 地域/年齢制限対策用

    # URL中の t= 開始秒は自動ダウンロードには効かない旨を案内（必要ならセクションDLを使う）
    tsec = _extract_timestamp_from_url(url)
    if tsec:
        print(f"[Info] URL に開始秒 t={tsec}s が含まれています。部分ダウンロードは自動では行いません。")
        print("       区間だけ欲しい場合は --download-sections（CLI）等をご検討ください。")

    download_video(
        url=url,
        output_dir=output_dir,
        prefer_ext=prefer_ext,
        cookies=cookies,
    )


if __name__ == "__main__":
    main()
