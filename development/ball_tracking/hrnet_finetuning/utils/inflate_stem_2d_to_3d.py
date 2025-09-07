import argparse
import os
from collections import OrderedDict
from typing import Any, Dict, Tuple

try:
    import torch
except ModuleNotFoundError:
    raise SystemExit("PyTorch (torch) が見つかりません。仮想環境を有効化するか、pip/poetryでインストールしてください。")


def _load_ckpt(path: str) -> Any:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location="cpu")


def _extract_state_dict(ckpt_obj: Any) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    meta: Dict[str, Any] = {"root_type": type(ckpt_obj).__name__, "root_keys": [], "selected_key": None}

    if isinstance(ckpt_obj, (dict, OrderedDict)):
        meta["root_keys"] = list(ckpt_obj.keys())
        for key in ["state_dict", "model_state_dict", "model_state", "model", "net", "weights"]:
            if key in ckpt_obj and isinstance(ckpt_obj[key], (dict, OrderedDict)):
                inner = ckpt_obj[key]
                if any(isinstance(v, torch.Tensor) for v in inner.values()):
                    meta["selected_key"] = key
                    return dict(inner), meta
        if any(isinstance(v, torch.Tensor) for v in ckpt_obj.values()):
            return dict(ckpt_obj), meta

    raise ValueError("Unsupported checkpoint format: could not locate a state_dict-like mapping")


def _inject_state_dict(ckpt_obj: Any, state_dict: Dict[str, torch.Tensor], meta: Dict[str, Any]) -> Any:
    if isinstance(ckpt_obj, (dict, OrderedDict)) and meta.get("selected_key"):
        key = meta["selected_key"]
        ckpt_obj[key] = state_dict
        return ckpt_obj
    else:
        # Save as plain state_dict
        return state_dict


def _channels_to_depth_conv1(w2d: torch.Tensor) -> torch.Tensor:
    # w2d: (out_c, in_c=3*F, kH, kW) -> w3d: (out_c, 3, F, kH, kW)
    if w2d.ndim != 4:
        raise ValueError(f"conv1 weight must be 4D (Conv2d), got shape {tuple(w2d.shape)}")
    out_c, in_c, kH, kW = w2d.shape
    if in_c % 3 != 0:
        raise ValueError(f"conv1 in_channels={in_c} が 3 の倍数ではありません。frames_in を推定できません。")
    F = in_c // 3
    return w2d.view(out_c, 3, F, kH, kW).contiguous()


def _inflate_repeat(w2d: torch.Tensor, kD: int, scale: bool) -> torch.Tensor:
    # w2d: (out_c, in_c, kH, kW) -> (out_c, in_c, kD, kH, kW)
    if w2d.ndim != 4:
        raise ValueError(f"weight must be 4D (Conv2d), got shape {tuple(w2d.shape)}")
    w3d = w2d.unsqueeze(2).repeat(1, 1, kD, 1, 1)
    if scale and kD > 0:
        w3d = w3d / float(kD)
    return w3d.contiguous()


def convert_stem_2d_to_3d(
    state_dict: Dict[str, torch.Tensor],
    conv1_mode: str = "channels_to_depth",
    conv1_kD: int = None,
    conv2_mode: str = "repeat",
    conv2_kD: int = 1,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    new_sd = dict(state_dict)
    report: Dict[str, Any] = {"conv1": {}, "conv2": {}}

    # conv1
    if "conv1.weight" in state_dict:
        w1 = state_dict["conv1.weight"]
        report["conv1"]["original_shape"] = tuple(w1.shape)
        if conv1_mode == "channels_to_depth":
            w1_3d = _channels_to_depth_conv1(w1)
            report["conv1"]["mode"] = "channels_to_depth"
            report["conv1"]["new_shape"] = tuple(w1_3d.shape)
        elif conv1_mode in ("repeat", "avg"):
            if not conv1_kD or conv1_kD <= 0:
                raise ValueError("conv1_kD を正の整数で指定してください (repeat/avg モード)")
            w1_3d = _inflate_repeat(w1, kD=conv1_kD, scale=(conv1_mode == "avg"))
            report["conv1"]["mode"] = conv1_mode
            report["conv1"]["new_shape"] = tuple(w1_3d.shape)
        else:
            raise ValueError(f"Unknown conv1_mode: {conv1_mode}")
        new_sd["conv1.weight"] = w1_3d
    else:
        report["conv1"]["skipped"] = True

    # conv2
    if "conv2.weight" in state_dict:
        w2 = state_dict["conv2.weight"]
        report["conv2"]["original_shape"] = tuple(w2.shape)
        if conv2_mode == "skip":
            report["conv2"]["skipped"] = True
        elif conv2_mode in ("repeat", "avg"):
            if not isinstance(conv2_kD, int) or conv2_kD <= 0:
                raise ValueError("conv2_kD を正の整数で指定してください")
            w2_3d = _inflate_repeat(w2, kD=conv2_kD, scale=(conv2_mode == "avg"))
            new_sd["conv2.weight"] = w2_3d
            report["conv2"]["mode"] = conv2_mode
            report["conv2"]["new_shape"] = tuple(w2_3d.shape)
        else:
            raise ValueError(f"Unknown conv2_mode: {conv2_mode}")
    else:
        report["conv2"]["skipped"] = True

    return new_sd, report


def main():
    parser = argparse.ArgumentParser(description="2D stem 重みを 3D stem 用にインフレート/再配置するツール")
    parser.add_argument(
        "--input",
        type=str,
        default="development/ball_tracking/hrnet_finetuning/weight/wasb_tennis_best.pth.tar",
        help="入力チェックポイント (.pth/.pth.tar)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="development/ball_tracking/hrnet_finetuning/weight/wasb_tennis_best_3dstem.pth.tar",
        help="出力チェックポイントパス",
    )
    parser.add_argument(
        "--conv1_mode",
        type=str,
        choices=["channels_to_depth", "repeat", "avg"],
        default="channels_to_depth",
        help="conv1 の変換モード: channels_to_depth=入力チャネル(3*F)を深さFへ再配置 / repeat=深さ方向に複製 / avg=複製して平均化",
    )
    parser.add_argument(
        "--conv1_kD",
        type=int,
        default=None,
        help="conv1 の深さカーネルサイズ (repeat/avg モード時のみ必須)",
    )
    parser.add_argument(
        "--conv2_mode",
        type=str,
        choices=["repeat", "avg", "skip"],
        default="repeat",
        help="conv2 の変換モード: repeat/avg/skip",
    )
    parser.add_argument(
        "--conv2_kD",
        type=int,
        default=1,
        help="conv2 の深さカーネルサイズ (repeat/avg 時)",
    )
    parser.add_argument("--dry_run", action="store_true", help="保存せずに形状のみ表示")

    args = parser.parse_args()

    ckpt_obj = _load_ckpt(args.input)
    state_dict, meta = _extract_state_dict(ckpt_obj)

    new_sd, rpt = convert_stem_2d_to_3d(
        state_dict,
        conv1_mode=args.conv1_mode,
        conv1_kD=args.conv1_kD,
        conv2_mode=args.conv2_mode,
        conv2_kD=args.conv2_kD,
    )

    print("[conv1]")
    print(rpt["conv1"])
    print("[conv2]")
    print(rpt["conv2"])

    if args.dry_run:
        print("dry_run 指定のため保存しません。")
        return

    out_obj = _inject_state_dict(ckpt_obj, new_sd, meta)
    out_dir = os.path.dirname(args.output)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    torch.save(out_obj, args.output)
    print(f"保存しました: {args.output}")


if __name__ == "__main__":
    main()
