import argparse
import json
import os
from collections import OrderedDict, defaultdict
from typing import Any, Dict, Tuple

try:
    import torch
except ModuleNotFoundError:
    raise SystemExit("PyTorch (torch) が見つかりません。仮想環境を有効化するか、pip/poetryでインストールしてください。")


def _load_state_dict(ckpt_obj: Any) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """
    Try to extract a PyTorch state_dict from various checkpoint formats.

    Returns
    -------
    state_dict: Dict[str, Tensor]
        The parameter dictionary.
    meta: Dict[str, Any]
        Misc metadata such as root keys and original type.
    """
    meta: Dict[str, Any] = {
        "root_type": type(ckpt_obj).__name__,
        "root_keys": [],
        "selected_key": None,
    }

    # If this looks like a plain state_dict (OrderedDict of tensors)
    if isinstance(ckpt_obj, (dict, OrderedDict)):
        meta["root_keys"] = list(ckpt_obj.keys())
        # Common wrappers
        for key in [
            "state_dict",
            "model_state_dict",
            "model_state",
            "model",
            "net",
            "weights",
        ]:
            if key in ckpt_obj and isinstance(ckpt_obj[key], (dict, OrderedDict)):
                inner = ckpt_obj[key]
                # Some trainers wrap under nested 'state_dict'
                if isinstance(inner, (dict, OrderedDict)) and any(isinstance(v, torch.Tensor) for v in inner.values()):
                    meta["selected_key"] = key
                    return dict(inner), meta

        # Heuristic: a dict whose values are tensors is likely a state_dict
        if any(isinstance(v, torch.Tensor) for v in ckpt_obj.values()):
            return dict(ckpt_obj), meta

    raise ValueError("Unsupported checkpoint format: could not locate a state_dict-like mapping")


def _shape_of(t: torch.Tensor) -> Tuple[int, ...]:
    try:
        return tuple(t.shape)
    except Exception:
        return ()


def _summarize_prefixes(state_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    groups: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "count": 0,
            "param_count": 0,
            "tensors": {},
        }
    )
    total_params = 0
    for name, tensor in state_dict.items():
        prefix = name.split(".")[0]
        numel = int(tensor.numel())
        total_params += numel
        g = groups[prefix]
        g["count"] += 1
        g["param_count"] += numel
        g["tensors"][name] = {
            "shape": _shape_of(tensor),
            "dtype": str(tensor.dtype),
        }

    # Sort groups by param_count desc for readability
    ordered = dict(sorted(groups.items(), key=lambda kv: kv[1]["param_count"], reverse=True))
    return {"groups": ordered, "total_params": total_params, "num_tensors": len(state_dict)}


def _analyze_stem(state_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    stem: Dict[str, Any] = {}

    conv1 = state_dict.get("conv1.weight")
    bn1_w = state_dict.get("bn1.weight")
    bn1_b = state_dict.get("bn1.bias")
    conv2 = state_dict.get("conv2.weight")
    bn2_w = state_dict.get("bn2.weight")
    bn2_b = state_dict.get("bn2.bias")

    if conv1 is not None:
        s = _shape_of(conv1)
        stem["conv1.shape"] = s
        if len(s) == 4:  # Conv2d: (out_c, in_c, kH, kW)
            out_c, in_c, kH, kW = s
            frames_in = in_c // 3 if in_c % 3 == 0 else None
            stem["conv1.type"] = "Conv2d"
            stem["conv1.out_channels"] = out_c
            stem["conv1.in_channels"] = in_c
            stem["conv1.kernel"] = [kH, kW]
            stem["frames_in_inferred"] = frames_in
        elif len(s) == 5:  # Conv3d: (out_c, in_c, kD, kH, kW)
            out_c, in_c, kD, kH, kW = s
            stem["conv1.type"] = "Conv3d"
            stem["conv1.out_channels"] = out_c
            stem["conv1.in_channels"] = in_c
            stem["conv1.kernel"] = [kD, kH, kW]
        else:
            stem["conv1.type"] = f"Unknown(dim={len(s)})"

    if conv2 is not None:
        s = _shape_of(conv2)
        stem["conv2.shape"] = s
        if len(s) == 4:
            out_c, in_c, kH, kW = s
            stem["conv2.type"] = "Conv2d"
            stem["conv2.out_channels"] = out_c
            stem["conv2.in_channels"] = in_c
            stem["conv2.kernel"] = [kH, kW]
        elif len(s) == 5:
            out_c, in_c, kD, kH, kW = s
            stem["conv2.type"] = "Conv3d"
            stem["conv2.out_channels"] = out_c
            stem["conv2.in_channels"] = in_c
            stem["conv2.kernel"] = [kD, kH, kW]

    if bn1_w is not None:
        stem["bn1.num_features"] = int(bn1_w.numel())
    if bn1_b is not None:
        stem["bn1.has_bias"] = True
    if bn2_w is not None:
        stem["bn2.num_features"] = int(bn2_w.numel())
    if bn2_b is not None:
        stem["bn2.has_bias"] = True

    return stem


def _detect_hrnet_parts(state_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """Detect and summarize common HRNet component prefixes.

    This is tailored to development/ball_tracking/hrnet_finetuning/model/base_hrnet.py.
    """
    parts = {
        "layer1": {},
        "transition1": {},
        "stage2": {},
        "transition2": {},
        "stage3": {},
        "transition3": {},
        "stage4": {},
        "deconv_layers": {},
        "final_layers": {},
    }

    for name, t in state_dict.items():
        head = name.split(".")[0]
        if head in parts:
            parts[head][name] = {
                "shape": _shape_of(t),
                "dtype": str(t.dtype),
            }

    # Also provide quick counts per part
    summary = {}
    for k, v in parts.items():
        num_tensors = len(v)
        num_params = sum(int(state_dict[n].numel()) for n in v.keys())
        summary[k] = {
            "num_tensors": num_tensors,
            "num_params": num_params,
            "present": num_tensors > 0,
        }

    return {"by_part": summary}


def analyze_checkpoint(ckpt_path: str, as_json: bool = False) -> Dict[str, Any]:
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    obj = torch.load(ckpt_path, map_location="cpu")
    state_dict, meta = _load_state_dict(obj)

    prefix_summary = _summarize_prefixes(state_dict)
    stem = _analyze_stem(state_dict)
    parts = _detect_hrnet_parts(state_dict)

    report: Dict[str, Any] = {
        "checkpoint_path": ckpt_path,
        "meta": meta,
        "summary": prefix_summary,
        "stem": stem,
        "hrnet_parts": parts,
    }

    if as_json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        # Pretty print in human-friendly layout (Japanese)
        print(f"[Checkpoint] {ckpt_path}")
        print(f"- ルート型: {meta['root_type']}")
        root_keys = meta.get("root_keys") or []
        if root_keys:
            print(f"- ルートキー: {root_keys[:10]}" + (" ..." if len(root_keys) > 10 else ""))
        if meta.get("selected_key"):
            print(f"- 使用state_dictキー: {meta['selected_key']}")

        print("\n[全体サマリ]")
        print(f"- テンソル数: {prefix_summary['num_tensors']}")
        print(f"- 総パラメータ: {prefix_summary['total_params']:,}")

        print("\n[モジュール別 (先頭プレフィックス)]")
        for head, info in prefix_summary["groups"].items():
            print(f"  - {head}: tensors={info['count']}, params={info['param_count']:,}")

        if stem:
            print("\n[Stem 推定]")
            if "conv1.shape" in stem:
                print(f"  - conv1: type={stem.get('conv1.type')} shape={stem['conv1.shape']}")
                if stem.get("frames_in_inferred") is not None:
                    print(f"    推定frames_in: {stem['frames_in_inferred']}")
            if "conv2.shape" in stem:
                print(f"  - conv2: type={stem.get('conv2.type')} shape={stem['conv2.shape']}")
            if stem.get("bn1.num_features") is not None:
                print(f"  - bn1: num_features={stem['bn1.num_features']}")
            if stem.get("bn2.num_features") is not None:
                print(f"  - bn2: num_features={stem['bn2.num_features']}")

        print("\n[HRNet 各部位の存在確認]")
        for part, s in parts["by_part"].items():
            mark = "OK" if s["present"] else "-"
            print(f"  - {part}: {mark} (tensors={s['num_tensors']}, params={s['num_params']:,})")

        # Hint for 3D stem migration
        if stem.get("conv1.type") == "Conv2d":
            print(
                "\n[ヒント] 2D Stem → 3D Stem 変換:\n"
                "  Conv3d の kernel 深さ kD に対して、Conv2d 重みを kD に複製して\n"
                "  平均(またはスケール 1/kD)することで初期化可能です。\n"
                "  例: w3d[:, :, d, :, :] = w2d / kD  (for d in 0..kD-1)"
            )

    return report


def main():
    parser = argparse.ArgumentParser(description="HRNet用チェックポイント構造アナライザ")
    parser.add_argument(
        "--ckpt",
        type=str,
        default="development/ball_tracking/hrnet_finetuning/weight/wasb_tennis_best.pth.tar",
        help="解析するチェックポイント(.pth/.pth.tar)のパス",
    )
    parser.add_argument("--json", action="store_true", help="JSON 形式で出力する")
    args = parser.parse_args()

    analyze_checkpoint(args.ckpt, as_json=args.json)


if __name__ == "__main__":
    main()
