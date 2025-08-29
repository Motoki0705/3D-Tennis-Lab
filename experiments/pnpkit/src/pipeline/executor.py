from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import Bundle, Stage
from ..spec.contract import can_run, ensure_versions


def _bundle_get(B: Bundle, path: str) -> Any:
    parts = path.split(".")
    cur: Any = B
    for p in parts:
        if isinstance(cur, dict):
            cur = cur.get(p)
        else:
            cur = getattr(cur, p, None)
        if cur is None:
            return None
    return cur


def _bundle_set(B: Bundle, path: str, value: Any) -> None:
    parts = path.split(".")
    cur: Any = B
    for p in parts[:-1]:
        nxt = None
        if isinstance(cur, dict):
            nxt = cur.get(p)
            if nxt is None:
                nxt = {}
                cur[p] = nxt
        else:
            nxt = getattr(cur, p, None)
            if nxt is None:
                setattr(cur, p, {})
                nxt = getattr(cur, p)
        cur = nxt
    last = parts[-1]
    if isinstance(cur, dict):
        cur[last] = value
    else:
        setattr(cur, last, value)


def _jsonable(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_jsonable(o) for o in obj]
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if hasattr(obj, "tolist"):
        try:
            return obj.tolist()
        except Exception:
            pass
    # fallback to string
    return str(obj)


def _fingerprint(stage: Stage, B: Bundle) -> str:
    reqs = getattr(stage, "required_inputs", []) or []
    payload = {
        "stage": getattr(stage, "STAGE_NAME", stage.__class__.__name__),
        "version": getattr(stage, "STAGE_VERSION", "0.0.0"),
        "cfg": _jsonable(getattr(stage, "cfg", {})),
        "inputs": {r: _jsonable(_bundle_get(B, r) if r not in {"frames"} else len(B.frames)) for r in reqs},
    }
    s = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _cache_dir(bundle_id: Optional[str]) -> Path:
    base = os.environ.get("PNPKIT_CACHE_DIR", os.path.join("experiments", "pnpkit", ".cache"))
    bid = str(bundle_id) if bundle_id else "__default__"
    p = Path(base) / bid
    p.mkdir(parents=True, exist_ok=True)
    return p


def _load_cache(bundle_id: Optional[str], stage: Stage, fp: str) -> Optional[Dict[str, Any]]:
    cdir = _cache_dir(bundle_id) / (getattr(stage, "STAGE_NAME", stage.__class__.__name__))
    meta = cdir / f"{fp}.json"
    if not meta.exists():
        return None
    try:
        with open(meta, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _save_cache(bundle_id: Optional[str], stage: Stage, fp: str, snapshot: Dict[str, Any]) -> None:
    cdir = _cache_dir(bundle_id) / (getattr(stage, "STAGE_NAME", stage.__class__.__name__))
    cdir.mkdir(parents=True, exist_ok=True)
    meta = cdir / f"{fp}.json"
    with open(meta, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, sort_keys=True)


def _snapshot_bundle(B: Bundle, produces: List[str]) -> Dict[str, Any]:
    snap: Dict[str, Any] = {}
    for p in produces:
        try:
            snap[p] = _jsonable(_bundle_get(B, p))
        except Exception:
            pass
    return snap


def _merge_snapshot(B: Bundle, snap: Dict[str, Any]) -> None:
    for k, v in (snap or {}).items():
        _bundle_set(B, k, v)


def run_dag(
    B: Bundle, stages: List[Stage], parallel: bool = False, max_workers: int = 2, use_cache: bool = True
) -> Bundle:
    ensure_versions(B)
    n = len(stages)
    done = [False] * n
    failed = [False] * n

    # Stage-local skip hint
    def _should_skip(stage: Stage) -> bool:
        return bool(getattr(stage, "should_skip", lambda _B: False)(B))

    progress = True
    while not all(done) and progress:
        progress = False
        # Collect ready tasks
        ready: List[int] = []
        reasons: Dict[int, List[str]] = {}
        for i, s in enumerate(stages):
            if done[i] or failed[i]:
                continue
            if _should_skip(s):
                done[i] = True
                progress = True
                # Mark version entry with cached/skipped flag
                B.report.setdefault("versions", {}).setdefault("stage_versions", []).append({
                    "name": getattr(s, "STAGE_NAME", s.__class__.__name__),
                    "version": getattr(s, "STAGE_VERSION", "0.0.0"),
                    "skipped": True,
                })
                continue
            ok, why = can_run(s, B)
            if ok:
                ready.append(i)
            else:
                reasons[i] = why

        if not ready:
            break

        def _execute(i: int):
            s = stages[i]
            # Cache
            if use_cache:
                fp = _fingerprint(s, B)
                snap = _load_cache(B.bundle_id, s, fp)
                if snap is not None:
                    _merge_snapshot(B, snap)
                    # record version with cached flag
                    B.report.setdefault("versions", {}).setdefault("stage_versions", []).append({
                        "name": getattr(s, "STAGE_NAME", s.__class__.__name__),
                        "version": getattr(s, "STAGE_VERSION", "0.0.0"),
                        "cached": True,
                    })
                    return (i, True, None)

            try:
                # Use Stage.__call__ to leverage pre/post/versions
                stages[i](B)
                # Snapshot outputs and save cache
                if use_cache:
                    fp = _fingerprint(stages[i], B)
                    snap = _snapshot_bundle(B, getattr(stages[i], "produces", []) or [])
                    _save_cache(B.bundle_id, stages[i], fp, snap)
                return (i, True, None)
            except Exception as e:
                return (i, False, str(e))

        if parallel and len(ready) > 1 and max_workers > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futs = {ex.submit(_execute, i): i for i in ready}
                for fut in as_completed(futs):
                    i, ok, err = fut.result()
                    if ok:
                        done[i] = True
                    else:
                        failed[i] = True
                        B.report.setdefault("errors", {}).setdefault("stages", {})[
                            getattr(stages[i], "STAGE_NAME", stages[i].__class__.__name__)
                        ] = err
                    progress = True
        else:
            for i in ready:
                i, ok, err = _execute(i)
                if ok:
                    done[i] = True
                else:
                    failed[i] = True
                    B.report.setdefault("errors", {}).setdefault("stages", {})[
                        getattr(stages[i], "STAGE_NAME", stages[i].__class__.__name__)
                    ] = err
                progress = True

    # Mark unresolved stages as skipped with reason
    if not all(done[i] or failed[i] for i in range(n)):
        B.report.setdefault("warnings", {}).setdefault("dag", [])
        for i, s in enumerate(stages):
            if not done[i] and not failed[i]:
                # unresolved
                B.report["warnings"]["dag"].append(
                    f"Stage '{getattr(s, 'STAGE_NAME', s.__class__.__name__)}' unresolved; unmet deps or skipped by design"
                )

    return B
