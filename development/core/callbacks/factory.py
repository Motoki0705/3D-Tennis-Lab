"""Callback factory helpers shared across experiments.

Typical usage inside a training script::

    from development.core.callbacks import build_callbacks
    callbacks = build_callbacks(cfg.callbacks)

The factory supports Hydra-style configs (with ``_target_`` entries), nested
structures (dict/list), and raw ``Callback`` instances. ``null`` entries are
ignored so experiments can toggle callbacks through config overrides.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, Iterable, List, Mapping

from pytorch_lightning.callbacks import Callback

try:  # Hydra is optional but preferred when available.
    from hydra.utils import instantiate as hydra_instantiate
except Exception:  # pragma: no cover - hydra not installed
    hydra_instantiate = None  # type: ignore

try:  # OmegaConf gives nicer container conversion when present.
    from omegaconf import DictConfig, ListConfig, OmegaConf
except Exception:  # pragma: no cover - OmegaConf not installed
    DictConfig = ()  # type: ignore
    ListConfig = ()  # type: ignore
    OmegaConf = None  # type: ignore


def build_callbacks(spec: Any) -> List[Callback]:
    """Instantiate callbacks from a config fragment.

    Parameters
    ----------
    spec:
        A mapping/list/dictconfig that describes the callbacks section of a
        Hydra config. ``None`` entries are skipped and existing ``Callback``
        instances are returned as-is.
    """

    print(spec)
    callbacks: List[Callback] = []
    for obj in _flatten(spec):
        if obj is None:
            continue
        if isinstance(obj, Callback):
            callbacks.append(obj)
            continue
        if isinstance(obj, Mapping):
            callbacks.append(_instantiate_mapping(obj))
            continue
        raise TypeError(f"Unsupported callback specification: {type(obj)!r}")
    return callbacks


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _flatten(spec: Any) -> Iterable[Any]:
    if spec is None:
        return []
    if isinstance(spec, Callback):
        return [spec]
    if OmegaConf is not None and isinstance(spec, (DictConfig, ListConfig)):
        spec = OmegaConf.to_container(spec, resolve=True)
    if isinstance(spec, Mapping):
        if "_target_" in spec:
            return [spec]
        flattened: List[Any] = []
        for value in spec.values():
            flattened.extend(list(_flatten(value)))
        return flattened
    if isinstance(spec, (list, tuple, set)):
        flattened: List[Any] = []
        for item in spec:
            flattened.extend(list(_flatten(item)))
        return flattened
    return [spec]


def _instantiate_mapping(cfg: Mapping[str, Any]) -> Callback:
    if "_target_" not in cfg:
        raise ValueError("Callback config missing '_target_' entry.")
    if hydra_instantiate is not None:
        return hydra_instantiate(cfg)  # type: ignore[arg-type]
    # Fallback manual instantiation when Hydra is absent.
    target = cfg.get("_target_", "")
    module_path, _, attr = target.rpartition(".")
    if not module_path:
        raise ValueError(f"Invalid callback target path: {target}")
    module = import_module(module_path)
    factory = getattr(module, attr)
    kwargs = {k: v for k, v in cfg.items() if k != "_target_"}
    return factory(**kwargs)
