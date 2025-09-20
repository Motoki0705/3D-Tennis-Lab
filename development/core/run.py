"""Hydra-driven entrypoint that wires experiments into the shared core stack."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from hydra import compose, initialize_config_dir, main as hydra_main
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig, OmegaConf

from .callbacks.factory import build_callbacks as build_core_callbacks

_LOGGER = logging.getLogger(__name__)


def _merge_experiment_config(cfg: DictConfig) -> DictConfig:
    """Merge the core config with an experiment-specific layer when provided."""

    exp_dir = cfg.get("experiment_config_dir")
    if not exp_dir:
        return cfg

    abs_exp_dir = Path(to_absolute_path(str(exp_dir))).resolve()
    if not abs_exp_dir.exists():
        raise FileNotFoundError(f"Experiment config directory not found: {abs_exp_dir}")

    core_config_dir = Path(__file__).resolve().parent / "configs"
    search_override = f"hydra.searchpath=[file://{core_config_dir.as_posix()}, file://{abs_exp_dir.as_posix()}]"

    # Compose the experiment config using its own search path, then merge with the runtime cfg
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=str(abs_exp_dir)):
        exp_cfg = compose(config_name="config", overrides=[search_override])

    merged = OmegaConf.merge(exp_cfg, cfg)
    return merged  # type: ignore[return-value]


def _instantiate_metrics(metrics_cfg: DictConfig | None) -> Dict[str, Any]:
    if metrics_cfg is None:
        return {}
    metrics: Dict[str, Any] = {}
    for name, spec in metrics_cfg.items():
        if spec is None:
            continue
        metrics[name] = instantiate(spec, _convert_="all")
    return metrics


def _extract_trainer_kwargs(training_cfg: DictConfig | None) -> Dict[str, Any]:
    if training_cfg is None:
        return {}
    container = OmegaConf.to_container(training_cfg, resolve=True)
    if not isinstance(container, dict):  # pragma: no cover - defensive guard
        return {}
    excluded = {"optimizer", "lr_scheduler"}
    return {k: v for k, v in container.items() if k not in excluded and v is not None}


def _build_logger(logger_cfg: DictConfig | None) -> TensorBoardLogger:
    if logger_cfg is None:
        save_dir = to_absolute_path("tb_logs")
        return TensorBoardLogger(save_dir=save_dir, name="default")
    save_dir = to_absolute_path(str(logger_cfg.get("save_dir", "tb_logs")))
    name = logger_cfg.get("name")
    version = logger_cfg.get("version")
    log_graph = bool(logger_cfg.get("log_graph", False))
    default_hp_metric = bool(logger_cfg.get("default_hp_metric", False))
    log_model = bool(logger_cfg.get("log_model", False))
    return TensorBoardLogger(
        save_dir=save_dir,
        name=name,
        version=version,
        log_graph=log_graph,
        default_hp_metric=default_hp_metric,
        log_model=log_model,
    )


def _resolve_ckpt_path(ckpt: Any) -> str | None:
    if ckpt in (None, "null", ""):
        return None
    if ckpt == "best":
        return ckpt
    return to_absolute_path(str(ckpt))


def _resolve_task(cfg: DictConfig) -> str:
    task = cfg.get("task", "train")
    if isinstance(task, str):
        return task.lower()
    return str(task).lower()


def _instantiate_optional(spec: DictConfig | None, **kwargs: Any) -> Any:
    if spec is None:
        return None
    return instantiate(spec, **kwargs)


def _describe_component(name: str, obj: Any) -> None:
    _LOGGER.info("%s: %s", name, obj.__class__.__name__ if obj is not None else "<none>")


def _run(cfg: DictConfig) -> None:
    full_cfg = _merge_experiment_config(cfg)
    seed = full_cfg.get("seed", 42)
    pl.seed_everything(int(seed), workers=True)

    datamodule = _instantiate_optional(full_cfg.get("datamodule"))
    if datamodule is None:
        raise ValueError("datamodule configuration must be provided by the experiment.")
    _describe_component("DataModule", datamodule)

    model = _instantiate_optional(full_cfg.get("model"))
    if model is None:
        raise ValueError("model configuration must be provided by the experiment.")
    _describe_component("Model", model)

    loss_fn = _instantiate_optional(full_cfg.get("loss"))
    if loss_fn is None:
        raise ValueError("loss configuration must be provided by the experiment.")
    _describe_component("Loss", loss_fn)

    metrics = _instantiate_metrics(full_cfg.get("metrics"))

    lit_module_cfg = full_cfg.get("lit_module")
    if lit_module_cfg is None:
        raise ValueError("lit_module configuration must be provided by the experiment.")
    lit_module = instantiate(
        lit_module_cfg,
        config=full_cfg,
        model=model,
        loss_fn=loss_fn,
        metric_fns=metrics,
        _convert_="partial",
    )
    _describe_component("LightningModule", lit_module)

    callbacks_cfg = full_cfg.get("callbacks")
    callbacks = build_core_callbacks(callbacks_cfg)

    logger = _build_logger(full_cfg.get("logger"))

    trainer_kwargs = _extract_trainer_kwargs(full_cfg.get("training"))
    trainer = pl.Trainer(logger=logger, callbacks=callbacks, **trainer_kwargs)

    task = _resolve_task(full_cfg)
    ckpt_path = _resolve_ckpt_path(full_cfg.get("ckpt_path"))

    if task in {"train", "fit"}:
        trainer.fit(lit_module, datamodule=datamodule, ckpt_path=ckpt_path)
    elif task in {"validate", "val"}:
        trainer.validate(lit_module, datamodule=datamodule, ckpt_path=ckpt_path)
    elif task in {"test"}:
        trainer.test(lit_module, datamodule=datamodule, ckpt_path=ckpt_path)
    elif task in {"predict"}:
        trainer.predict(lit_module, datamodule=datamodule, ckpt_path=ckpt_path)
    else:
        raise ValueError(f"Unsupported task '{task}'.")


@hydra_main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:  # pragma: no cover - exercised via CLI
    """Hydra main entrypoint."""

    _run(cfg)


if __name__ == "__main__":  # pragma: no cover - CLI guard
    main()
