from __future__ import annotations

from omegaconf import DictConfig

from development.ball_tracking.ball_heatmap.strategies.semisup.none_strategy import NoneSemisup
from development.ball_tracking.ball_heatmap.strategies.semisup.selftrain_strategy import SelfTrainStrategy
from development.ball_tracking.ball_heatmap.strategies.semisup.mean_teacher_strategy import MeanTeacherStrategy

from development.ball_tracking.ball_heatmap.strategies.adversary.none_adversary import NoneAdversary
from development.ball_tracking.ball_heatmap.strategies.adversary.wgan_gp_adversary import WGAN_GP_Adversary


def build_semisup_strategy(cfg: DictConfig):
    name = cfg.semisup.name if cfg.semisup.get("enable", False) else "none"
    if name == "none":
        return NoneSemisup()
    if name == "selftrain":
        return SelfTrainStrategy(cfg)
    if name == "mean_teacher":
        return MeanTeacherStrategy(cfg)
    raise ValueError(f"Unknown semisup strategy: {name}")


def build_adversary(cfg: DictConfig):
    if not cfg.gan.get("enable", False):
        return NoneAdversary()
    name = cfg.gan.name
    if name == "wgan_gp":
        return WGAN_GP_Adversary(cfg)
    raise ValueError(f"Unknown adversary: {name}")
