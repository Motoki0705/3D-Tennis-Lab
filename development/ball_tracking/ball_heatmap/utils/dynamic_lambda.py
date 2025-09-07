from typing import Dict

from omegaconf import DictConfig


class DynamicLambdaController:
    """
    Manages the dynamic weighting of multiple loss terms.
    Supports static weights, scheduled ramps, and stagnation triggers.
    """

    def __init__(self, config: DictConfig):
        self.config = config
        self.base_lambdas: Dict[str, float] = {}
        self._lambdas: Dict[str, float] = {}

        if hasattr(config.training, "loss"):
            for key, value in config.training.loss.items():
                if key.startswith("lambda_"):
                    loss_name = key.replace("lambda_", "")
                    self.base_lambdas[loss_name] = float(value)
        self._lambdas = self.base_lambdas.copy()

        self.stagnation_cfg = getattr(config.training, "stagnation_trigger", {})
        if self.stagnation_cfg.get("enabled", False):
            self.stagnation_metric = self.stagnation_cfg["metric"]
            self.stagnation_patience = self.stagnation_cfg["patience"]
            self.stagnation_counter = 0
            self.best_metric_val = float("inf")

    def get_lambdas(self) -> Dict[str, float]:
        return self._lambdas

    def update_on_epoch_start(self, current_epoch: int):
        warmup_epochs = self.config.training.get("warmup_epochs", 0)
        ramp_epochs = self.config.training.get("ramp_epochs", 0)

        if current_epoch < warmup_epochs:
            self._lambdas["adv"] = 0.0
            self._lambdas["self"] = 0.0
        elif current_epoch < warmup_epochs + ramp_epochs:
            ramp_progress = (current_epoch - warmup_epochs) / ramp_epochs
            self._lambdas["self"] = self.base_lambdas.get("self", 0.0) * ramp_progress
            self._lambdas["adv"] = 0.0  # Keep GAN off during SSL ramp
        else:
            self._lambdas["adv"] = self.base_lambdas.get("adv", 0.0)
            self._lambdas["self"] = self.base_lambdas.get("self", 0.0)

    def update_on_validation_end(self, val_metrics: Dict[str, float]):
        if not self.stagnation_cfg.get("enabled", False):
            return

        metric_val = val_metrics.get(self.stagnation_metric)
        if metric_val is None:
            return

        if metric_val < self.best_metric_val:
            self.best_metric_val = metric_val
            self.stagnation_counter = 0
        else:
            self.stagnation_counter += 1

        if self.stagnation_counter >= self.stagnation_patience:
            print(f"\nStagnation detected on metric {self.stagnation_metric}! Adjusting lambdas.")
            self._lambdas["adv"] = self._lambdas.get("adv", 0.0) * 0.9
            self._lambdas["sharp"] = self._lambdas.get("sharp", 0.0) * 1.1
            self.stagnation_counter = 0

    def __getitem__(self, key: str) -> float:
        return self._lambdas.get(key, 0.0)
