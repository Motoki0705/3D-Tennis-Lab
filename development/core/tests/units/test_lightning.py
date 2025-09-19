from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from torch.utils.data import Dataset

from development.core.lightning.base_datamodule import BaseDataModule
from development.core.lightning.base_lit_module import BaseLitModule


class _TrackingDataset(Dataset):
    def __init__(self, length: int = 10):
        object.__setattr__(self, "_length", length)
        object.__setattr__(self, "_transform_assignments", [])
        object.__setattr__(self, "_transform", None)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return self._length

    def __getitem__(self, idx):  # pragma: no cover - not exercised
        data = torch.tensor([float(idx)])
        target = torch.tensor([float(idx)])
        current = self._transform
        if callable(current):
            return current((data, target))
        return data, target

    def __setattr__(self, name, value):
        if name == "transform":
            self._transform_assignments.append(value)
            object.__setattr__(self, "_transform", value)
        else:
            object.__setattr__(self, name, value)

    @property
    def transform(self):  # pragma: no cover - trivial accessor
        return self._transform

    @property
    def transform_assignments(self):
        return list(self._transform_assignments)


@pytest.mark.unit
def test_base_datamodule_split_and_transform_assignment():
    dataset = _TrackingDataset(length=10)
    config = SimpleNamespace(
        dataset={"foo": "bar"},
        dataloader={"batch_size": 2, "num_workers": 0, "pin_memory": False, "persistent_workers": False},
        splits={"train_ratio": 0.6, "val_ratio": 0.2, "test_ratio": 0.2},
    )
    datamodule = BaseDataModule(
        config=config,
        dataset=dataset,
        train_transforms="train_tf",
        val_transforms="val_tf",
        test_transforms="test_tf",
    )

    datamodule.setup()

    assert len(datamodule.train_dataset) + len(datamodule.val_dataset) + len(datamodule.test_dataset) == len(dataset)
    assert len(dataset.transform_assignments) == 3
    assert dataset.transform_assignments == ["train_tf", "val_tf", "test_tf"]

    train_loader = datamodule.train_dataloader()
    assert train_loader.batch_size == 2
    assert train_loader.num_workers == 0


@pytest.mark.unit
def test_base_lit_module_steps_and_optimizer(monkeypatch):
    model = torch.nn.Linear(4, 2)
    loss_fn = torch.nn.MSELoss()

    config = SimpleNamespace(training=SimpleNamespace(lr=0.01, weight_decay=0.0))

    def dummy_metric(preds, targets):  # pragma: no cover - deterministic small metric
        return torch.tensor(0.0)

    lit_module = BaseLitModule(config=config, model=model, loss_fn=loss_fn, metric_fns={"dummy": dummy_metric})

    lit_module.log = MagicMock()
    batch = (torch.randn(3, 4), torch.randn(3, 2))
    loss = lit_module.training_step(batch, 0)
    assert isinstance(loss, torch.Tensor)
    lit_module.log.assert_any_call("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)

    lit_module.log = MagicMock()
    val_loss = lit_module.validation_step(batch, 0)
    assert isinstance(val_loss, torch.Tensor)
    logged_keys = [args[0] for args, _ in lit_module.log.call_args_list]
    assert "val/loss" in logged_keys
    assert "val/dummy" in logged_keys

    lit_module.log = MagicMock()
    lit_module.test_step(batch, 0)
    logged_keys = [args[0] for args, _ in lit_module.log.call_args_list]
    assert "test/loss" in logged_keys
    assert "test/dummy" in logged_keys

    optim_spec = lit_module.configure_optimizers()
    assert "optimizer" in optim_spec
    assert "lr_scheduler" in optim_spec
    assert optim_spec["lr_scheduler"]["monitor"] == "val/loss"
