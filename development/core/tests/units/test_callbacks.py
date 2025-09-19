from typing import Any

import pytest
from pytorch_lightning.callbacks import Callback

from development.core.callbacks.factory import build_callbacks, _instantiate_mapping


class SampleCallback(Callback):
    def __init__(self, value: Any = None):
        super().__init__()
        self.value = value


@pytest.mark.unit
def test_build_callbacks_flattens_nested_structures(monkeypatch):
    monkeypatch.setattr("development.core.callbacks.factory.hydra_instantiate", None)

    spec = {
        "primary": [
            None,
            {"_target_": "development.core.tests.units.test_callbacks.SampleCallback", "value": 10},
        ],
        "secondary": {
            "nested": {"_target_": "development.core.tests.units.test_callbacks.SampleCallback", "value": 20},
        },
    }

    callbacks = build_callbacks(spec)
    assert len(callbacks) == 2
    values = sorted(cb.value for cb in callbacks)
    assert values == [10, 20]


@pytest.mark.unit
def test_build_callbacks_accepts_instances_and_none(monkeypatch):
    monkeypatch.setattr("development.core.callbacks.factory.hydra_instantiate", None)

    existing = SampleCallback(value="keep")
    spec = [existing, None]
    callbacks = build_callbacks(spec)
    assert callbacks == [existing]


@pytest.mark.unit
def test_instantiate_mapping_requires_target(monkeypatch):
    monkeypatch.setattr("development.core.callbacks.factory.hydra_instantiate", None)
    with pytest.raises(ValueError):
        _instantiate_mapping({})
