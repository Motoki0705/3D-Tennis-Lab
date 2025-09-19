import types

import numpy as np
import pytest
import torch

from development.core.data_core import replay as replay_mod


class DummyReplayCompose:
    """Minimal stub mimicking Albumentations ReplayCompose for unit tests."""

    def __call__(self, *, image, keypoints=None, bboxes=None, class_labels=None):
        self.last_call = {
            "image": np.array(image, copy=True),
            "keypoints": keypoints,
            "bboxes": bboxes,
            "class_labels": class_labels,
        }
        result = {"image": np.array(image, copy=True), "replay": {"seed": 123}}
        if keypoints is not None:
            result["keypoints"] = keypoints
        if bboxes is not None:
            result["bboxes"] = bboxes
        if class_labels is not None:
            result["class_labels"] = class_labels
        return result

    @staticmethod
    def replay(state, *, image, keypoints=None, bboxes=None, class_labels=None):
        result = {"image": np.array(image, copy=True)}
        if keypoints is not None:
            result["keypoints"] = keypoints
        if bboxes is not None:
            result["bboxes"] = bboxes
        if class_labels is not None:
            result["class_labels"] = class_labels
        return result


@pytest.fixture(autouse=True)
def _patch_albumentations(monkeypatch):
    """Ensure replay helpers see a ReplayCompose implementation."""

    dummy_module = types.SimpleNamespace(ReplayCompose=DummyReplayCompose)
    monkeypatch.setattr(replay_mod, "A", dummy_module)
    yield


@pytest.mark.unit
def test_clip_replay_adapter_preserves_shapes_and_targets():
    pipeline = DummyReplayCompose()
    adapter = replay_mod.make_clip_replay_adapter(
        pipeline,
        keypoints_field="keypoints",
        bboxes_field="bboxes",
        classes_field="classes",
    )

    clip = torch.rand(2, 3, 4, 4)
    keypoints = [[(1.0, 2.0)], [(3.0, 1.0)]]
    bboxes = [[(0.0, 0.0, 2.0, 2.0)], [(1.0, 1.0, 2.0, 2.0)]]
    classes = [[5], [6]]
    sample = {"inputs": clip.clone(), "targets": {"keypoints": keypoints, "bboxes": bboxes, "classes": classes}}

    output = adapter(sample)

    assert isinstance(output["inputs"], torch.Tensor)
    assert output["inputs"].shape == (2, 3, 4, 4)
    assert output["targets"]["keypoints"] == keypoints
    assert output["targets"]["bboxes"] == bboxes
    assert output["targets"]["classes"] == classes


@pytest.mark.unit
def test_clip_replay_adapter_accepts_float_tensor_inputs():
    pipeline = DummyReplayCompose()
    adapter = replay_mod.make_clip_replay_adapter(pipeline)

    clip = torch.rand(1, 3, 2, 2, dtype=torch.float32)
    sample = {"inputs": clip.clone(), "targets": {}}
    output = adapter(sample)

    assert output["inputs"].shape == (1, 3, 2, 2)
    assert torch.allclose(output["inputs"], clip, atol=1e-2)
