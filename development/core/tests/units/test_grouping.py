import pytest

from development.core.data_core import grouping


@pytest.mark.unit
def test_group_clips_handles_alphanumeric_filenames():
    frames = [
        {"id": 1, "file_name": "clips/set1/PuXlxKdUIes_2450.png", "width": 1280, "height": 720, "frame_id": "f001"},
        {"id": 2, "file_name": "clips/set1/3iQCaCGROsE_650.png", "width": 1280, "height": 720, "frame_id": "frame-2"},
        {"id": 3, "file_name": "clips/set1/A2_frame.png", "width": 1280, "height": 720, "frame_id": None},
    ]

    clips = grouping.group_clips(frames)
    assert len(clips) == 1
    ordered_names = [frame["file_name"] for frame in clips[0]["frames"]]
    assert ordered_names[0].endswith("3iQCaCGROsE_650.png")
    assert ordered_names[-1].endswith("PuXlxKdUIes_2450.png")


@pytest.mark.unit
def test_enumerate_sequences_respects_partial_and_drop_short():
    total = 3
    sequences = grouping.enumerate_sequences(
        total_frames=total,
        sequence_length=4,
        frame_stride=2,
        allow_partial_last=True,
        drop_short_clips=False,
    )
    assert sequences == [[0, 1, 2]]

    sequences_drop = grouping.enumerate_sequences(
        total_frames=total,
        sequence_length=4,
        frame_stride=2,
        allow_partial_last=True,
        drop_short_clips=True,
    )
    assert sequences_drop == []
