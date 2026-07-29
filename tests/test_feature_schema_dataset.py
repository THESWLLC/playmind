from __future__ import annotations

from pathlib import Path

import pytest

from playmind.demonstrations import DemonstrationRecorder
from playmind.models.feature_schema import FEATURE_DIM, FEATURE_NAMES, structured_feature_vector_v2
from playmind.training.dataset import DemonstrationDataset, assert_no_split_leakage


def _index(name: str) -> int:
    return FEATURE_NAMES.index(name)


def test_unknown_false_and_unknown_zero_are_distinct() -> None:
    unknown = structured_feature_vector_v2({})
    known_false = structured_feature_vector_v2({"has_target": False})
    assert unknown[_index("has_target_value")] == known_false[_index("has_target_value")] == 0.0
    assert unknown[_index("has_target_known")] == 0.0
    assert known_false[_index("has_target_known")] == 1.0

    unknown_hp = structured_feature_vector_v2({})
    dead_hp = structured_feature_vector_v2({"player_hp": 0.0})
    assert unknown_hp[_index("player_hp_value")] == dead_hp[_index("player_hp_value")] == 0.0
    assert unknown_hp[_index("player_hp_known")] == 0.0
    assert dead_hp[_index("player_hp_known")] == 1.0


def _record(
    root: Path,
    *,
    session_id: str,
    episode_id: str,
    values: list[float],
    skill: str = "explore",
) -> Path:
    recorder = DemonstrationRecorder(root=root, session_id=session_id)
    recorder.start(episode_id=episode_id)
    for index, hp in enumerate(values):
        recorder.append(
            observation={"player_hp": hp, "has_target": False},
            skill=skill,
            episode_id=episode_id,
            timestamp=float(index),
        )
    recorder.mark("success")
    return recorder.stop()


def test_left_padding_mask_and_episode_boundaries(tmp_path: Path) -> None:
    root = tmp_path / "demos"
    _record(root, session_id="s1", episode_id="same", values=[0.1, 0.2])
    _record(root, session_id="s2", episode_id="same", values=[0.8, 0.9])
    dataset = DemonstrationDataset(root, history_length=4, split="all")
    assert len(dataset) == 4
    first = dataset[0]
    assert first["length"] == 1
    assert first["padding_mask"] == [False, False, False, True]
    assert first["features"][:3] == [[0.0] * FEATURE_DIM] * 3
    for item in dataset:
        valid_hp = [
            row[_index("player_hp_value")]
            for row, valid in zip(item["features"], item["padding_mask"])
            if valid
        ]
        assert not (any(value < 0.5 for value in valid_hp) and any(value > 0.5 for value in valid_hp))


def test_split_leakage_check_and_train_only_normalization(tmp_path: Path) -> None:
    root = tmp_path / "demos"
    for index in range(20):
        _record(
            root,
            session_id=f"s{index}",
            episode_id=f"episode-{index}",
            values=[index / 20.0],
            skill="explore" if index % 2 else "wait",
        )
    train = DemonstrationDataset(root, history_length=2, split="train", seed=4)
    val = DemonstrationDataset(root, history_length=2, split="val", seed=4)
    test = DemonstrationDataset(root, history_length=2, split="test", seed=4)
    assert_no_split_leakage(train, val, test)
    assert train.split_episode_keys().isdisjoint(val.split_episode_keys())
    assert train.split_episode_keys().isdisjoint(test.split_episode_keys())
    normalizer = train.fit_normalizer()
    assert len(normalizer.mean) == FEATURE_DIM
    with pytest.raises(ValueError, match="training split"):
        val.fit_normalizer()


def test_bad_and_unlabeled_samples_excluded_by_default(tmp_path: Path) -> None:
    root = tmp_path / "demos"
    recorder = DemonstrationRecorder(root=root, session_id="quality")
    recorder.start(episode_id="ep")
    good = recorder.append(
        observation={"player_hp": 1.0}, skill="wait", timestamp=1.0
    )
    bad = recorder.append(
        observation={"player_hp": 0.5}, skill="explore", timestamp=2.0
    )
    recorder.append(observation={"player_hp": 0.2}, timestamp=3.0)
    recorder.mark("bad", sample_id=bad["sample_id"])
    recorder.stop()

    default = DemonstrationDataset(root, split="all")
    assert len(default) == 1
    assert default[0]["sample_id"] == good["sample_id"]
    permissive = DemonstrationDataset(
        root, split="all", include_bad=True, include_unlabeled=True
    )
    assert len(permissive) == 3
