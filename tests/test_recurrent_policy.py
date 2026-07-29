from __future__ import annotations

import json
from pathlib import Path

import pytest

from playmind.models.feature_schema import FEATURE_DIM
from playmind.models.policy_v2 import LegacyCheckpointError, SkillPolicyV2, TORCH_AVAILABLE
from playmind.models.recurrent_policy import (
    RecurrentSkillPolicyNet,
    RecurrentSkillPolicyV2,
)


pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")


def test_sequence_uses_more_than_final_timestep() -> None:
    import torch

    net = RecurrentSkillPolicyNet(FEATURE_DIM, 3, seed=7)
    net.eval()
    first = torch.zeros(1, 3, FEATURE_DIM)
    second = first.clone()
    second[0, 0, 0] = 4.0
    logits_a, _ = net(first)
    logits_b, _ = net(second)
    assert not torch.allclose(logits_a, logits_b)


def test_padding_does_not_affect_valid_output() -> None:
    import torch

    net = RecurrentSkillPolicyNet(FEATURE_DIM, 2, seed=3)
    net.eval()
    valid = torch.randn(1, 2, FEATURE_DIM)
    left_padded = torch.cat((torch.full((1, 2, FEATURE_DIM), 99.0), valid), dim=1)
    logits_valid, _ = net(valid, lengths=[2])
    logits_padded, _ = net(
        left_padded,
        lengths=[2],
        padding_mask=[[False, False, True, True]],
    )
    assert torch.allclose(logits_valid, logits_padded, atol=1e-6)


def test_variable_lengths_and_single_step() -> None:
    import torch

    net = RecurrentSkillPolicyNet(FEATURE_DIM, 4, seed=1)
    features = torch.randn(3, 5, FEATURE_DIM)
    logits, aux = net(features, lengths=[5, 3, 1])
    assert logits.shape == (3, 4)
    assert all(value.shape == (3,) for value in aux.values())
    single_logits, _ = net(torch.randn(2, FEATURE_DIM))
    assert single_logits.shape == (2, 4)


def test_checkpoint_round_trip_and_schema_rejection(tmp_path: Path) -> None:
    import torch

    policy = RecurrentSkillPolicyV2(
        ["explore", "wait"], trained=True, seed=11, history_length=4
    )
    sequence = [[[0.0] * FEATURE_DIM, [0.25] * FEATURE_DIM]]
    before, _ = policy.predict_sequence(sequence, lengths=[2])
    checkpoint = policy.save(tmp_path / "recurrent.json")
    loaded = RecurrentSkillPolicyV2.load(checkpoint)
    after, _ = loaded.predict_sequence(sequence, lengths=[2])
    assert torch.allclose(before, after)
    assert loaded.history_length == 4

    metadata = json.loads(checkpoint.read_text(encoding="utf-8"))
    metadata["feature_schema_version"] = 999
    checkpoint.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="feature_schema_version"):
        RecurrentSkillPolicyV2.load(checkpoint)


def test_allowed_skill_mask_is_applied_before_softmax() -> None:
    import torch

    policy = RecurrentSkillPolicyV2(["forbidden", "allowed"], trained=True)
    with torch.no_grad():
        for parameter in policy._net.parameters():
            parameter.zero_()
        policy._net.skill_head.bias.copy_(torch.tensor([100.0, -100.0]))
    decision = policy.choose_skill(
        {"feature_sequence": [[0.0] * FEATURE_DIM]}, ["allowed"]
    )
    assert decision.skill == "allowed"
    assert decision.confidence == pytest.approx(1.0)


def test_architecture_specific_loaders_reject_other_checkpoint(tmp_path: Path) -> None:
    recurrent = RecurrentSkillPolicyV2(["wait"])
    path = recurrent.save(tmp_path / "recurrent.json")
    with pytest.raises(LegacyCheckpointError, match="recurrent checkpoint"):
        SkillPolicyV2.load(path)
