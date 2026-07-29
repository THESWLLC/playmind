from __future__ import annotations

import json
from pathlib import Path

import pytest

from playmind.planner_training.evaluate import evaluate_backends
from playmind.planner_training.export_gguf import main as gguf_main
from playmind.planner_training.presets import PRESETS, get_preset
from playmind.planner_training.train_dpo import DPOTrainingConfig, train_dpo
from playmind.planner_training.train_sft import SFTTrainingConfig, train_sft
from playmind.planner_v2.model_registry import ModelRegistry


def test_presets_validate_expected_4070_ti_limits() -> None:
    preset = get_preset("rtx_4070_ti_3b_qlora", base_model="local/model")
    assert preset.max_seq_length == 1024
    assert preset.micro_batch_size == 1
    assert preset.gradient_accumulation_steps == 16
    assert preset.gradient_checkpointing
    assert preset.lora_r == 16
    assert preset.lora_alpha == 32
    assert preset.load_in_4bit
    assert preset.quant_type == "nf4"
    assert PRESETS["rtx_4070_ti_7b_qlora_experimental"].experimental
    with pytest.raises(ValueError, match="requires base_model"):
        get_preset("rtx_4070_ti_3b_qlora")


def test_sft_smoke_uses_synthetic_data_and_registers_only_candidate(
    tmp_path: Path,
) -> None:
    registry_path = tmp_path / "registry.sqlite"
    result = train_sft(
        SFTTrainingConfig(
            smoke=True,
            runs_root=tmp_path / "runs",
            registry_path=registry_path,
            run_id="sft-smoke",
        )
    )
    assert result["status"] == "completed"
    assert result["metrics"]["train_steps"] == 2
    assert Path(result["manifest_path"]).is_file()
    assert (Path(result["run_dir"]) / "metrics.csv").is_file()
    registry = ModelRegistry(registry_path)
    assert registry.get("sft-smoke")["status"] == "candidate"
    assert registry.get_production() is None


def test_dpo_smoke_is_dependency_free_candidate(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.sqlite"
    result = train_dpo(
        DPOTrainingConfig(
            smoke=True,
            runs_root=tmp_path / "runs",
            registry_path=registry_path,
            run_id="dpo-smoke",
        )
    )
    assert result["status"] == "completed"
    assert result["metrics"]["train_preferences"] == 2
    assert ModelRegistry(registry_path).get("dpo-smoke")["status"] == "candidate"


def test_evaluate_compares_mock_backends_and_writes_all_reports(
    tmp_path: Path,
) -> None:
    scenarios = [
        {
            "scenario_id": "recover",
            "category": "recovery",
            "planner_state": {
                "available_skills": ["recover_health", "wait"],
            },
            "expected_plan": {"skills": ["recover_health"]},
        },
        {
            "scenario_id": "wait",
            "category": "unknown",
            "planner_state": {"available_skills": ["wait"]},
            "expected_plan": {"skills": ["wait"]},
        },
    ]

    def perfect(_state):
        return {"skills": ["recover_health"]}

    responses = iter(
        [
            json.dumps({"skills": ["recover_health"]}),
            "not json",
        ]
    )
    report = evaluate_backends(
        {"mapping": perfect, "mixed": lambda _state: next(responses)},
        scenarios,
        output_dir=tmp_path,
    )
    assert set(report["backends"]) == {"mapping", "mixed"}
    assert report["backends"]["mixed"]["metrics"]["json_fail_rate"] == 0.5
    assert report["backends"]["mapping"]["metrics"]["benchmark_components"]
    assert all(Path(path).is_file() for path in report["artifacts"].values())


def test_evaluation_updates_candidate_metrics_without_promotion(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path / "registry.sqlite")
    registry.register("candidate", status="candidate")
    scenario = {
        "scenario_id": "wait",
        "category": "unknown",
        "planner_state": {"available_skills": ["wait"]},
        "expected_plan": {"skills": ["wait"]},
    }
    backend = lambda _state: {"skills": ["wait"]}
    report = evaluate_backends(
        {"candidate": backend, "baseline": backend},
        [scenario],
        output_dir=tmp_path / "reports",
        registry=registry,
        registry_model_ids={"candidate": "candidate"},
    )
    assert report["backends"]["candidate"]["registry_status"] == "candidate"
    assert registry.get("candidate")["status"] == "candidate"
    assert registry.get_production() is None


def test_gguf_missing_converter_returns_nonzero_with_setup_help(
    tmp_path: Path, capsys
) -> None:
    model = tmp_path / "merged"
    model.mkdir()
    missing = tmp_path / "missing" / "convert_hf_to_gguf.py"
    code = gguf_main(
        [
            "--model-path",
            str(model),
            "--output",
            str(tmp_path / "model.gguf"),
            "--converter",
            str(missing),
        ]
    )
    assert code != 0
    assert "llama.cpp GGUF converter was not found" in capsys.readouterr().err
