from __future__ import annotations

import json
from pathlib import Path

from playmind.planner_training.evaluate import evaluate_backends
from playmind.studio.eval_index import discover_reports, write_index


def test_evaluator_writes_run_report_and_normalized_index(tmp_path: Path) -> None:
    scenario = {
        "scenario_id": "wait",
        "category": "loading",
        "planner_state": {"available_skills": ["wait"]},
        "expected_plan": {"skills": ["wait"]},
    }
    backend = lambda _state: {"skills": ["wait"]}
    report = evaluate_backends(
        {"candidate": backend, "baseline": backend},
        [scenario],
        output_dir=tmp_path,
    )
    assert Path(report["artifacts"]["report"]).is_file()
    assert (tmp_path / "index.json").is_file()
    discovered = discover_reports(tmp_path)
    assert len(discovered) == 1
    assert set(discovered[0]["comparisons"]) == {"candidate", "baseline"}


def test_index_discovers_legacy_and_run_reports(tmp_path: Path) -> None:
    legacy = tmp_path / "planner_benchmark_old.json"
    legacy.write_text(
        json.dumps({"created_at": "2020", "backends": {"one": {"metrics": {}}}})
    )
    run = tmp_path / "runs" / "new"
    run.mkdir(parents=True)
    (run / "report.json").write_text(
        json.dumps({"created_at": "2021", "comparisons": {"two": {"score": 1}}})
    )
    index = write_index(tmp_path)
    assert index["report_count"] == 2
