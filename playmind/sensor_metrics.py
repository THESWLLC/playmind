"""Sensor accuracy metrics: compare predictions against human labels.

Tracks binary counters (target, combat, death, ghost, movement, modal,
hostile) plus continuous health / objective MAE and a life-phase confusion
matrix. Reports land under ``data/playmind/labels/``.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

DEFAULT_LABELS_DIR = Path("data/playmind/labels")
DEFAULT_LABELS_PATH = DEFAULT_LABELS_DIR / "sensor_labels.jsonl"
DEFAULT_REPORT_PATH = DEFAULT_LABELS_DIR / "sensor_metrics_report.json"

BINARY_SENSORS = (
    "target",
    "combat",
    "death",
    "ghost",
    "movement",
    "modal",
    "hostile",
)

CONTINUOUS_SENSORS = (
    "player_hp",
    "target_hp",
    "objective_progress",
)

# Canonical report key -> accepted field names on prediction / label dicts.
_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "target": ("has_target", "target"),
    "combat": ("in_combat", "combat"),
    "death": ("is_dead", "death"),
    "ghost": ("is_ghost", "ghost"),
    "movement": ("moving", "is_moving", "movement"),
    "modal": ("modal", "blocking_modal", "modal_menu"),
    "hostile": ("hostiles_near", "hostile", "hostiles"),
    "player_hp": ("player_hp", "vision_player_hp"),
    "target_hp": ("target_hp", "target_hp_est"),
    "objective_progress": ("objective_progress",),
}

LIFE_PHASES = (
    "alive",
    "dead_dialog",
    "confirm",
    "rez_picker",
    "ghost",
    "unknown",
)

_MOTION_MOVE_THRESHOLD = 0.5


@dataclass
class BinaryCounter:
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    def update(self, predicted: bool, label: bool) -> None:
        if predicted and label:
            self.tp += 1
        elif predicted and not label:
            self.fp += 1
        elif not predicted and not label:
            self.tn += 1
        else:
            self.fn += 1

    def metrics(self) -> dict[str, Optional[float]]:
        precision = _safe_div(self.tp, self.tp + self.fp)
        recall = _safe_div(self.tp, self.tp + self.fn)
        f1 = None
        if precision is not None and recall is not None and (precision + recall) > 0:
            f1 = 2.0 * precision * recall / (precision + recall)
        elif precision is not None and recall is not None:
            f1 = 0.0
        return {
            "tp": self.tp,
            "fp": self.fp,
            "tn": self.tn,
            "fn": self.fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "fpr": _safe_div(self.fp, self.fp + self.tn),
            "fnr": _safe_div(self.fn, self.fn + self.tp),
            "support": self.tp + self.fp + self.tn + self.fn,
        }


@dataclass
class ContinuousCounter:
    abs_err_sum: float = 0.0
    n: int = 0

    def update(self, predicted: float, label: float) -> None:
        self.abs_err_sum += abs(float(predicted) - float(label))
        self.n += 1

    def metrics(self) -> dict[str, Optional[float]]:
        return {
            "n": self.n,
            "mae": (self.abs_err_sum / self.n) if self.n else None,
            "abs_err_sum": self.abs_err_sum,
        }


@dataclass
class SensorMetrics:
    """Accumulate prediction-vs-label comparisons across labeled frames."""

    binary: dict[str, BinaryCounter] = field(
        default_factory=lambda: {name: BinaryCounter() for name in BINARY_SENSORS}
    )
    continuous: dict[str, ContinuousCounter] = field(
        default_factory=lambda: {name: ContinuousCounter() for name in CONTINUOUS_SENSORS}
    )
    life_phase_confusion: dict[str, dict[str, int]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(int))
    )
    n_labeled: int = 0
    n_compared: int = 0

    def update(self, prediction: Mapping[str, Any], label: Mapping[str, Any]) -> None:
        """Update all counters from one prediction / label pair."""
        self.n_compared += 1
        for name in BINARY_SENSORS:
            pred_b = _extract_bool(prediction, name)
            lab_b = _extract_bool(label, name)
            if pred_b is None or lab_b is None:
                continue
            self.binary[name].update(pred_b, lab_b)

        for name in CONTINUOUS_SENSORS:
            pred_f = _extract_float(prediction, name)
            lab_f = _extract_float(label, name)
            if pred_f is None or lab_f is None:
                continue
            self.continuous[name].update(pred_f, lab_f)

        pred_phase = _extract_life_phase(prediction)
        lab_phase = _extract_life_phase(label)
        if pred_phase is not None and lab_phase is not None:
            self.life_phase_confusion[lab_phase][pred_phase] += 1

    def update_from_pairs(
        self, pairs: Iterable[tuple[Mapping[str, Any], Mapping[str, Any]]]
    ) -> None:
        for pred, lab in pairs:
            self.update(pred, lab)

    def note_label(self) -> None:
        """Count a labeled frame (whether or not a prediction was compared)."""
        self.n_labeled += 1

    def compute_report(self) -> dict[str, Any]:
        binary_report = {name: counter.metrics() for name, counter in self.binary.items()}
        continuous_report = {
            name: counter.metrics() for name, counter in self.continuous.items()
        }
        health = {
            "player_hp": continuous_report.get("player_hp", {}),
            "target_hp": continuous_report.get("target_hp", {}),
        }
        confusion = {
            truth: dict(preds)
            for truth, preds in sorted(self.life_phase_confusion.items())
        }
        return {
            "n_labeled": self.n_labeled,
            "n_compared": self.n_compared,
            "binary": binary_report,
            "continuous": continuous_report,
            "health_mae": health,
            "life_phase_confusion": confusion,
            "summary": _summarize(binary_report, continuous_report),
        }


def update_from_prediction_vs_label(
    metrics: SensorMetrics,
    prediction: Mapping[str, Any],
    label: Mapping[str, Any],
) -> SensorMetrics:
    """Convenience wrapper: update metrics in place and return them."""
    metrics.update(prediction, label)
    return metrics


def compute_report(metrics: SensorMetrics) -> dict[str, Any]:
    return metrics.compute_report()


def save_report(
    report: Mapping[str, Any],
    path: Path | str | None = None,
) -> Path:
    out = Path(path) if path else DEFAULT_REPORT_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def load_report(path: Path | str | None = None) -> dict[str, Any]:
    src = Path(path) if path else DEFAULT_REPORT_PATH
    return json.loads(src.read_text(encoding="utf-8-sig"))


def load_labels_jsonl(path: Path | str | None = None) -> list[dict[str, Any]]:
    src = Path(path) if path else DEFAULT_LABELS_PATH
    if not src.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in src.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def append_label_jsonl(
    record: Mapping[str, Any],
    path: Path | str | None = None,
) -> Path:
    out = Path(path) if path else DEFAULT_LABELS_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(dict(record), sort_keys=True) + "\n")
    return out


def report_to_markdown(report: Mapping[str, Any]) -> str:
    """Render a compute_report() dict as a Markdown summary."""
    lines = [
        "# Sensor metrics report",
        "",
        f"- **n_labeled**: {report.get('n_labeled', 0)}",
        f"- **n_compared**: {report.get('n_compared', 0)}",
        "",
        "## Binary sensors",
        "",
        "| sensor | precision | recall | F1 | FPR | FNR | support |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, m in (report.get("binary") or {}).items():
        lines.append(
            "| {sensor} | {precision} | {recall} | {f1} | {fpr} | {fnr} | {support} |".format(
                sensor=name,
                precision=_fmt(m.get("precision")),
                recall=_fmt(m.get("recall")),
                f1=_fmt(m.get("f1")),
                fpr=_fmt(m.get("fpr")),
                fnr=_fmt(m.get("fnr")),
                support=m.get("support", 0),
            )
        )
    lines.extend(["", "## Health MAE", ""])
    health = report.get("health_mae") or {}
    for name in ("player_hp", "target_hp"):
        m = health.get(name) or {}
        lines.append(f"- **{name}**: MAE={_fmt(m.get('mae'))} (n={m.get('n', 0)})")
    obj = (report.get("continuous") or {}).get("objective_progress") or {}
    lines.append(
        f"- **objective_progress**: MAE={_fmt(obj.get('mae'))} (n={obj.get('n', 0)})"
    )
    lines.extend(["", "## Life-phase confusion (rows=truth, cols=pred)", ""])
    confusion = report.get("life_phase_confusion") or {}
    if not confusion:
        lines.append("_no life_phase comparisons_")
    else:
        phases = sorted(
            {
                *confusion.keys(),
                *(p for row in confusion.values() for p in row.keys()),
            }
        )
        lines.append("| truth \\ pred | " + " | ".join(phases) + " |")
        lines.append("| --- | " + " | ".join("---:" for _ in phases) + " |")
        for truth in phases:
            row = confusion.get(truth) or {}
            cells = [str(row.get(pred, 0)) for pred in phases]
            lines.append(f"| {truth} | " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def metrics_from_labels_and_predictions(
    labels: Iterable[Mapping[str, Any]],
    predictions: Iterable[Mapping[str, Any]] | None = None,
    *,
    key_field: str = "frame",
) -> SensorMetrics:
    """Build metrics by joining labels to predictions on ``key_field`` (frame path)."""
    metrics = SensorMetrics()
    pred_by_key: dict[str, Mapping[str, Any]] = {}
    if predictions is not None:
        for pred in predictions:
            key = _frame_key(pred, key_field)
            if key:
                pred_by_key[key] = pred

    for label in labels:
        metrics.note_label()
        key = _frame_key(label, key_field)
        if key and key in pred_by_key:
            metrics.update(pred_by_key[key], label)
        elif predictions is None:
            # Labels-only: still count n_labeled; no comparisons.
            continue
    return metrics


# --- helpers -----------------------------------------------------------------


def _safe_div(num: float, den: float) -> Optional[float]:
    if den == 0:
        return None
    return float(num) / float(den)


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _summarize(
    binary_report: Mapping[str, Mapping[str, Any]],
    continuous_report: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    f1s = [
        m["f1"]
        for m in binary_report.values()
        if isinstance(m.get("f1"), (int, float))
    ]
    maes = [
        m["mae"]
        for name, m in continuous_report.items()
        if name in ("player_hp", "target_hp") and isinstance(m.get("mae"), (int, float))
    ]
    return {
        "mean_binary_f1": (sum(f1s) / len(f1s)) if f1s else None,
        "mean_health_mae": (sum(maes) / len(maes)) if maes else None,
    }


def _get_aliased(row: Mapping[str, Any], sensor: str) -> Any:
    for key in _FIELD_ALIASES.get(sensor, (sensor,)):
        if key in row and row[key] is not None:
            return row[key]
    return None


def _extract_bool(row: Mapping[str, Any], sensor: str) -> Optional[bool]:
    raw = _get_aliased(row, sensor)
    if raw is None and sensor == "movement":
        motion = row.get("motion")
        if motion is not None:
            try:
                return float(motion) >= _MOTION_MOVE_THRESHOLD
            except (TypeError, ValueError):
                return None
        return None
    if raw is None:
        return None
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return bool(raw)
    text = str(raw).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return None


def _extract_float(row: Mapping[str, Any], sensor: str) -> Optional[float]:
    raw = _get_aliased(row, sensor)
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _extract_life_phase(row: Mapping[str, Any]) -> Optional[str]:
    raw = row.get("life_phase")
    if raw is None or raw == "":
        return None
    return str(raw)


def _frame_key(row: Mapping[str, Any], key_field: str) -> str:
    for field_name in (key_field, "frame", "path", "frame_path"):
        if field_name in row and row[field_name] is not None:
            return str(Path(str(row[field_name])).as_posix())
    return ""
