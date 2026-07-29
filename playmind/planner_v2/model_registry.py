"""SQLite model registry with explicit, audited promotion gates."""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_REGISTRY_PATH = Path("data/playmind/planner/registry.sqlite")
MODEL_STATUSES = frozenset(
    {"production", "candidate", "experimental", "rejected", "archived"}
)
DEFAULT_PROMOTION_GATES: dict[str, Any] = {
    "valid_plan_rate": 0.99,
    "illegal_skill_rate": 0.005,
    "min_scenarios": 100,
    "benchmark_metric": "benchmark_score",
    "benchmark_margin": 0.0,
    "require_better_than_production": True,
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value: Mapping[str, Any] | None) -> str:
    return json.dumps(dict(value or {}), sort_keys=True, separators=(",", ":"), default=str)


def _loads(value: Any) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(str(value))
    except (json.JSONDecodeError, TypeError):
        return {}
    return dict(parsed) if isinstance(parsed, Mapping) else {}


class ModelRegistry:
    def __init__(
        self,
        path: str | Path = DEFAULT_REGISTRY_PATH,
        *,
        db_path: str | Path | None = None,
        promotion_gates: Mapping[str, Any] | None = None,
        gates: Mapping[str, Any] | None = None,
    ) -> None:
        self.path = Path(db_path or path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.promotion_gates = dict(DEFAULT_PROMOTION_GATES)
        self.promotion_gates.update(dict(promotion_gates or gates or {}))
        self._lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.path), timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS models (
                    model_id TEXT PRIMARY KEY,
                    display_name TEXT NOT NULL,
                    base_model TEXT,
                    adapter_path TEXT,
                    merged_path TEXT,
                    gguf_path TEXT,
                    quantization TEXT,
                    dataset_version TEXT,
                    train_metrics TEXT NOT NULL DEFAULT '{}',
                    eval_metrics TEXT NOT NULL DEFAULT '{}',
                    status TEXT NOT NULL,
                    reason TEXT,
                    created_at TEXT NOT NULL,
                    parent_id TEXT,
                    FOREIGN KEY(parent_id) REFERENCES models(model_id)
                );
                CREATE UNIQUE INDEX IF NOT EXISTS one_production_model
                    ON models(status) WHERE status = 'production';
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id TEXT,
                    action TEXT NOT NULL,
                    previous_status TEXT,
                    new_status TEXT,
                    reason TEXT,
                    warning INTEGER NOT NULL DEFAULT 0,
                    details TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );
                """
            )

    @staticmethod
    def _row(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        result = dict(row)
        result["train_metrics"] = _loads(result.get("train_metrics"))
        result["eval_metrics"] = _loads(result.get("eval_metrics"))
        return result

    @staticmethod
    def _validate_status(status: str) -> str:
        value = str(status).strip().lower()
        if value not in MODEL_STATUSES:
            raise ValueError(
                f"status must be one of {sorted(MODEL_STATUSES)}, got {status!r}"
            )
        return value

    @staticmethod
    def _audit(
        connection: sqlite3.Connection,
        *,
        model_id: str | None,
        action: str,
        previous_status: str | None,
        new_status: str | None,
        reason: str,
        warning: bool = False,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        connection.execute(
            """
            INSERT INTO audit_log (
                model_id, action, previous_status, new_status, reason,
                warning, details, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                model_id,
                action,
                previous_status,
                new_status,
                reason,
                int(bool(warning)),
                _json(details),
                _now(),
            ),
        )

    def register(
        self,
        model_id: str | Mapping[str, Any],
        *,
        display_name: str | None = None,
        base_model: str | None = None,
        adapter_path: str | None = None,
        merged_path: str | None = None,
        gguf_path: str | None = None,
        quantization: str | None = None,
        dataset_version: str | None = None,
        train_metrics: Mapping[str, Any] | None = None,
        eval_metrics: Mapping[str, Any] | None = None,
        status: str = "experimental",
        reason: str | None = None,
        created_at: str | None = None,
        parent_id: str | None = None,
        **extra: Any,
    ) -> dict[str, Any]:
        if isinstance(model_id, Mapping):
            data = dict(model_id)
            identifier = str(data.pop("model_id", "")).strip()
            display_name = data.pop("display_name", display_name)
            base_model = data.pop("base_model", base_model)
            adapter_path = data.pop("adapter_path", adapter_path)
            merged_path = data.pop("merged_path", merged_path)
            gguf_path = data.pop("gguf_path", gguf_path)
            quantization = data.pop("quantization", quantization)
            dataset_version = data.pop("dataset_version", dataset_version)
            train_metrics = data.pop("train_metrics", train_metrics)
            eval_metrics = data.pop("eval_metrics", eval_metrics)
            status = data.pop("status", status)
            reason = data.pop("reason", reason)
            created_at = data.pop("created_at", created_at)
            parent_id = data.pop("parent_id", parent_id)
            extra.update(data)
        else:
            identifier = str(model_id).strip()
        if not identifier:
            raise ValueError("model_id must be non-empty")
        state = self._validate_status(status)
        if extra:
            raise TypeError(f"unknown model fields: {', '.join(sorted(extra))}")
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO models (
                    model_id, display_name, base_model, adapter_path, merged_path,
                    gguf_path, quantization, dataset_version, train_metrics,
                    eval_metrics, status, reason, created_at, parent_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    identifier,
                    str(display_name or identifier),
                    base_model,
                    adapter_path,
                    merged_path,
                    gguf_path,
                    quantization,
                    dataset_version,
                    _json(train_metrics),
                    _json(eval_metrics),
                    state,
                    reason,
                    created_at or _now(),
                    parent_id,
                ),
            )
            self._audit(
                connection,
                model_id=identifier,
                action="register",
                previous_status=None,
                new_status=state,
                reason=str(reason or ""),
            )
        result = self.get(identifier)
        assert result is not None
        return result

    def list(
        self,
        status: str | None = None,
        *,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM models"
        params: list[Any] = []
        if status is not None:
            query += " WHERE status = ?"
            params.append(self._validate_status(status))
        query += " ORDER BY created_at DESC, model_id"
        if limit is not None:
            query += " LIMIT ?"
            params.append(max(0, int(limit)))
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [item for row in rows if (item := self._row(row)) is not None]

    list_models = list

    def get(self, model_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM models WHERE model_id = ?", (str(model_id),)
            ).fetchone()
        return self._row(row)

    def update_metrics(
        self,
        model_id: str,
        *,
        train_metrics: Mapping[str, Any] | None = None,
        eval_metrics: Mapping[str, Any] | None = None,
        reason: str = "metrics updated",
    ) -> dict[str, Any]:
        """Update benchmark data without changing or promoting model status."""
        current = self._require(model_id)
        next_train = (
            dict(train_metrics)
            if train_metrics is not None
            else dict(current["train_metrics"])
        )
        next_eval = (
            dict(eval_metrics)
            if eval_metrics is not None
            else dict(current["eval_metrics"])
        )
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                UPDATE models SET train_metrics = ?, eval_metrics = ?
                WHERE model_id = ?
                """,
                (_json(next_train), _json(next_eval), model_id),
            )
            self._audit(
                connection,
                model_id=model_id,
                action="update_metrics",
                previous_status=str(current["status"]),
                new_status=str(current["status"]),
                reason=reason,
                details={
                    "train_metrics_updated": train_metrics is not None,
                    "eval_metrics_updated": eval_metrics is not None,
                },
            )
        return self._require(model_id)

    def _require(self, model_id: str) -> dict[str, Any]:
        model = self.get(model_id)
        if model is None:
            raise KeyError(f"unknown model_id: {model_id!r}")
        return model

    def set_status(
        self,
        model_id: str,
        status: str,
        *,
        reason: str = "",
        warning: bool = False,
    ) -> dict[str, Any]:
        new_status = self._validate_status(status)
        current = self._require(model_id)
        if new_status == "production":
            raise ValueError("use promote() to set production status")
        with self._lock, self._connect() as connection:
            connection.execute(
                "UPDATE models SET status = ?, reason = ? WHERE model_id = ?",
                (new_status, reason, model_id),
            )
            self._audit(
                connection,
                model_id=model_id,
                action="set_status",
                previous_status=str(current["status"]),
                new_status=new_status,
                reason=reason,
                warning=warning,
            )
        result = self._require(model_id)
        return result

    @staticmethod
    def _metric(model: Mapping[str, Any], *names: str) -> float | None:
        for source_name in ("eval_metrics", "train_metrics"):
            source = model.get(source_name)
            if not isinstance(source, Mapping):
                continue
            for name in names:
                value = source.get(name)
                if isinstance(value, (list, tuple, set)):
                    return float(len(value))
                if value is not None:
                    try:
                        return float(value)
                    except (TypeError, ValueError):
                        continue
        return None

    def promotion_errors(
        self,
        model_id: str,
        *,
        production: Mapping[str, Any] | None = None,
    ) -> list[str]:
        model = self._require(model_id)
        gates = self.promotion_gates
        errors: list[str] = []

        valid_threshold = float(
            gates.get(
                "valid_plan_rate_min",
                gates.get("min_valid_plan_rate", gates.get("valid_plan_rate", 0.99)),
            )
        )
        valid_rate = self._metric(model, "valid_plan_rate")
        if valid_rate is None or valid_rate < valid_threshold:
            errors.append(
                f"valid_plan_rate {valid_rate!r} is below {valid_threshold}"
            )

        illegal_limit = float(
            gates.get(
                "illegal_skill_rate_max",
                gates.get(
                    "max_illegal_skill_rate",
                    gates.get("illegal_skill_rate", 0.005),
                ),
            )
        )
        illegal_rate = self._metric(model, "illegal_skill_rate")
        if illegal_rate is None or illegal_rate > illegal_limit:
            errors.append(
                f"illegal_skill_rate {illegal_rate!r} exceeds {illegal_limit}"
            )

        minimum_scenarios = int(
            gates.get("min_scenarios", gates.get("minimum_scenarios", 100))
        )
        scenarios = self._metric(
            model,
            "scenario_count",
            "scenarios_evaluated",
            "num_scenarios",
            "scenarios",
        )
        if scenarios is None or scenarios < minimum_scenarios:
            errors.append(
                f"scenario_count {scenarios!r} is below {minimum_scenarios}"
            )

        current_production = (
            dict(production) if production is not None else self.get_production()
        )
        if (
            bool(
                gates.get(
                    "require_better_than_production",
                    gates.get("better_than_production", True),
                )
            )
            and current_production is not None
            and current_production.get("model_id") != model_id
        ):
            metric_name = str(gates.get("benchmark_metric", "benchmark_score"))
            margin = float(gates.get("benchmark_margin", 0.0))
            candidate_score = self._metric(model, metric_name, "benchmark_score")
            production_score = self._metric(
                current_production, metric_name, "benchmark_score"
            )
            if candidate_score is None:
                errors.append(f"candidate is missing benchmark metric {metric_name!r}")
            elif production_score is None:
                errors.append(f"production is missing benchmark metric {metric_name!r}")
            elif candidate_score <= production_score + margin:
                errors.append(
                    f"{metric_name} {candidate_score} is not better than production "
                    f"{production_score} by margin {margin}"
                )
        return errors

    def promote(
        self,
        model_id: str,
        *,
        reason: str = "",
        manual_override: bool = False,
        override: bool | None = None,
    ) -> dict[str, Any]:
        if override is not None:
            manual_override = bool(override)
        candidate = self._require(model_id)
        production = self.get_production()
        if candidate["status"] == "production":
            return candidate
        if candidate["status"] in {"rejected", "archived"} and not manual_override:
            raise ValueError(
                f"cannot promote model with status {candidate['status']!r}"
            )
        errors = self.promotion_errors(model_id, production=production)
        if errors and not manual_override:
            raise ValueError("promotion gates failed: " + "; ".join(errors))

        previous_id = (
            str(production["model_id"]) if production is not None else None
        )
        with self._lock, self._connect() as connection:
            if previous_id and previous_id != model_id:
                connection.execute(
                    "UPDATE models SET status = 'archived', reason = ? WHERE model_id = ?",
                    (f"superseded by {model_id}", previous_id),
                )
            connection.execute(
                "UPDATE models SET status = 'production', reason = ? WHERE model_id = ?",
                (reason, model_id),
            )
            self._audit(
                connection,
                model_id=model_id,
                action="promote",
                previous_status=str(candidate["status"]),
                new_status="production",
                reason=reason,
                warning=bool(manual_override),
                details={
                    "manual_override": bool(manual_override),
                    "gate_errors": errors,
                    "previous_production_id": previous_id,
                },
            )
        result = self._require(model_id)
        result["warning"] = bool(manual_override)
        result["gate_errors"] = errors
        return result

    def reject(self, model_id: str, *, reason: str) -> dict[str, Any]:
        current = self._require(model_id)
        if current["status"] == "production":
            raise ValueError("rollback or promote a replacement before rejecting production")
        return self.set_status(model_id, "rejected", reason=reason)

    def rollback(
        self,
        model_id: str | None = None,
        *,
        reason: str = "manual rollback",
    ) -> dict[str, Any]:
        current = self.get_production()
        if current is None:
            raise ValueError("cannot rollback without a production model")
        target_id = str(model_id) if model_id is not None else None
        if target_id is None:
            with self._connect() as connection:
                rows = connection.execute(
                    """
                    SELECT details FROM audit_log
                    WHERE action = 'promote' AND model_id = ?
                    ORDER BY id DESC
                    """,
                    (current["model_id"],),
                ).fetchall()
            for row in rows:
                details = _loads(row["details"])
                prior = details.get("previous_production_id")
                if prior:
                    target_id = str(prior)
                    break
        if not target_id:
            raise ValueError("no previous production model is available for rollback")
        if target_id == current["model_id"]:
            return current
        target = self._require(target_id)
        with self._lock, self._connect() as connection:
            connection.execute(
                "UPDATE models SET status = 'archived', reason = ? WHERE model_id = ?",
                (f"rolled back to {target_id}", current["model_id"]),
            )
            connection.execute(
                "UPDATE models SET status = 'production', reason = ? WHERE model_id = ?",
                (reason, target_id),
            )
            self._audit(
                connection,
                model_id=target_id,
                action="rollback",
                previous_status=str(target["status"]),
                new_status="production",
                reason=reason,
                details={"replaced_production_id": current["model_id"]},
            )
        return self._require(target_id)

    def get_production(self) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM models WHERE status = 'production' LIMIT 1"
            ).fetchone()
        return self._row(row)

    def audit_log(
        self,
        *,
        model_id: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM audit_log"
        params: list[Any] = []
        if model_id is not None:
            query += " WHERE model_id = ?"
            params.append(str(model_id))
        query += " ORDER BY id DESC"
        if limit is not None:
            query += " LIMIT ?"
            params.append(max(0, int(limit)))
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["warning"] = bool(item["warning"])
            item["details"] = _loads(item.get("details"))
            result.append(item)
        return result

    list_audit_log = audit_log


__all__ = [
    "DEFAULT_PROMOTION_GATES",
    "DEFAULT_REGISTRY_PATH",
    "MODEL_STATUSES",
    "ModelRegistry",
]
