"""Outcome, validity, temporal, and label metrics for offline replay."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence


def kills_per_hour(
    episode_records: Sequence[Mapping[str, Any]],
    *,
    kill_key: str = "kills",
    duration_key: str = "duration_s",
) -> float:
    """Stub KPI: total kills / total hours across episode records.

    Looks for ``metadata[kill_key]`` or top-level ``kill_key``. Returns 0.0 when
    duration is missing or zero.
    """
    kills = 0.0
    seconds = 0.0
    for rec in episode_records:
        meta = rec.get("metadata") if isinstance(rec.get("metadata"), Mapping) else {}
        k = rec.get(kill_key, meta.get(kill_key, 0) if meta else 0)
        try:
            kills += float(k or 0)
        except (TypeError, ValueError):
            pass
        d = rec.get(duration_key, 0.0)
        try:
            seconds += float(d or 0)
        except (TypeError, ValueError):
            pass
    hours = seconds / 3600.0
    if hours <= 0:
        return 0.0
    return kills / hours


def skill_success_rates(
    episode_records: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    """Aggregate skill success rates from EpisodeRecord-like dicts.

    Prefer per-skill breakdown in ``metadata["skill_stats"]`` when present:
    ``{skill: {"attempts": n, "successes": m}}``. Otherwise fall back to
    global ``skill_successes / skill_attempts``.
    """
    per: dict[str, list[float]] = {}
    total_ok = 0
    total_try = 0
    for rec in episode_records:
        meta = rec.get("metadata") if isinstance(rec.get("metadata"), Mapping) else {}
        stats = meta.get("skill_stats") if meta else None
        if isinstance(stats, Mapping):
            for skill, row in stats.items():
                if not isinstance(row, Mapping):
                    continue
                attempts = float(row.get("attempts") or 0)
                successes = float(row.get("successes") or 0)
                if attempts <= 0:
                    continue
                per.setdefault(str(skill), []).append(successes / attempts)
        attempts = float(rec.get("skill_attempts") or 0)
        successes = float(rec.get("skill_successes") or 0)
        total_try += attempts
        total_ok += successes

    out: dict[str, float] = {}
    for skill, rates in per.items():
        out[skill] = sum(rates) / float(len(rates)) if rates else 0.0
    if "overall" not in out:
        out["overall"] = (total_ok / total_try) if total_try else 0.0
    return out


def invalid_action_counts(
    records: Sequence[Mapping[str, Any]],
    *,
    key: str = "invalid_actions",
) -> dict[str, int]:
    """Count invalid / masked-out actions from episode or replay metadata."""
    totals: dict[str, int] = {"total": 0}
    for rec in records:
        meta = rec.get("metadata") if isinstance(rec.get("metadata"), Mapping) else {}
        raw = rec.get(key, meta.get(key) if meta else None)
        if isinstance(raw, Mapping):
            for name, count in raw.items():
                try:
                    c = int(count)
                except (TypeError, ValueError):
                    continue
                totals[str(name)] = totals.get(str(name), 0) + c
                totals["total"] += c
        elif raw is not None:
            try:
                c = int(raw)
            except (TypeError, ValueError):
                continue
            totals["total"] += c
            totals["unspecified"] = totals.get("unspecified", 0) + c
    return totals


def death_rate(episode_records: Sequence[Mapping[str, Any]]) -> float:
    """Deaths per episode (mean of ``death_count``)."""
    if not episode_records:
        return 0.0
    total = 0.0
    for rec in episode_records:
        try:
            total += float(rec.get("death_count") or 0)
        except (TypeError, ValueError):
            pass
    return total / float(len(episode_records))


def mean_reward(episode_records: Sequence[Mapping[str, Any]]) -> float:
    if not episode_records:
        return 0.0
    total = 0.0
    for rec in episode_records:
        try:
            total += float(rec.get("total_reward") or 0)
        except (TypeError, ValueError):
            pass
    return total / float(len(episode_records))


def aggregate_episode_metrics(
    episode_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Bundle common offline KPIs from episode JSONL-style records."""
    return {
        "n_episodes": len(episode_records),
        "kills_per_hour": kills_per_hour(episode_records),
        "skill_success_rates": skill_success_rates(episode_records),
        "invalid_action_counts": invalid_action_counts(episode_records),
        "death_rate": death_rate(episode_records),
        "mean_reward": mean_reward(episode_records),
        "mean_duration_s": (
            sum(float(r.get("duration_s") or 0) for r in episode_records)
            / float(len(episode_records))
            if episode_records
            else 0.0
        ),
    }


def _parts(result: Any) -> tuple[Any, Mapping[str, Any], Mapping[str, Any], list[str]]:
    """Return decision, sample, observation, and the mask from either result shape."""
    if isinstance(result, Mapping):
        decision = result.get("decision") or {}
        sample = result.get("sample") or result
        observation = result.get("observation") or sample.get("observation") or {}
        allowed = result.get("allowed_skills") or sample.get("allowed_skills")
    else:
        decision = getattr(result, "decision", None)
        sample = getattr(result, "sample", None) or {}
        observation = getattr(result, "observation", None) or sample.get("observation") or {}
        allowed = getattr(result, "allowed_skills", None) or sample.get("allowed_skills")
    if not isinstance(sample, Mapping):
        sample = {}
    if not isinstance(observation, Mapping):
        observation = {}
    if not allowed:
        allowed = (
            decision.get("allowed_skills", [])
            if isinstance(decision, Mapping)
            else getattr(decision, "allowed_skills", [])
        )
    return decision, sample, observation, [str(x) for x in (allowed or [])]


def _decision_value(decision: Any, key: str, default: Any = None) -> Any:
    if isinstance(decision, Mapping):
        return decision.get(key, default)
    return getattr(decision, key, default)


def _debug_scores(decision: Any) -> Mapping[str, Any]:
    scores = _decision_value(decision, "debug_scores", {})
    return scores if isinstance(scores, Mapping) else {}


def classification_metrics(results: Sequence[Any]) -> dict[str, Any]:
    """Classification agreement, top-k (when scores exist), F1, and confusion."""
    truths: list[str] = []
    predictions: list[str] = []
    score_rows: list[tuple[str, Mapping[str, Any]]] = []
    label_space: set[str] = set()
    for result in results:
        decision, sample, _obs, allowed = _parts(result)
        truth = sample.get("skill")
        if not truth:
            continue
        truth_s = str(truth)
        pred = str(_decision_value(decision, "skill", "wait") or "wait")
        truths.append(truth_s)
        predictions.append(pred)
        label_space.update((truth_s, pred, *allowed))
        scores = _debug_scores(decision)
        usable = {
            str(key): value
            for key, value in scores.items()
            if str(key) in label_space or str(key) in allowed
        }
        if usable:
            score_rows.append((truth_s, usable))

    labels = sorted(label_space)
    confusion = {
        truth: {prediction: 0 for prediction in labels}
        for truth in labels
    }
    for truth, prediction in zip(truths, predictions):
        confusion.setdefault(truth, {}).setdefault(prediction, 0)
        confusion[truth][prediction] += 1

    per_skill: dict[str, dict[str, float | int]] = {}
    for label in labels:
        tp = sum(t == label and p == label for t, p in zip(truths, predictions))
        fp = sum(t != label and p == label for t, p in zip(truths, predictions))
        fn = sum(t == label and p != label for t, p in zip(truths, predictions))
        precision = tp / float(tp + fp) if tp + fp else 0.0
        recall = tp / float(tp + fn) if tp + fn else 0.0
        per_skill[label] = {
            "precision": precision,
            "recall": recall,
            "f1": 2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0,
            "support": sum(t == label for t in truths),
        }

    top_k: dict[str, float | int | None] = {
        "n_scored": len(score_rows),
        "top2_accuracy": None,
        "top3_accuracy": None,
    }
    if score_rows:
        for k in (2, 3):
            hits = 0
            for truth, scores in score_rows:
                ranked = sorted(
                    scores,
                    key=lambda name: float(scores.get(name) or 0.0),
                    reverse=True,
                )
                hits += truth in ranked[:k]
            top_k[f"top{k}_accuracy"] = hits / float(len(score_rows))

    count = len(truths)
    truth_counts = Counter(truths)
    return {
        "n_labeled": count,
        "accuracy": (
            sum(t == p for t, p in zip(truths, predictions)) / float(count)
            if count
            else 0.0
        ),
        "top1_accuracy": (
            sum(t == p for t, p in zip(truths, predictions)) / float(count)
            if count
            else 0.0
        ),
        **top_k,
        "labels": labels,
        "label_counts": dict(sorted(truth_counts.items())),
        "label_rates": {
            label: truth_counts[label] / float(count) if count else 0.0
            for label in labels
        },
        "per_skill": per_skill,
        "confusion": confusion,
    }


def decision_validity_metrics(
    results: Sequence[Any],
    *,
    low_confidence_threshold: float = 0.45,
) -> dict[str, Any]:
    n = len(results)
    counts: Counter[str] = Counter()
    for result in results:
        decision, _sample, _obs, allowed = _parts(result)
        skill = str(_decision_value(decision, "skill", "wait") or "wait")
        debug = _debug_scores(decision)
        reason = str(_decision_value(decision, "reason", "") or "").lower()
        raw_proposal = debug.get("proposed_skill", debug.get("raw_skill", skill))
        invalid = bool(allowed) and str(raw_proposal) not in set(allowed)
        counts["invalid_skill_proposals"] += invalid
        masked = bool(
            debug.get("mask_reject")
            or debug.get("masked")
            or "masked" in reason
            or "outside mask" in reason
        )
        counts["masked"] += masked
        used_fallback = bool(_decision_value(decision, "used_fallback", False))
        counts["fallbacks"] += used_fallback
        model_version = str(_decision_value(decision, "model_version", "") or "").lower()
        counts["scripted_fallbacks"] += bool(
            debug.get("scripted_fallback")
            or "scripted fallback" in reason
            or "primary conf=" in reason
            or "primary unavailable" in reason
            or (used_fallback and model_version.startswith("scripted"))
        )
        counts["emergency_activations"] += bool(
            debug.get("emergency") or reason.startswith("emergency:")
        )
        try:
            confidence = float(_decision_value(decision, "confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        counts["low_confidence"] += confidence < low_confidence_threshold

    def rate(key: str) -> float:
        return counts[key] / float(n) if n else 0.0

    return {
        "n_decisions": n,
        "invalid_skill_proposal_count": counts["invalid_skill_proposals"],
        "invalid_skill_proposal_rate": rate("invalid_skill_proposals"),
        "masked_count": counts["masked"],
        "masked_rate": rate("masked"),
        "fallback_count": counts["fallbacks"],
        "fallback_rate": rate("fallbacks"),
        "scripted_fallback_count": counts["scripted_fallbacks"],
        "scripted_fallback_rate": rate("scripted_fallbacks"),
        "emergency_activation_count": counts["emergency_activations"],
        "emergency_activation_rate": rate("emergency_activations"),
        "low_confidence_count": counts["low_confidence"],
        "low_confidence_rate": rate("low_confidence"),
        "low_confidence_threshold": float(low_confidence_threshold),
    }


def _numeric(mapping: Mapping[str, Any], *keys: str) -> float | None:
    for key in keys:
        if key not in mapping:
            continue
        try:
            return float(mapping[key])
        except (TypeError, ValueError):
            continue
    return None


def temporal_metrics(results: Sequence[Any]) -> dict[str, Any]:
    """Measure decision stability. Rates use available transition opportunities."""
    skills: list[str] = []
    actions: list[str] = []
    timestamps: list[float | None] = []
    explicit_durations: list[float] = []
    prevented_snapshots: list[float] = []
    premature_snapshots: list[float] = []
    provided_oscillation: list[float] = []
    runtime_statuses: list[str | None] = []
    sequence_starts: set[int] = set()
    for result in results:
        decision, sample, _obs, _allowed = _parts(result)
        skill = str(_decision_value(decision, "skill", "wait") or "wait")
        skills.append(skill)
        if bool(sample.get("_sequence_start")):
            sequence_starts.add(len(skills) - 1)
        runtime_result = (
            result.get("runtime_result") if isinstance(result, Mapping)
            else getattr(result, "runtime_result", None)
        )
        if isinstance(runtime_result, Mapping):
            action = runtime_result.get("requested_action")
            runtime_status = runtime_result.get("status")
        else:
            action = getattr(runtime_result, "requested_action", None)
            runtime_status = getattr(runtime_result, "status", None)
        runtime_statuses.append(str(runtime_status) if runtime_status is not None else None)
        actions.append(str(action or sample.get("action") or sample.get("requested_action") or skill))
        timestamps.append(_numeric(sample, "timestamp"))
        stats = sample.get("commitment_stats")
        if not isinstance(stats, Mapping):
            stats = {}
        sample_stats = sample
        debug = _debug_scores(decision)
        duration = _numeric(
            stats,
            "commitment_duration",
            "commitment_duration_s",
            "current_commitment_duration",
        )
        if duration is None:
            duration = _numeric(debug, "commitment_duration", "commitment_duration_s")
        if duration is None:
            duration = _numeric(
                sample_stats,
                "commitment_duration",
                "commitment_duration_s",
                "current_commitment_duration",
            )
        if duration is not None:
            explicit_durations.append(duration)
        for target, keys in (
            (
                prevented_snapshots,
                ("prevented_switches", "prevented_switch_count", "prevented_switch"),
            ),
            (
                premature_snapshots,
                ("premature_interrupts", "premature_interrupt_count", "premature_interrupt"),
            ),
            (provided_oscillation, ("oscillation_count", "oscillations")),
        ):
            value = _numeric(stats, *keys)
            if value is None:
                value = _numeric(debug, *keys)
            if value is None:
                value = _numeric(sample_stats, *keys)
            if value is not None:
                target.append(value)

    transition_indices = [i for i in range(1, len(skills)) if i not in sequence_starts]
    transitions = len(transition_indices)
    switches = sum(skills[i - 1] != skills[i] for i in transition_indices)
    oscillation_indices = [
        i
        for i in range(2, len(skills))
        if i not in sequence_starts and i - 1 not in sequence_starts
    ]
    detected_oscillations = sum(
        skills[i] == skills[i - 2] and skills[i] != skills[i - 1]
        for i in oscillation_indices
    )
    repeated = sum(actions[i - 1] == actions[i] for i in transition_indices)
    inferred_premature = sum(
        skills[i - 1] != skills[i]
        and runtime_statuses[i - 1] is not None
        and runtime_statuses[i - 1] not in {"success", "failed", "timeout", "cancelled"}
        for i in transition_indices
    )

    # Commitment runs are measured in elapsed time when monotonic timestamps
    # exist, otherwise in replay steps.
    run_durations: list[float] = []
    if skills:
        starts = [0]
        for i in range(1, len(skills)):
            if skills[i] != skills[i - 1] or i in sequence_starts:
                starts.append(i)
        starts.append(len(skills))
        positive_deltas = [
            float(timestamps[i]) - float(timestamps[i - 1])
            for i in transition_indices
            if timestamps[i - 1] is not None
            and timestamps[i] is not None
            and float(timestamps[i]) > float(timestamps[i - 1])
        ]
        final_delta = sorted(positive_deltas)[len(positive_deltas) // 2] if positive_deltas else 1.0
        for start, end in zip(starts, starts[1:]):
            if timestamps[start] is not None and all(
                timestamps[i] is not None for i in range(start, end)
            ):
                boundary = (
                    timestamps[end]
                    if end < len(timestamps)
                    and end not in sequence_starts
                    and timestamps[end] is not None
                    else float(timestamps[end - 1]) + final_delta
                )
                run_durations.append(max(0.0, float(boundary) - float(timestamps[start])))
            else:
                run_durations.append(float(end - start))

    durations = explicit_durations or run_durations
    prevented = int(max(prevented_snapshots, default=0.0))
    premature = int(max(premature_snapshots, default=float(inferred_premature)))
    oscillations = int(max(provided_oscillation, default=float(detected_oscillations)))
    return {
        "skill_switch_count": switches,
        "skill_switch_rate": switches / float(transitions) if transitions else 0.0,
        "prevented_switch_count": prevented,
        "prevented_switch_rate": prevented / float(max(1, prevented + switches)),
        "oscillation_count": oscillations,
        "oscillation_rate": oscillations / float(max(1, len(oscillation_indices))),
        "avg_commitment_duration": sum(durations) / float(len(durations)) if durations else 0.0,
        "commitment_duration_unit": "seconds_or_steps",
        "premature_interrupt_count": premature,
        "premature_interrupt_rate": premature / float(max(1, switches)),
        "repeated_action_count": repeated,
        "repeated_action_rate": repeated / float(transitions) if transitions else 0.0,
    }


def _event_parts(event: Any) -> tuple[str, float]:
    if isinstance(event, Mapping):
        name = str(
            event.get("type")
            or event.get("event")
            or event.get("name")
            or event.get("kind")
            or ""
        ).lower()
        try:
            count = float(event.get("count", event.get("value", 1)) or 0)
        except (TypeError, ValueError):
            count = 1.0
        return name.replace("-", "_").replace(" ", "_"), count
    return str(event or "").lower().replace("-", "_").replace(" ", "_"), 1.0


def observed_outcome_metrics(results: Sequence[Any]) -> dict[str, Any]:
    """Extract only recorded labels, events, and state transitions."""
    counts: Counter[str] = Counter()
    prior_obs: Mapping[str, Any] | None = None
    reward_total = 0.0
    objective_start: float | None = None
    objective_end: float | None = None
    session_outcomes: Counter[str] = Counter()
    seen_outcome_sessions: set[str] = set()
    event_evidence = 0
    kill_events = {
        "kill",
        "killed",
        "enemy_killed",
        "confirmed_kill",
        "kill_confirmed",
        "target_killed",
    }
    for result in results:
        _decision, sample, obs, _allowed = _parts(result)
        if bool(sample.get("_sequence_start")):
            prior_obs = None
        outcome = sample.get("_session_outcome") or sample.get("outcome")
        session_key = str(
            sample.get("session_id")
            or sample.get("_session_dir")
            or sample.get("episode_id")
            or "default"
        )
        if outcome and session_key not in seen_outcome_sessions:
            session_outcomes[str(outcome)] += 1
            seen_outcome_sessions.add(session_key)
        events = sample.get("key_events") or sample.get("events") or []
        counts["combat_steps"] += bool(obs.get("in_combat"))
        counts["target_steps"] += bool(obs.get("has_target"))
        if isinstance(events, (str, Mapping)):
            events = [events]
        step_categories: set[str] = set()
        for event in events:
            name, value = _event_parts(event)
            if not name:
                continue
            event_evidence += 1
            if name in kill_events or ("kill" in name and "skill" not in name):
                counts["confirmed_kills"] += value
                step_categories.add("kill")
            if name in {"death", "player_death", "death_confirmed"}:
                counts["deaths"] += value
                step_categories.add("death")
            if name in {"target_acquired", "target_acquisition"}:
                counts["target_acquisitions"] += value
                step_categories.add("target")
            if name in {"combat_started", "entered_combat", "combat_entry"}:
                counts["combat_entries"] += value
                step_categories.add("combat")
            if name in {"stuck_recovered", "unstuck_success"}:
                counts["stuck_recoveries"] += value
                step_categories.add("stuck")
            if name in {"death_recovered", "resurrected", "revived"}:
                counts["death_recoveries"] += value
                step_categories.add("recovery")

        if prior_obs is not None:
            if (
                "target" not in step_categories
                and not bool(prior_obs.get("has_target"))
                and bool(obs.get("has_target"))
            ):
                counts["target_acquisitions"] += 1
            if (
                "combat" not in step_categories
                and not bool(prior_obs.get("in_combat"))
                and bool(obs.get("in_combat"))
            ):
                counts["combat_entries"] += 1
            prior_dead = bool(prior_obs.get("is_dead") or prior_obs.get("is_ghost")) or str(
                prior_obs.get("life_phase") or ""
            ) in {"dead_dialog", "confirm", "rez_picker", "ghost"}
            now_dead = bool(obs.get("is_dead") or obs.get("is_ghost")) or str(
                obs.get("life_phase") or ""
            ) in {"dead_dialog", "confirm", "rez_picker", "ghost"}
            if not prior_dead and now_dead and "death" not in step_categories:
                counts["deaths"] += 1
            if prior_dead and not now_dead and "recovery" not in step_categories:
                counts["death_recoveries"] += 1
            if (
                "stuck" not in step_categories
                and bool(prior_obs.get("stuck"))
                and not bool(obs.get("stuck"))
            ):
                counts["stuck_recoveries"] += 1
        prior_obs = obs

        progress = _numeric(obs, "objective_progress", "progress")
        if progress is not None:
            if objective_start is None:
                objective_start = progress
            objective_end = progress
        reward = _numeric(sample, "reward", "reward_proxy", "total_reward")
        if reward is not None:
            reward_total += reward

    n = len(results)
    objective_delta = (
        objective_end - objective_start
        if objective_start is not None and objective_end is not None
        else 0.0
    )
    composite_reward_proxy = (
        reward_total
        + float(counts["confirmed_kills"])
        + max(0.0, objective_delta)
        - float(counts["deaths"])
    )
    return {
        "source": "demonstration_events_and_observations",
        "n_steps": n,
        "event_evidence_count": event_evidence,
        "target_acquisition_count": int(counts["target_acquisitions"]),
        "target_acquisition_rate": counts["target_acquisitions"] / float(n) if n else 0.0,
        "combat_entry_count": int(counts["combat_entries"]),
        "combat_rate": counts["combat_entries"] / float(n) if n else 0.0,
        "combat_step_count": int(counts["combat_steps"]),
        "combat_observed_rate": counts["combat_steps"] / float(n) if n else 0.0,
        "target_present_step_count": int(counts["target_steps"]),
        "confirmed_kill_count": int(counts["confirmed_kills"]),
        "confirmed_kills_source": "recorded_events_only",
        "death_count": int(counts["deaths"]),
        "death_rate": counts["deaths"] / float(n) if n else 0.0,
        "objective_progress_delta": objective_delta,
        "stuck_recovery_count": int(counts["stuck_recoveries"]),
        "death_recovery_count": int(counts["death_recoveries"]),
        "reward_proxy_total": reward_total,
        "mean_reward_proxy": reward_total / float(n) if n else 0.0,
        "reward_proxy_components": {
            "recorded_reward": reward_total,
            "confirmed_kills": int(counts["confirmed_kills"]),
            "positive_objective_progress": max(0.0, objective_delta),
            "deaths": int(counts["deaths"]),
        },
        "composite_reward_proxy": composite_reward_proxy,
        "session_outcomes": dict(session_outcomes),
    }


def model_prediction_metrics(results: Sequence[Any]) -> dict[str, Any]:
    """Average model auxiliary predictions; these are not observed outcomes."""
    values: dict[str, list[float]] = {}
    for result in results:
        decision, _sample, _obs, _allowed = _parts(result)
        for key, value in _debug_scores(decision).items():
            if not str(key).startswith("aux_"):
                continue
            try:
                values.setdefault(str(key)[4:], []).append(float(value))
            except (TypeError, ValueError):
                continue
    return {
        "source": "policy_auxiliary_heads",
        "mean_auxiliary_predictions": {
            key: sum(rows) / float(len(rows)) for key, rows in sorted(values.items())
        },
    }


def outcome_evaluation_report(
    results: Sequence[Any],
    *,
    policy_name: str = "policy",
    low_confidence_threshold: float = 0.45,
) -> dict[str, Any]:
    """Build the four explicitly separated evidence/estimate sections."""
    labels = classification_metrics(results)
    validity = decision_validity_metrics(
        results, low_confidence_threshold=low_confidence_threshold
    )
    temporal = temporal_metrics(results)
    agreement = float(labels["accuracy"])
    return {
        "policy": policy_name,
        "n_steps": len(results),
        "observed_outcomes": observed_outcome_metrics(results),
        "label_agreement": labels,
        "model_predicted": model_prediction_metrics(results),
        "counterfactual_estimates": {
            "status": "estimated_not_confirmed",
            "is_confirmed": False,
            "warning": (
                "Offline policy choices were not executed. These estimates are "
                "counterfactual and must never be presented as confirmed outcomes."
            ),
            "demo_action_match_proxy": agreement,
        },
        "decision_validity": validity,
        "temporal": temporal,
    }


def summarize_replay_results(
    results: Sequence[Any],
    *,
    policy_name: str = "policy",
) -> dict[str, Any]:
    """Summarize replay while retaining compatibility with legacy callers."""
    n = len(results)
    hist: dict[str, int] = {}
    for r in results:
        decision, _sample, _obs, _allowed = _parts(r)
        skill_s = str(_decision_value(decision, "skill", "wait") or "wait")
        hist[skill_s] = hist.get(skill_s, 0) + 1
    report = outcome_evaluation_report(results, policy_name=policy_name)
    labels = report["label_agreement"]
    validity = report["decision_validity"]
    report.update({
        "agreement_rate": labels["accuracy"],
        "n_labeled": labels["n_labeled"],
        "fallback_rate": validity["fallback_rate"],
        "skill_histogram": hist,
        "invalid_skill_vs_label": int(labels["n_labeled"])
        - round(float(labels["accuracy"]) * int(labels["n_labeled"])),
    })
    return report
