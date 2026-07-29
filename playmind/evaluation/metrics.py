"""Offline evaluation metrics from episode records and replay results.

kills/hour and related rates are stubs derived from episode metadata when
available — they do not require a live game connection.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional, Sequence


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


def summarize_replay_results(
    results: Sequence[Any],
    *,
    policy_name: str = "policy",
) -> dict[str, Any]:
    """Summarize ReplayEnv step results (agreement, skill histogram, fallbacks)."""
    n = len(results)
    if n == 0:
        return {
            "policy": policy_name,
            "n_steps": 0,
            "agreement_rate": 0.0,
            "fallback_rate": 0.0,
            "skill_histogram": {},
            "invalid_skill_vs_label": 0,
        }

    labeled = 0
    hits = 0
    fallbacks = 0
    hist: dict[str, int] = {}
    for r in results:
        decision = getattr(r, "decision", None)
        sample = getattr(r, "sample", None) or {}
        skill = getattr(decision, "skill", None) if decision is not None else None
        if skill is None and isinstance(r, Mapping):
            skill = (r.get("decision") or {}).get("skill")
            sample = r.get("sample") or {}
            decision = r.get("decision")
        skill_s = str(skill or "wait")
        hist[skill_s] = hist.get(skill_s, 0) + 1
        label = sample.get("skill") if isinstance(sample, Mapping) else None
        if label:
            labeled += 1
            if skill_s == label:
                hits += 1
        used_fb = getattr(decision, "used_fallback", False) if decision is not None else False
        if isinstance(decision, Mapping):
            used_fb = bool(decision.get("used_fallback"))
        if used_fb:
            fallbacks += 1

    return {
        "policy": policy_name,
        "n_steps": n,
        "agreement_rate": (hits / float(labeled)) if labeled else 0.0,
        "n_labeled": labeled,
        "fallback_rate": fallbacks / float(n),
        "skill_histogram": hist,
        "invalid_skill_vs_label": max(0, labeled - hits),
    }
