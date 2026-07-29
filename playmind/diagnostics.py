"""Diagnostic bundle exporter for Learning Architecture V2 (Phase 19).

Writes a folder under ``data/playmind/diagnostics/<timestamp>/`` and optionally
a zip archive. Redacts user home paths from text/JSON payloads.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time
import traceback
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

DEFAULT_DIAG_ROOT = Path("data/playmind/diagnostics")
DEFAULT_OWNED_DIR = Path("data/playmind/owned")


def _homes() -> list[str]:
    homes: list[str] = []
    for key in ("HOME", "USERPROFILE"):
        v = os.environ.get(key)
        if v:
            homes.append(str(Path(v)))
    # Common Linux / Windows fallbacks
    try:
        homes.append(str(Path.home()))
    except (RuntimeError, OSError):
        pass
    # Deduplicate preserving order
    seen: set[str] = set()
    out: list[str] = []
    for h in homes:
        norm = h.rstrip("/\\")
        if norm and norm not in seen:
            seen.add(norm)
            out.append(norm)
    return out


def redact_text(text: str, *, replacement: str = "<HOME>") -> str:
    """Replace absolute user-home path prefixes in a string."""
    if not text:
        return text
    out = text
    for home in sorted(_homes(), key=len, reverse=True):
        # Forward and backslash variants
        variants = {home, home.replace("\\", "/"), home.replace("/", "\\")}
        for v in variants:
            if v and v in out:
                out = out.replace(v, replacement)
        # Case-insensitive drive-letter Windows paths
        try:
            pattern = re.compile(re.escape(home), re.IGNORECASE)
            out = pattern.sub(replacement, out)
        except re.error:
            pass
    return out


def redact_obj(obj: Any) -> Any:
    """Deep-redact strings inside JSON-like structures."""
    if isinstance(obj, str):
        return redact_text(obj)
    if isinstance(obj, Mapping):
        return {str(k): redact_obj(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [redact_obj(v) for v in obj]
    if isinstance(obj, tuple):
        return [redact_obj(v) for v in obj]
    return obj


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    safe = redact_obj(payload)
    path.write_text(json.dumps(safe, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(redact_text(text), encoding="utf-8")


def _tail_jsonl(path: Path, limit: int) -> list[Any]:
    if not path.exists() or limit <= 0:
        return []
    try:
        lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    except OSError:
        return []
    rows: list[Any] = []
    for line in lines[-limit:]:
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            rows.append({"_raw": line[:2000]})
    return rows


def _safe_read_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"_error": str(exc), "_path": str(path)}


@dataclass
class DiagnosticsBundle:
    """In-memory diagnostic payload before export."""

    recent_observations: list[Any] = field(default_factory=list)
    temporal_summary: dict[str, Any] = field(default_factory=dict)
    policy_decisions: list[Any] = field(default_factory=list)
    skill_state: dict[str, Any] = field(default_factory=dict)
    reward_breakdown: dict[str, Any] = field(default_factory=dict)
    sensor_warnings: list[str] = field(default_factory=list)
    config_snapshot: dict[str, Any] = field(default_factory=dict)
    model_metadata: dict[str, Any] = field(default_factory=dict)
    episode_summary: dict[str, Any] = field(default_factory=dict)
    exceptions: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


def collect_from_disk(
    *,
    owned_dir: Path | str = DEFAULT_OWNED_DIR,
    config_path: Path | str | None = None,
    recent_limit: int = 50,
    screenshot_names: Sequence[str] = ("latest.png", "prev.png", "latest_next.png"),
) -> tuple[DiagnosticsBundle, list[Path]]:
    """Best-effort gather diagnostics from on-disk owned-game artifacts."""
    root = Path(owned_dir)
    bundle = DiagnosticsBundle()
    screenshots: list[Path] = []

    # Recent obs: prefer experience JSONL tails; also any obs_*.json snippets.
    experience = root / "experience.jsonl"
    bundle.recent_observations = _tail_jsonl(experience, recent_limit)

    obs_dir = root / "obs"
    if obs_dir.is_dir():
        for p in sorted(obs_dir.glob("*.json"))[-recent_limit:]:
            bundle.recent_observations.append(_safe_read_json(p))

    # Temporal / decisions / skills if status sidecars exist
    for name, attr in (
        ("temporal_summary.json", "temporal_summary"),
        ("skill_state.json", "skill_state"),
        ("reward_breakdown.json", "reward_breakdown"),
        ("episode_summary.json", "episode_summary"),
        ("model_metadata.json", "model_metadata"),
    ):
        path = root / name
        data = _safe_read_json(path)
        if data is not None:
            setattr(bundle, attr, data if isinstance(data, dict) else {"value": data})

    decisions_path = root / "policy_decisions.jsonl"
    if decisions_path.exists():
        bundle.policy_decisions = _tail_jsonl(decisions_path, recent_limit)
    else:
        # Fall back to dryrun / keys logs as crude decision traces
        for alt in ("dryrun.jsonl", "keys.jsonl"):
            alt_path = root / alt
            if alt_path.exists():
                bundle.policy_decisions = _tail_jsonl(alt_path, recent_limit)
                break

    # Config snapshot
    if config_path is not None and Path(config_path).exists():
        bundle.config_snapshot = _safe_read_json(Path(config_path)) or {}
    else:
        for candidate in (
            Path("config/owned_game.json"),
            Path("config/owned_game.example.json"),
        ):
            if candidate.exists():
                bundle.config_snapshot = _safe_read_json(candidate) or {}
                break

    # Model metadata from policy / bc checkpoint markers
    policy = _safe_read_json(root / "policy.json")
    if isinstance(policy, dict):
        bundle.model_metadata.setdefault(
            "policy",
            {
                "legacy": bool(policy.get("legacy")),
                "schema_version": policy.get("schema_version"),
                "note": policy.get("note"),
                "n_keys": len(policy.get("q") or policy.get("values") or {}),
            },
        )
    legacy = root / "policy.legacy.json"
    if legacy.exists():
        bundle.model_metadata["policy_legacy_present"] = True

    # Episodes
    ep_dir = root / "episodes"
    if ep_dir.is_dir():
        summaries = sorted(ep_dir.glob("*.json"))[-10:]
        bundle.episode_summary = {
            "count_files": len(list(ep_dir.glob("*.json"))),
            "recent": [_safe_read_json(p) for p in summaries],
        }

    # Screenshots
    for name in screenshot_names:
        p = root / name
        if p.exists() and p.is_file():
            screenshots.append(p)

    # Sensor warnings from process / ui memory
    for mem_name in ("process_memory.json", "ui_memory.json"):
        mem = _safe_read_json(root / mem_name)
        if isinstance(mem, dict) and mem.get("sensor_warnings"):
            bundle.sensor_warnings.extend(str(x) for x in mem["sensor_warnings"])

    # Exceptions log if present
    exc_path = root / "exceptions.txt"
    if exc_path.exists():
        try:
            text = exc_path.read_text(encoding="utf-8", errors="replace")
            if text.strip():
                bundle.exceptions.append(text)
        except OSError as exc:
            bundle.exceptions.append(f"failed to read exceptions.txt: {exc}")

    return bundle, screenshots


def export_diagnostics(
    *,
    out_root: Path | str = DEFAULT_DIAG_ROOT,
    owned_dir: Path | str = DEFAULT_OWNED_DIR,
    config_path: Path | str | None = None,
    bundle: DiagnosticsBundle | None = None,
    screenshots: Sequence[Path] | None = None,
    make_zip: bool = True,
    timestamp: str | None = None,
    extra_exceptions: Iterable[BaseException] | None = None,
) -> Path:
    """Write diagnostic folder (and optional zip). Returns folder path."""
    stamp = timestamp or time.strftime("%Y%m%d-%H%M%S")
    dest = Path(out_root) / stamp
    dest.mkdir(parents=True, exist_ok=True)

    if bundle is None or screenshots is None:
        disk_bundle, disk_shots = collect_from_disk(
            owned_dir=owned_dir, config_path=config_path
        )
        if bundle is None:
            bundle = disk_bundle
        if screenshots is None:
            screenshots = disk_shots

    if extra_exceptions:
        for exc in extra_exceptions:
            bundle.exceptions.append(
                "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            )

    _write_json(dest / "recent_observations.json", bundle.recent_observations)
    _write_json(dest / "temporal_summary.json", bundle.temporal_summary)
    _write_json(dest / "policy_decisions.json", bundle.policy_decisions)
    _write_json(dest / "skill_state.json", bundle.skill_state)
    _write_json(dest / "reward_breakdown.json", bundle.reward_breakdown)
    _write_json(dest / "sensor_warnings.json", bundle.sensor_warnings)
    _write_json(dest / "config_snapshot.json", bundle.config_snapshot)
    _write_json(dest / "model_metadata.json", bundle.model_metadata)
    _write_json(dest / "episode_summary.json", bundle.episode_summary)
    _write_json(dest / "extra.json", bundle.extra)

    exc_text = "\n\n".join(bundle.exceptions) if bundle.exceptions else ""
    _write_text(dest / "exceptions.txt", exc_text or "(none)\n")

    shots_dir = dest / "screenshots"
    shots_dir.mkdir(exist_ok=True)
    for src in screenshots or []:
        src_p = Path(src)
        if src_p.exists() and src_p.is_file():
            try:
                shutil.copy2(src_p, shots_dir / src_p.name)
            except OSError as exc:
                bundle.exceptions.append(f"screenshot_copy_failed:{src_p}: {exc}")
                _write_text(dest / "exceptions.txt", "\n\n".join(bundle.exceptions) + "\n")

    manifest = {
        "created_at": time.time(),
        "timestamp": stamp,
        "owned_dir": redact_text(str(owned_dir)),
        "files": sorted(p.name for p in dest.iterdir() if p.is_file()),
        "screenshots": sorted(p.name for p in shots_dir.iterdir()) if shots_dir.exists() else [],
    }
    _write_json(dest / "manifest.json", manifest)

    if make_zip:
        zip_path = dest.with_suffix(".zip")
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for path in dest.rglob("*"):
                if path.is_file():
                    zf.write(path, arcname=str(path.relative_to(dest.parent)))
        manifest["zip"] = redact_text(str(zip_path))
        _write_json(dest / "manifest.json", manifest)

    return dest
