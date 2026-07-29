"""Legacy persistence migration for Learning Architecture V2 (Phase 17).

Does not destroy existing user data. Marks Q-tables as legacy, stamps
``schema_version`` on process/experience sidecars, and backs up corrupt JSON.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
DEFAULT_OWNED_DIR = Path("data/playmind/owned")

# Files we attempt to migrate / stamp.
MEMORY_JSON_FILES = (
    "process_memory.json",
    "ui_memory.json",
    "ability_memory.json",
    "travel_memory.json",
)
EXPERIENCE_JSONL = "experience.jsonl"
EXPERIENCE_SIDECAR = "experience.meta.json"
POLICY_JSON = "policy.json"
POLICY_LEGACY_JSON = "policy.legacy.json"


def atomic_write_json(path: Path | str, payload: Mapping[str, Any], *, indent: int = 2) -> Path:
    """Write JSON atomically via temp file + ``os.replace`` (Windows-safe)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + f".{os.getpid()}.{int(time.time() * 1000)}.tmp")
    try:
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(dict(payload), f, indent=indent, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
    return target


def backup_corrupt(path: Path, *, reason: str = "corrupt") -> Path | None:
    """Copy ``path`` to ``path.bak`` (timestamped if ``.bak`` exists). Returns backup path."""
    if not path.exists():
        return None
    bak = path.with_suffix(path.suffix + ".bak")
    if bak.exists():
        stamp = time.strftime("%Y%m%d-%H%M%S")
        bak = path.with_suffix(f"{path.suffix}.bak.{stamp}")
    shutil.copy2(path, bak)
    logger.warning("Backed up %s → %s (%s)", path, bak, reason)
    return bak


def _load_json_or_backup(path: Path) -> dict[str, Any] | None:
    """Load JSON object; on failure back up and return None."""
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        backup_corrupt(path, reason=f"json_error:{exc}")
        return None
    if not isinstance(raw, dict):
        backup_corrupt(path, reason="not_object")
        return None
    return raw


@dataclass
class MigrationReport:
    """Summary of a migration run."""

    data_dir: str
    actions: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    policy_legacy_path: str | None = None
    schema_stamped: list[str] = field(default_factory=list)
    backups: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "data_dir": self.data_dir,
            "actions": list(self.actions),
            "warnings": list(self.warnings),
            "policy_legacy_path": self.policy_legacy_path,
            "schema_stamped": list(self.schema_stamped),
            "backups": list(self.backups),
            "schema_version": SCHEMA_VERSION,
        }


def mark_policy_legacy(data_dir: Path | str, *, overwrite: bool = False) -> Path | None:
    """Copy ``policy.json`` → ``policy.legacy.json`` and stamp legacy metadata.

    Leaves the original ``policy.json`` in place so legacy_q mode still works.
    Returns the legacy path, or None if no policy existed.
    """
    root = Path(data_dir)
    src = root / POLICY_JSON
    dst = root / POLICY_LEGACY_JSON
    if not src.exists():
        return None

    raw = _load_json_or_backup(src)
    if raw is None:
        # Still copy bytes if unreadable as object (e.g. truncated).
        if not dst.exists() or overwrite:
            shutil.copy2(src, dst)
        return dst

    payload = dict(raw)
    payload["legacy"] = True
    payload["schema_version"] = int(payload.get("schema_version") or SCHEMA_VERSION)
    payload["migrated_at"] = time.time()
    payload["note"] = (
        payload.get("note")
        or "Legacy tabular Q-table. Disabled by default in hybrid/scripted modes."
    )

    if dst.exists() and not overwrite:
        # Keep existing legacy copy; still stamp source if missing markers.
        if not raw.get("legacy"):
            stamped = dict(raw)
            stamped["legacy"] = True
            stamped.setdefault("schema_version", SCHEMA_VERSION)
            atomic_write_json(src, stamped)
        return dst

    atomic_write_json(dst, payload)
    # Mark the live policy as legacy-compatible without deleting Q values.
    live = dict(raw)
    live["legacy"] = True
    live.setdefault("schema_version", SCHEMA_VERSION)
    atomic_write_json(src, live)
    return dst


def ensure_schema_version(
    path: Path,
    *,
    extra: Mapping[str, Any] | None = None,
) -> bool:
    """Add ``schema_version`` to a JSON object file if missing. Returns True if wrote."""
    raw = _load_json_or_backup(path)
    if raw is None:
        if not path.exists():
            return False
        # Corrupt — already backed up; do not rewrite.
        return False
    changed = False
    if "schema_version" not in raw:
        raw["schema_version"] = SCHEMA_VERSION
        changed = True
    if extra:
        for k, v in extra.items():
            if k not in raw:
                raw[k] = v
                changed = True
    if changed:
        atomic_write_json(path, raw)
    return changed


def ensure_experience_sidecar(data_dir: Path | str) -> Path:
    """Create/update ``experience.meta.json`` beside the experience JSONL log."""
    root = Path(data_dir)
    sidecar = root / EXPERIENCE_SIDECAR
    jsonl = root / EXPERIENCE_JSONL
    existing = _load_json_or_backup(sidecar) if sidecar.exists() else None
    payload: dict[str, Any] = dict(existing or {})
    payload.setdefault("schema_version", SCHEMA_VERSION)
    payload["experience_path"] = EXPERIENCE_JSONL
    payload["legacy"] = True
    payload["format"] = "jsonl"
    payload["updated_at"] = time.time()
    if jsonl.exists():
        try:
            payload["bytes"] = int(jsonl.stat().st_size)
        except OSError:
            pass
    atomic_write_json(sidecar, payload)
    return sidecar


def migrate_owned_data(
    data_dir: Path | str | None = None,
    *,
    overwrite_legacy_policy: bool = False,
) -> MigrationReport:
    """Run full Phase-17 migration against an owned-game data directory."""
    root = Path(data_dir) if data_dir is not None else DEFAULT_OWNED_DIR
    root.mkdir(parents=True, exist_ok=True)
    report = MigrationReport(data_dir=str(root))

    legacy = mark_policy_legacy(root, overwrite=overwrite_legacy_policy)
    if legacy is not None:
        report.policy_legacy_path = str(legacy)
        report.actions.append(f"marked_policy_legacy:{legacy.name}")
    else:
        report.actions.append("no_policy_json")

    for name in MEMORY_JSON_FILES:
        path = root / name
        if not path.exists():
            continue
        before_bak = list(root.glob(f"{name}.bak*"))
        wrote = ensure_schema_version(path)
        after_bak = list(root.glob(f"{name}.bak*"))
        for b in after_bak:
            if b not in before_bak:
                report.backups.append(str(b))
                report.warnings.append(f"corrupt_backed_up:{b.name}")
        if wrote:
            report.schema_stamped.append(name)
            report.actions.append(f"schema_version:{name}")
        elif path.exists() and name not in report.schema_stamped:
            # Already stamped or unreadable
            raw = None
            try:
                raw = json.loads(path.read_text(encoding="utf-8-sig"))
            except (OSError, json.JSONDecodeError):
                pass
            if isinstance(raw, dict) and "schema_version" in raw:
                report.actions.append(f"schema_already:{name}")

    # Travel memory may only live inside process_memory; stamp process always if present.
    proc = root / "process_memory.json"
    if proc.exists():
        ensure_schema_version(proc, extra={"kind": "process_memory"})

    sidecar = ensure_experience_sidecar(root)
    report.actions.append(f"experience_sidecar:{sidecar.name}")
    report.schema_stamped.append(sidecar.name)

    # Stamp ability/ui if they gained schema via ensure above — already tracked.
    report.actions.append("migration_complete")
    return report
