"""Compliance guards for the research repository."""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

REQUIRED_DOCS = [
    "docs/FEASIBILITY_STUDY.md",
    "docs/COMPLIANCE_BOUNDARIES.md",
    "docs/ARCHITECTURE.md",
    "docs/POC_PLAN.md",
    "docs/EVALUATION_PLAN.md",
]

# Filenames / symbols that strongly suggest prohibited official-client automation.
BANNED_PATH_SUBSTRINGS = [
    "sendinput",
    "keybd_event",
    "mouse_event",
    "interception_driver",
    "memory_inject",
    "packet_edit",
    "anti_cheat_bypass",
    "warden_bypass",
]


def test_required_docs_present() -> None:
    missing = [p for p in REQUIRED_DOCS if not (ROOT / p).is_file()]
    assert not missing, f"Missing required docs: {missing}"


def test_no_banned_automation_filenames() -> None:
    offenders: list[str] = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if ".git" in path.parts:
            continue
        lower = str(path).lower()
        for banned in BANNED_PATH_SUBSTRINGS:
            if banned in lower:
                offenders.append(str(path.relative_to(ROOT)))
    assert not offenders, f"Banned automation-related paths found: {offenders}"


@pytest.mark.parametrize(
    "phrase",
    [
        "automated control of a character",
        "no key presses",
    ],
)
def test_policy_phrases_anchored_in_docs(phrase: str) -> None:
    blob = "\n".join((ROOT / p).read_text(encoding="utf-8").lower() for p in REQUIRED_DOCS)
    assert phrase.lower() in blob
