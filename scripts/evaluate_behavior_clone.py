#!/usr/bin/env python3
"""Evaluate SkillPolicyV2 behavior-clone checkpoint on demonstration splits."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from playmind.training.evaluate_behavior_clone import main

if __name__ == "__main__":
    raise SystemExit(main())
