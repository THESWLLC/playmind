#!/usr/bin/env python3
"""Behavior-cloning training CLI — thin wrapper around playmind.training.train_behavior_clone."""

from __future__ import annotations

import sys

from playmind.training.train_behavior_clone import main

if __name__ == "__main__":
    raise SystemExit(main())
