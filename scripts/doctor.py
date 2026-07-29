#!/usr/bin/env python3
"""Structured local environment checks for PlayMind."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _module(name: str) -> dict[str, Any]:
    available = importlib.util.find_spec(name) is not None
    version = None
    if available:
        try:
            module = __import__(name)
            version = getattr(module, "__version__", None)
        except Exception:  # noqa: BLE001
            pass
    return {"available": available, "version": version}


def _ram() -> dict[str, Any]:
    total = None
    available = None
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        values: dict[str, int] = {}
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            try:
                values[key] = int(value.strip().split()[0]) * 1024
            except (ValueError, IndexError):
                pass
        total, available = values.get("MemTotal"), values.get("MemAvailable")
    elif os.name == "nt":
        try:
            import ctypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("length", ctypes.c_ulong),
                    ("load", ctypes.c_ulong),
                    ("total", ctypes.c_ulonglong),
                    ("available", ctypes.c_ulonglong),
                    ("total_page", ctypes.c_ulonglong),
                    ("available_page", ctypes.c_ulonglong),
                    ("total_virtual", ctypes.c_ulonglong),
                    ("available_virtual", ctypes.c_ulonglong),
                    ("available_extended", ctypes.c_ulonglong),
                ]

            status = MemoryStatus()
            status.length = ctypes.sizeof(status)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            total, available = int(status.total), int(status.available)
        except Exception:  # noqa: BLE001
            pass
    return {
        "total_bytes": total,
        "available_bytes": available,
        "total_gb": round(total / 1024**3, 2) if total else None,
        "available_gb": round(available / 1024**3, 2) if available else None,
    }


def _ollama(timeout: float) -> dict[str, Any]:
    try:
        with urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return {
            "reachable": True,
            "tags": [
                str(model.get("name") or model.get("model"))
                for model in payload.get("models", [])
                if isinstance(model, dict)
            ],
        }
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        return {"reachable": False, "tags": [], "error": str(exc)}


def _config(root: Path) -> dict[str, Any]:
    path = root / "config/owned_game.json"
    if not path.exists():
        path = root / "config/owned_game.example.json"
    if not path.exists():
        return {"valid": False, "path": None, "error": "No owned game config found."}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"valid": False, "path": str(path), "error": str(exc)}
    errors: list[str] = []
    if not isinstance(payload, dict):
        errors.append("Root must be a JSON object.")
    if isinstance(payload, dict) and payload.get("mode", "shadow") not in {
        "observe",
        "shadow",
        "assist",
        "hybrid",
        "autonomous",
        "replay",
    }:
        errors.append("mode is invalid")
    return {
        "valid": not errors,
        "path": str(path),
        "errors": errors,
        "safe_defaults": {
            "mode": payload.get("mode", "shadow") if isinstance(payload, dict) else None,
            "keyboard_enabled": bool(payload.get("enable_keyboard", False))
            if isinstance(payload, dict)
            else False,
        },
    }


def _torch() -> dict[str, Any]:
    result = _module("torch")
    result.update({"cuda_available": False, "cuda_version": None, "gpu_name": None, "vram_bytes": None})
    if not result["available"]:
        return result
    try:
        import torch

        result["cuda_available"] = bool(torch.cuda.is_available())
        result["cuda_version"] = getattr(torch.version, "cuda", None)
        if result["cuda_available"]:
            result["gpu_name"] = torch.cuda.get_device_name(0)
            result["vram_bytes"] = int(torch.cuda.get_device_properties(0).total_memory)
    except Exception as exc:  # noqa: BLE001
        result["error"] = str(exc)
    return result


def doctor_report(root: str | Path | None = None, *, timeout: float = 0.35) -> dict[str, Any]:
    repo = Path(root).resolve() if root is not None else ROOT
    usage = shutil.disk_usage(repo)
    nvidia_smi = shutil.which("nvidia-smi")
    cuda = {"nvidia_smi": nvidia_smi, "driver_visible": False}
    if nvidia_smi:
        try:
            probe = subprocess.run(  # noqa: S603
                [nvidia_smi, "--query-gpu=name,memory.total", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
            cuda.update({"driver_visible": probe.returncode == 0, "gpus": probe.stdout.strip().splitlines()})
        except (OSError, subprocess.TimeoutExpired) as exc:
            cuda["error"] = str(exc)
    optional = {name: _module(name) for name in ("mss", "PIL", "pynput", "pytesseract")}
    training = {
        name: _module(name)
        for name in ("torch", "transformers", "datasets", "peft", "accelerate")
    }
    paths = [repo / "data", repo / "models", repo / "config"]
    permissions = {
        str(path): {
            "exists": path.exists(),
            "readable": os.access(path if path.exists() else path.parent, os.R_OK),
            "writable": os.access(path if path.exists() else path.parent, os.W_OK),
        }
        for path in paths
    }
    report = {
        "ok": True,
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
            "supported": sys.version_info >= (3, 10),
        },
        "platform": {"system": platform.system(), "release": platform.release(), "machine": platform.machine()},
        "cuda": cuda,
        "torch": _torch(),
        "ram": _ram(),
        "disk": {
            "path": str(repo),
            "total_bytes": usage.total,
            "free_bytes": usage.free,
            "free_gb": round(usage.free / 1024**3, 2),
        },
        "ollama": _ollama(timeout),
        "training_dependencies": training,
        "llama_cpp": {
            "python_package": _module("llama_cpp"),
            "executable": shutil.which("llama-cli") or shutil.which("llama"),
        },
        "capture_dependencies": optional,
        "config": _config(repo),
        "permissions": permissions,
    }
    report["ok"] = bool(report["python"]["supported"] and report["config"]["valid"])
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Emit compact JSON")
    args = parser.parse_args(argv)
    report = doctor_report()
    print(json.dumps(report, indent=None if args.json else 2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
