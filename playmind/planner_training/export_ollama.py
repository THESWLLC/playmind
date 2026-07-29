"""Create an Ollama Modelfile for a merged model or LoRA adapter."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

from playmind.planner_v2.ollama_client import SYSTEM_PROMPT


def build_modelfile(
    *,
    merged_path: str | Path | None = None,
    adapter_path: str | Path | None = None,
    base_model: str | Path | None = None,
    system_prompt: str = SYSTEM_PROMPT,
    temperature: float = 0.1,
) -> str:
    if bool(merged_path) == bool(adapter_path):
        raise ValueError("provide exactly one of merged_path or adapter_path")
    if adapter_path and not base_model:
        raise ValueError("adapter export requires base_model for the Ollama FROM line")
    source = Path(merged_path or base_model or "").expanduser().resolve()
    lines = [
        f"FROM {source}",
        f'PARAMETER temperature {float(temperature):g}',
        'PARAMETER stop "```"',
        'SYSTEM """',
        system_prompt.strip(),
        '"""',
    ]
    if adapter_path:
        lines.insert(1, f"ADAPTER {Path(adapter_path).expanduser().resolve()}")
    return "\n".join(lines) + "\n"


def export_ollama(
    *,
    model_name: str,
    output_path: str | Path,
    merged_path: str | Path | None = None,
    adapter_path: str | Path | None = None,
    base_model: str | Path | None = None,
    run_create: bool = False,
    ollama_binary: str = "ollama",
) -> dict[str, Any]:
    if not str(model_name).strip():
        raise ValueError("model_name must be non-empty")
    modelfile = build_modelfile(
        merged_path=merged_path,
        adapter_path=adapter_path,
        base_model=base_model,
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(modelfile, encoding="utf-8")
    command = [ollama_binary, "create", str(model_name), "-f", str(path)]
    printable = shlex.join(command)
    print(f"Modelfile written: {path}")
    print(f"Run: {printable}")
    result: dict[str, Any] = {
        "status": "modelfile_created",
        "modelfile": str(path),
        "model_name": str(model_name),
        "create_command": printable,
        "create_ran": False,
        "create_succeeded": False,
    }
    if run_create:
        completed = subprocess.run(command, check=False, text=True, capture_output=True)
        result["create_ran"] = True
        result["returncode"] = completed.returncode
        result["stdout"] = completed.stdout
        result["stderr"] = completed.stderr
        result["create_succeeded"] = completed.returncode == 0
        result["status"] = (
            "ollama_create_succeeded"
            if completed.returncode == 0
            else "ollama_create_failed"
        )
        if completed.returncode != 0:
            raise subprocess.CalledProcessError(
                completed.returncode,
                command,
                output=completed.stdout,
                stderr=completed.stderr,
            )
        print(f"ollama create succeeded for {model_name}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--merged-path")
    source.add_argument("--adapter-path")
    parser.add_argument("--base-model")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output", default="models/playmind/Modelfile")
    parser.add_argument("--create", action="store_true")
    parser.add_argument("--ollama-binary", default="ollama")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = export_ollama(
            model_name=args.model_name,
            output_path=args.output,
            merged_path=args.merged_path,
            adapter_path=args.adapter_path,
            base_model=args.base_model,
            run_create=args.create,
            ollama_binary=args.ollama_binary,
        )
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        print(f"Ollama export failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_modelfile", "export_ollama", "main"]
