"""Convert a merged Hugging Face planner model to a verified GGUF artifact."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

SETUP_INSTRUCTIONS = """llama.cpp GGUF converter was not found.
Set --llama-cpp-dir or LLAMA_CPP_DIR to a llama.cpp checkout containing
convert_hf_to_gguf.py. For example:
  git clone https://github.com/ggml-org/llama.cpp.git third_party/llama.cpp
  python3 -m pip install -r third_party/llama.cpp/requirements.txt
  python3 scripts/export_planner_gguf.py --llama-cpp-dir third_party/llama.cpp ...
An adapter alone cannot be converted; merge it with its licensed base model first.
"""


class GGUFConverterNotFound(RuntimeError):
    pass


def find_converter(llama_cpp_dir: str | Path | None = None) -> Path | None:
    candidates: list[Path] = []
    if llama_cpp_dir:
        candidates.append(Path(llama_cpp_dir))
    if os.environ.get("LLAMA_CPP_DIR"):
        candidates.append(Path(os.environ["LLAMA_CPP_DIR"]))
    candidates.extend(
        [
            Path("third_party/llama.cpp"),
            Path("vendor/llama.cpp"),
            Path("../llama.cpp"),
        ]
    )
    for root in candidates:
        for name in ("convert_hf_to_gguf.py", "convert.py"):
            script = root.expanduser() / name
            if script.is_file():
                return script.resolve()
    return None


def _find_quantizer(converter: Path) -> Path | None:
    root = converter.parent
    for path in (
        root / "build" / "bin" / "llama-quantize",
        root / "build" / "bin" / "quantize",
        root / "quantize",
    ):
        if path.is_file() and os.access(path, os.X_OK):
            return path
    system = shutil.which("llama-quantize") or shutil.which("quantize")
    return Path(system) if system else None


def export_gguf(
    *,
    model_path: str | Path,
    output_path: str | Path,
    llama_cpp_dir: str | Path | None = None,
    converter_path: str | Path | None = None,
    outtype: str = "f16",
    quantization: str | None = None,
    python_binary: str = sys.executable,
) -> dict[str, Any]:
    model = Path(model_path).expanduser()
    if not model.is_dir():
        raise FileNotFoundError(f"merged Hugging Face model directory not found: {model}")
    converter = (
        Path(converter_path).expanduser()
        if converter_path is not None
        else find_converter(llama_cpp_dir)
    )
    if converter is None or not converter.is_file():
        raise GGUFConverterNotFound(SETUP_INSTRUCTIONS)
    output = Path(output_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    intermediate = (
        output
        if not quantization
        else output.with_name(f"{output.stem}.f16{output.suffix or '.gguf'}")
    )
    convert_command = [
        python_binary,
        str(converter),
        str(model),
        "--outfile",
        str(intermediate),
        "--outtype",
        str(outtype),
    ]
    subprocess.run(convert_command, check=True)
    if not intermediate.is_file() or intermediate.stat().st_size == 0:
        raise RuntimeError(
            f"converter exited successfully but produced no GGUF artifact: {intermediate}"
        )
    commands = [convert_command]
    if quantization:
        quantizer = _find_quantizer(converter)
        if quantizer is None:
            raise GGUFConverterNotFound(
                "GGUF conversion succeeded, but llama-quantize was not found. "
                "Build llama.cpp with: cmake -B build && cmake --build build "
                "--config Release. The unquantized artifact is at "
                f"{intermediate}."
            )
        quantize_command = [
            str(quantizer),
            str(intermediate),
            str(output),
            str(quantization),
        ]
        subprocess.run(quantize_command, check=True)
        commands.append(quantize_command)
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"GGUF artifact missing after conversion: {output}")
    result = {
        "status": "succeeded",
        "artifact": str(output),
        "size_bytes": output.stat().st_size,
        "converter": str(converter),
        "commands": commands,
        "quantization": quantization or outtype,
    }
    print(f"GGUF export succeeded: {output}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, help="merged HF model directory")
    parser.add_argument("--output", required=True)
    parser.add_argument("--llama-cpp-dir")
    parser.add_argument("--converter")
    parser.add_argument("--outtype", default="f16")
    parser.add_argument("--quantization")
    parser.add_argument("--python-binary", default=sys.executable)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = export_gguf(
            model_path=args.model_path,
            output_path=args.output,
            llama_cpp_dir=args.llama_cpp_dir,
            converter_path=args.converter,
            outtype=args.outtype,
            quantization=args.quantization,
            python_binary=args.python_binary,
        )
    except (
        GGUFConverterNotFound,
        FileNotFoundError,
        RuntimeError,
        subprocess.CalledProcessError,
    ) as exc:
        print(f"GGUF export failed:\n{exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "GGUFConverterNotFound",
    "SETUP_INSTRUCTIONS",
    "export_gguf",
    "find_converter",
    "main",
]
