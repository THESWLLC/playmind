"""Small, dependency-free Ollama client for structured plans."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Callable, Mapping

from .contract import PLAN_SCHEMA_VERSION, PlannerState

SYSTEM_PROMPT = """You are PlayMind Planner V2.
Return ONLY one valid JSON object. Do not use markdown, comments, explanations,
analysis, or chain-of-thought. The exact plan schema is:
{"schema_version":1,"goal":"...","skills":[{"name":"...","until":null,"max_seconds":30,"constraints":{}}],"replan_on":["death","health_critical"],"confidence":0.8,"reason_code":"...","summary":"..."}
Use only skill names listed in available_skills. Use one to five skills.
max_seconds must be an integer from 1 through 120. confidence is from 0 to 1.
"""

UrlOpen = Callable[..., Any]


def _state_dict(state: PlannerState | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(state, PlannerState):
        return state.to_dict()
    return dict(state)


def _request_json(
    url: str,
    *,
    data: bytes | None,
    timeout: float,
    opener: UrlOpen,
) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
        method="POST" if data is not None else "GET",
    )
    with opener(request, timeout=float(timeout)) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Ollama returned a non-object response")
    return payload


class OllamaClient:
    """Injectable client wrapper; tests can provide a fake ``opener``."""

    def __init__(
        self,
        host: str = "http://127.0.0.1:11434",
        *,
        timeout: float = 60.0,
        opener: UrlOpen | None = None,
    ) -> None:
        self.host = host.rstrip("/")
        self.timeout = float(timeout)
        # Resolve at construction time so monkeypatching urllib.request.urlopen
        # works for the module-level convenience functions.
        self.opener = opener or urllib.request.urlopen

    def generate_plan(
        self,
        state: PlannerState | Mapping[str, Any],
        model: str,
        *,
        timeout: float | None = None,
    ) -> str:
        state_payload = _state_dict(state)
        prompt = (
            f"{SYSTEM_PROMPT}\n"
            f"planner_state={json.dumps(state_payload, sort_keys=True, separators=(',', ':'))}\n"
            "plan_json:"
        )
        body = json.dumps(
            {
                "model": str(model),
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {"temperature": 0.1, "num_predict": 768},
            }
        ).encode("utf-8")
        payload = _request_json(
            f"{self.host}/api/generate",
            data=body,
            timeout=self.timeout if timeout is None else float(timeout),
            opener=self.opener,
        )
        if "response" not in payload:
            raise ValueError("Ollama response is missing 'response'")
        return str(payload["response"])

    def tags(self, *, timeout: float | None = None) -> dict[str, Any]:
        return _request_json(
            f"{self.host}/api/tags",
            data=None,
            timeout=self.timeout if timeout is None else float(timeout),
            opener=self.opener,
        )

    def available(self, *, timeout: float | None = None) -> bool:
        try:
            self.tags(timeout=timeout)
            return True
        except (urllib.error.URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError):
            return False


def generate_plan(
    state: PlannerState | Mapping[str, Any],
    model: str,
    host: str = "http://127.0.0.1:11434",
    timeout: float = 60.0,
) -> str:
    """Generate raw plan text. Validation is intentionally a separate step."""
    return OllamaClient(host, timeout=timeout).generate_plan(state, model)


def tags(
    host: str = "http://127.0.0.1:11434",
    timeout: float = 2.0,
) -> dict[str, Any]:
    return OllamaClient(host, timeout=timeout).tags()


def ollama_available(
    host: str = "http://127.0.0.1:11434",
    timeout: float = 2.0,
) -> bool:
    return OllamaClient(host, timeout=timeout).available()


assert PLAN_SCHEMA_VERSION == 1  # Keep the embedded prompt schema synchronized.

__all__ = [
    "OllamaClient",
    "SYSTEM_PROMPT",
    "generate_plan",
    "ollama_available",
    "tags",
]
