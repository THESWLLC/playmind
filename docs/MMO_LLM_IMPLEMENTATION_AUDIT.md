# MMO LLM Planner — Implementation Audit

**Date:** 2026-07-29  
**Branch:** `cursor/mmo-llm-planner-milestone-3737`  
**Baseline:** `main` (139 tests passing)

Verified against source code, not prior documentation claims.

---

## Status labels

- **Implemented** — production path exists in source.
- **Automated-test covered** — repository tests exercise the stated behavior.
- **Smoke/mock tested** — only synthetic data, mocked backends, or failure
  handling is covered; this is not a trained-model or gameplay result.
- **Deferred** — requires user data, hardware, external tooling, or later work.

---

## Capability matrix

| Capability | Status | Notes |
|------------|--------|-------|
| Ollama HTTP `/api/generate` + `/api/tags` | **Existing and usable** | `planner.py`, `screen_llm.py` |
| Free-form action string planners | **Existing but incomplete** | No structured skill-plan JSON |
| Screen/VLM planner | **Existing and usable** | Periodic in owned_loop when V2 off |
| Learning V2 skills / commitment / episodes | **Existing and usable** | Preempts LLM when V2 enabled |
| Recurrent skill policy | **Existing and usable** | Sequence GRU + schema v2 |
| Demonstration recorder (obs/skill/outcome) | **Existing but incomplete** | No physical keyboard/mouse stream |
| F9 demo toggle | **Existing and usable** | `owned_gui.py` + pynput |
| Episode lifecycle / kill evidence | **Existing and usable** | `life_episode.py`, `events.py` |
| Owned GUI monitor | **Existing but incomplete** | Needs planner/learning-proof tabs |
| LoRA finetune PoC | **Existing but incomplete** | Full-precision LoRA; no QLoRA/DPO |
| Ollama Modelfile builder | **Existing and usable** | Few-shot only |
| Windows `.bat` launchers | **Existing but incomplete** | Hardcoded machine paths |
| Planner V2 contract + validator | **Newly implemented; automated-test covered** | Strict structured plans and skill allowlist |
| Planner runtime modes (observe/shadow/…) | **Newly implemented; automated-test covered** | Centralized input authorization |
| Physical human input capture | **Newly implemented; automated-test covered** | GUI focus provider remains deferred |
| Skill segmentation (rule-based) | **Newly implemented; automated-test covered** | Narrow deterministic rules; GUI manual override deferred |
| Planner SFT / preference datasets | **Newly implemented; automated-test covered** | Episode-safe split and source eligibility |
| QLoRA SFT + DPO trainers | **Newly implemented; smoke/mock tested** | Synthetic smoke only; real 4070 Ti QLoRA untested |
| Model registry + promotion gates | **Newly implemented; automated-test covered** | Manual override and rollback audit covered |
| Planner evaluation suite | **Newly implemented; mock-backend tested** | Frozen suite is synthetic; no gameplay result |
| Learning-proof GUI | **Newly implemented; API/UI test covered** | Extends `owned_gui`; benchmark report discovery path mismatch remains |
| Portable one-command Windows start | **Newly implemented; dry-run/test covered** | `start_playmind.bat` / doctor; not executed on Windows CI here |
| GGUF exporter failure/setup path | **Implemented; automated-test covered** | Full conversion deferred pending llama.cpp + merged weights |
| Live human-like play without demos | **Deferred** | Needs user’s real demonstrations |
| Anti-cheat bypass / injection | **Deferred (rejected)** | Explicitly out of scope |
| Milestone documentation | **Newly implemented** | Quickstart, architecture, data, training, evaluation, promotion, Windows/WSL2, troubleshooting |

---

## Reuse plan

- Extend `demonstrations.py` rather than replace.
- Call Learning V2 / skill commitment from Planner V2 executor (LLM plans skills, runtime executes).
- Keep `OllamaPlanner` / `ScreenLLMPlanner` as fallbacks.
- Upgrade `owned_gui.py` with tabs instead of discarding it.
- Upgrade `finetune_lora.py` patterns into `playmind/planner_training/`.
- Replace hardcoded bat paths with repo-relative `start_playmind.bat`.

---

## Safety defaults (unchanged + strengthened)

- `i_own_this_game=false`, `enable_keyboard=false`
- Default mode: `observe` or `shadow`
- Autonomous requires explicit authorization
- No stealth / injection / anti-cheat features

---

## Implementation order (this branch)

1. This audit  
2. `playmind/planner_v2/` contract + state + validator + ollama client  
3. Runtime integration + modes  
4. Physical input + segmentation  
5. Datasets + exporters  
6. QLoRA/DPO training + registry + eval  
7. GUI upgrade + Windows launcher + doctor  
8. Tests + docs + acceptance smoke
