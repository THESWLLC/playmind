# Offline Studio — Implementation Audit

**Date:** 2026-07-29  
**Branch:** `cursor/offline-studio-milestone-3737`  
**Baseline:** `main` @ PR #10 (170 tests)

Verified against source; not docs alone.

---

## Capability matrix

| Feature | Status | Notes |
|---------|--------|-------|
| Planner V2 / modes / registry / QLoRA / DPO | **Existing and usable** | Extend; do not rebuild |
| Owned GUI + learning_proof foundation | **Existing but incomplete** | Eval path/schema mismatch; smoke not prominently labeled |
| Evaluation report writer | **Existing and usable** | Writes `data/playmind/planner/evaluation/planner_benchmark_*.{json,csv,md}` |
| GUI eval discovery | **Existing but incomplete** | Looks for `report.json` under wrong dirs; expects `comparisons` not `backends` |
| Human input capture | **Existing but incomplete** | Global pynput listeners; no focus gate by default |
| Sensor frame review script | **Existing but incomplete** | Image-only; not Studio video timeline |
| Vision/OCR on still images | **Existing and usable** | `vision.read_frame` offline-capable |
| Video import / FFmpeg pipeline | **Missing** | Newly implemented |
| PlayMind Studio entrypoint (no actuators) | **Missing** | Newly implemented |
| `retail_wow_offline_only` profile | **Missing** | Newly implemented |
| Provenance / permission tracking | **Missing** | Newly implemented |
| Timeline annotation UI | **Missing** | Newly implemented |
| Transcript-assisted suggestions | **Missing** | Newly implemented (optional/local) |
| Real frozen benchmark builder | **Missing** | Synthetic suite exists only |
| Training readiness gate UI/CLI | **Missing** | Newly implemented |
| Correction-driven learning loop | **Existing but incomplete** | Validator correction prompt unused |
| Registry `live_use_prohibited` | **Missing** | Newly implemented |
| Smoke artifact prominent labeling | **Existing but incomplete** | Manifest has smoke; GUI weak |
| Studio Windows launcher | **Missing** | Newly implemented |
| Docs recommending Studio offline workflow | **Existing but incomplete** | README warns vs WoW; no Studio START_HERE |
| Auto Twitch/YouTube scrape | **Deferred** | Out of scope / permission risk |
| Cloud ASR transcription required | **Deferred** | Optional local/import only |
| Real 3B QLoRA on this CI host | **Mock/synthetic only** | No GPU; smoke + presets remain |

---

## Critical bugs to fix

1. **Eval discovery mismatch** — GUI never finds planner benchmark reports.  
2. **Schema mismatch** — GUI expects `comparisons`; evaluator emits `backends`.  
3. **Smoke as ordinary candidate** — need `SMOKE / NO REAL WEIGHTS` labeling.  
4. **Startup ambiguity** — `start_playmind.bat` opens owned-game lab, not offline Studio.

---

## Reuse plan

- Extend `model_registry`, `planner_data`, `planner_training`, `vision`, `observations`, GUI patterns.
- New `playmind/studio/` with **hard import boundary** against `actuators` and SendInput.
- Separate `scripts/start_studio.py` / `start_playmind_studio.bat`.
- Keep owned-game launcher for authorized labs only; document separation.

---

## Implementation order

1. This audit  
2. Studio safety + video import + provenance + project store  
3. Offline analysis + annotation store + transcripts  
4. Dataset ingestion from Studio + real benchmark builder  
5. Eval index fix + training readiness + correction loop + registry restrictions  
6. Studio GUI + wizard + startup  
7. Tests + docs + START_HERE.md
