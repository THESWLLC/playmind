# Offline Studio — Implementation Audit

**Date:** 2026-07-29  
**Branch:** `cursor/offline-studio-milestone-3737`  
**Baseline:** `main` @ PR #10 (170 tests)
**Current verification:** 190 tests passed; Studio startup dry-run passed with
FFmpeg/ffprobe available on the verification host.

Verified against source; not docs alone.

---

## Capability matrix

| Feature | Status | Notes |
|---------|--------|-------|
| Planner V2 / modes / registry / QLoRA / DPO | **Existing and usable** | Extend; do not rebuild |
| Owned GUI + learning_proof foundation | **Existing and newly integrated** | Now consumes canonical evaluation index and rejects smoke as proof |
| Evaluation report writer | **Existing and newly extended** | Writes timestamped files, canonical run report, normalized comparisons, and index |
| Evaluation discovery/index | **Newly implemented and unit-tested** | Writes canonical run reports, normalizes `backends` to `comparisons`, and feeds both GUIs |
| Human input capture | **Existing but incomplete** | Global pynput listeners; no focus gate by default |
| Sensor frame review script | **Existing but incomplete** | Image-only; not Studio video timeline |
| Vision/OCR on still images | **Existing and usable** | `vision.read_frame` offline-capable |
| Video import / FFmpeg pipeline | **Newly implemented and core-tested** | Probe/hash/copy/reference/extraction implemented; codec matrix not integration-tested |
| PlayMind Studio backend coordinator (no actuators) | **Newly implemented and safety-tested** | `StudioApp` plus local GUI/CLI entrypoint; no actuator imports |
| `retail_wow_offline_only` profile | **Newly implemented and unit-tested** | All live/input/process capabilities prohibited |
| Provenance / permission tracking | **Newly implemented and unit-tested** | Conservative code gate; operator statements are not legal proof |
| Timeline annotation storage/review | **Newly implemented and unit-tested** | Browser time-range review + focus-gated hotkeys; no video playback timeline |
| Transcript-assisted suggestions | **Newly implemented (mock/rule-based)** | Local SRT/VTT/TXT keyword suggestions; no ASR; always review-required |
| Real frozen benchmark builder | **Newly implemented and unit-tested** | Versioned immutable envelope; evaluator still requires explicit JSONL conversion |
| Training readiness | **Newly implemented and unit-tested** | Smoke/experimental/normal backend with Studio view/API |
| Correction-driven learning records/export | **Newly implemented and unit-tested** | JSON correction editor/review and SFT/preference rows; automatic candidate generation and SFT→DPO chaining missing |
| Registry `live_use_prohibited` / smoke restrictions | **Newly implemented and unit-tested** | Hard-blocks promotion/use; Studio provenance is not automatically propagated by trainer registration |
| Smoke artifact prominent labeling | **Newly implemented** | Both GUIs, manifest/artifact/registry, and docs identify `SMOKE / NO REAL WEIGHTS TRAINED` |
| Studio Windows launcher | **Newly implemented** | Separate setup/PowerShell/BAT launcher on port 8787 |
| Studio GUI | **Newly implemented** | Local dashboard/wizard and workflow tabs; path-form UI, no video playback timeline |
| Docs recommending Studio offline workflow | **Newly implemented** | README, `START_HERE.md`, and focused Studio guides |
| Auto Twitch/YouTube scrape | **Deferred** | Out of scope / permission risk |
| Cloud ASR transcription required | **Deferred** | Optional local/import only |
| Real 3B QLoRA on this CI host | **Mock/synthetic only** | No GPU; smoke + presets remain |

---

## Critical findings after implementation

1. **Resolved:** evaluator writes canonical run reports, `comparisons`, and
   `index.json`; both GUIs consume the normalized latest report.
2. **Resolved in registry:** smoke artifacts and live-use-prohibited artifacts
   cannot be promoted, including by manual override.
3. **Open integration gap:** normal Studio-derived SFT training does not
   automatically copy protected profile/provenance into registry
   `live_use_prohibited` fields.
4. **Open input mismatch:** benchmark builder freezes a JSON envelope while
   evaluation `--suite` consumes JSONL; documentation provides conversion.
5. **Resolved:** separate Studio and owned-game launchers are labeled and use
   ports 8787 and 8777 respectively.
6. **Open UI mismatch:** Studio Analysis offers `uniform` and `scene_change`,
   while the backend accepts `overview`, `change_aware`, `keyframes`, and
   `manual`; the two extra UI selections currently error.

---

## Reuse plan

- Extend `model_registry`, `planner_data`, `planner_training`, `vision`, `observations`, GUI patterns.
- New `playmind/studio/` with **hard import boundary** against `actuators` and SendInput.
- Separate `scripts/start_studio.py` / `start_playmind_studio.bat`.
- Keep owned-game launcher for authorized labs only; document separation.

---

## Implementation record

1. **Done:** audit, Studio safety/profile, video import, provenance, project
   store, extraction, offline analysis, annotation store, and transcripts.
2. **Done:** Studio dataset bridge, real benchmark builder, evaluation index,
   readiness backend, correction records, and registry restrictions.
3. **Done:** separate Studio GUI/wizard/dashboard, CLI and Windows launchers,
   owned-GUI evaluation discovery, smoke labeling, core tests, and Studio
   documentation/`START_HERE.md`.
4. **Deferred/not done:** playable video timeline, structured benchmark editor,
   automatic protected-lineage registry propagation, one-command
   benchmark-to-evaluator bridge, local ASR, UI extraction-name alignment, and
   real GPU validation.
