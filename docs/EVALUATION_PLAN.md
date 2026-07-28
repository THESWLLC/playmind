# Evaluation Plan — Metrics, Suites, and Success Criteria

**Applies to:** offline coaching PoC, simulator agents, and future human-in-the-loop studies.  
**Does not authorize:** live automated play on official servers.

---

## 1. Evaluation principles

1. **Reproducibility** — fixed seeds, dataset version hashes, config snapshots.  
2. **Task isolation** — score perception separately from decision-making.  
3. **Ambiguity awareness** — multiple actions can be correct; support top-k and expert soft labels.  
4. **Compliance gates** — automated tests fail if prohibited actuation modules appear.  
5. **No live-server KPIs** as project goals (e.g. “unattended gold/hour”).

---

## 2. Success criteria (measurable)

| Metric | Definition | PoC target | Stretch |
|--------|------------|------------|---------|
| **UI detection accuracy** | Correct detection/localization of key UI elements (IoU≥0.5 or ROI presence) | ≥90% default UI | ≥85% across 3 UI layouts |
| **HP/resource read error** | Mean abs error of percent bars | ≤5 pp | ≤3 pp |
| **Ability recommendation accuracy** | Top-1 match to expert label on unambiguous frames | ≥70% | ≥85% |
| **Top-3 recommendation recall** | Expert label in top 3 | ≥90% | ≥95% |
| **Average reaction time** | Pipeline latency from frame to recommendation | ≤100 ms GPU offline | ≤50 ms |
| **Combat damage vs scripted baseline** | Sim-only DPS/HPS ratio vs priority script | ≥0.85 | ≥0.95 |
| **Survival rate** | Sim encounters survived / attempts | ≥0.80 on tier-1 bosses | ≥0.90 |
| **Navigation success rate** | Sim waypoint courses completed | n/a in PoC | ≥0.80 |
| **Stuck frequency** | Stuck events / minute in sim nav | n/a in PoC | ≤0.1 |
| **False-action rate** | Illegal/impossible recommendations / windows | ≤5% | ≤1% |
| **Recovery success** | Sim recoveries from death/stuck | n/a in PoC | ≥0.75 |
| **Generalization** | Metric drop from 1080p default UI → alternate resolution/UI pack | ≤10 pp drop | ≤5 pp |

---

## 3. Perception evaluation protocol

### Datasets

- `eval_ui_default` — held-out frames, default UI  
- `eval_ui_alt` — different resolution / UI scale  
- `eval_occlusion` — busy combat, particles  

### Metrics detail

- **Detection:** precision, recall, F1, mAP@0.5 for enemies/UI widgets  
- **Bars:** MAE, % within tolerance  
- **OCR:** character error rate on cast names / quest titles  
- **Temporal stability:** flicker rate (label changes without real state change)

### Pass/fail gates (Phase 2)

- Player HP MAE ≤5 pp on `eval_ui_default`  
- Core action-button ready/not-ready F1 ≥0.85  

---

## 4. Decision evaluation protocol

### Unambiguous vs ambiguous windows

- **Unambiguous:** single expert label (e.g. interrupt available on kickable cast)  
- **Ambiguous:** set-valued labels; score with IoU-over-set / top-k  

### Alignment to combat logs

1. Build player cast timeline from log.  
2. For each recommendation time \(t\), find next cast in \((t, t+W]\).  
3. Score match; also score whether recommendation was **legal** given estimated state.  

### Baselines

| Baseline | Purpose |
|----------|---------|
| Random legal ability | Floor |
| “Always spam builder” | Naive |
| Static priority list with **ground-truth state** | Oracle-state ceiling |
| Priority list with **vision state** | PoC system |
| Human player actions | Behavioral reference (not always optimal) |

---

## 5. Simulator evaluation protocol (Phases 4–6)

| Suite | Measures |
|-------|----------|
| `combat_dummy` | Rotation DPS vs script |
| `pack_ai` | Target swap, AoE legality |
| `boss_script_v1` | Mechanic reaction timers, survival |
| `nav_waypoints` | Success, time, stuck count |
| `recovery` | Corpse run / unstick policies |
| `party_follow` | Distance error, separation events |

Report mean ± std over ≥30 seeds.

---

## 6. Human-in-the-loop evaluation (Phase 7)

- Users see recommendations; **they** retain full control.  
- Collect: accept/reject rates, self-reported cognitive load, false-action catches.  
- **Forbidden metric:** unattended hours played.

---

## 7. Proposed automated test suite

### 7.1 Compliance tests (`tests/compliance/`)

| Test | Asserts |
|------|---------|
| `test_no_input_drivers.py` | Repo has no modules named/implementing key send, SendInput, interception, etc. for game control |
| `test_no_memory_hooks.py` | Banlist imports (`pymem`, frida attach patterns, etc.) |
| `test_docs_present.py` | Required docs exist |

### 7.2 Unit tests (`tests/unit/`)

| Test | Asserts |
|------|---------|
| `test_priority_list_basic` | Known `GameState` → expected action |
| `test_defensive_threshold` | Low HP triggers defensive recommendation |
| `test_illegal_action_filter` | On CD abilities not recommended |
| `test_bar_reader_synthetic` | Synthetic bar images → pct within tolerance |

### 7.3 Integration tests (`tests/integration/`)

| Test | Asserts |
|------|---------|
| `test_video_clip_smoke` | Short fixture clip produces ≥N recommendations |
| `test_log_alignment_smoke` | Fixture log aligns within skew budget |
| `test_report_generation` | Metrics JSON/Markdown emitted |

### 7.4 Regression golden files (`tests/golden/`)

- Freeze recommendation sequences on fixture clips; fail on unexpected drift > threshold.

### 7.5 Simulator tests (when present)

| Test | Asserts |
|------|---------|
| `test_env_step_determinism` | Same seed → same trajectory |
| `test_scripted_agent_completes_dummy` | Baseline clears scenario |

### 7.6 Suggested CI command

```bash
pytest tests/compliance tests/unit -q
pytest tests/integration -q --skip-heavy  # optional GPU job separately
```

---

## 8. Reporting format

Each eval run writes `evaluation/reports/<run_id>/`:

```text
metrics.json       # machine-readable
summary.md         # human-readable
confusion.png      # optional
config_snapshot/   # yaml + git sha
dataset_manifest.json
```

### `metrics.json` (example keys)

```json
{
  "run_id": "2026-07-28T18-00-00Z",
  "git_sha": "…",
  "ui_detection_f1": 0.0,
  "hp_mae_pp": 0.0,
  "recommend_top1": 0.0,
  "recommend_top3": 0.0,
  "false_action_rate": 0.0,
  "latency_ms_p50": 0.0,
  "latency_ms_p95": 0.0
}
```

---

## 9. Failure analysis playbook

| Symptom | Likely cause | Action |
|---------|--------------|--------|
| High HP MAE | UI scale / colorblind pack | Add layout profiles |
| Good oracle, bad vision policy | Perception errors | Improve detectors before ML policy |
| Low top-1, high top-3 | Label ambiguity | Soft labels |
| High false-action rate | State desync | GCD/CD estimators |
| Latency spikes | OCR every frame | Cache + ROI duty cycle |

---

## 10. What “success” means for this repository

The project is successful if it:

1. Quantifies what vision+rules can and cannot do on real UI footage.  
2. Provides a reproducible coaching analyzer.  
3. Offers a simulator path for autonomy research **without** official-server automation.  
4. Maintains a clean compliance boundary with automated guards.

The project is **not** successful if it measures itself by covert live-bot performance.
