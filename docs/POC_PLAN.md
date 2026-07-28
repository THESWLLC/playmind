# Proof of Concept Plan — Offline Combat Coaching Analyzer

**Recommendation:** Build the **smallest viable non-controlling prototype** before any navigation or autonomy work.  
**Policy stance:** Prerecorded footage only; recommendations only; **no key presses** into World of Warcraft.

---

## 1. Why this PoC

| Goal | How this PoC serves it |
|------|------------------------|
| Test CV feasibility | Detect player/target/HP/resources/CDs from real UI pixels |
| Test decision quality | Recommend next ability; score vs combat log / expert |
| Stay compliant | No automated input, memory, packets, or anti-cheat work |
| Defer hard problems | No pathfinding, raids, PvP, or AH automation |
| Produce reusable modules | `vision/`, `decision_engine/`, `evaluation/`, `ui/` |

This is the safest and most technically useful first prototype identified in the feasibility study.

---

## 2. In scope

1. Ingest **prerecorded** gameplay video (user-consented).  
2. Optionally ingest a **combat log** export aligned by timestamp.  
3. Detect / estimate:
   - Player presence / frames of interest  
   - Target presence  
   - Player & target health (bars and/or OCR)  
   - Primary resource  
   - Action-bar cooldown states for a **single specialization**  
   - Cast bar if visible  
4. Run a **deterministic priority-list** recommender for that spec.  
5. **Display** the recommendation on a timeline UI (never execute).  
6. Record whether the recommendation matched:
   - the ability the player actually used (from log or manual label), and/or  
   - an expert-labeled “correct” action when multiple options exist.  
7. Export accuracy, latency, and confusion matrices.

---

## 3. Explicitly out of scope

- Live screen capture driving decisions in a closed loop with the official client  
- Any OS-level keyboard/mouse synthesis toward WoW  
- Memory reading, injection, packet tools  
- Navigation, questing, dungeon routing, raid assignment, PvP, auction house  
- Multi-spec support (defer)  
- Reinforcement learning (defer to simulator phase)

---

## 4. Smallest viable vertical slice

### Slice definition (MVP)

**One class specialization · default Blizzard UI · 1080p · indoor training-dummy or simple world combat · 10–20 minutes of video · aligned combat log.**

### MVP pipeline

```text
data_collection/import_video
        +
data_collection/import_combat_log
        ↓
vision/detect_ui_state(frame) → GameState
        ↓
decision_engine/priority_list(GameState) → RecommendedAction
        ↓
evaluation/score(recommendation, log_event_or_label)
        ↓
ui/timeline_viewer (playback + overlay)
```

### Acceptance criteria (MVP done)

| Criterion | Threshold |
|-----------|-----------|
| Health bar read accuracy (player) | ≥90% within 5% relative error on held-out frames |
| Cooldown icon state (ready vs not) for core abilities | ≥85% accuracy |
| Top-1 recommendation agreement with expert labels | ≥70% on non-ambiguous windows |
| Median pipeline latency (offline, GPU) | ≤100 ms/frame processed (batch OK if reported separately) |
| Input actuation code | **Zero** modules; CI grep guard |
| Human control | N/A (offline only) |

---

## 5. Component design (PoC-level)

### 5.1 Data collection

- Store videos under a local `data/` path (gitignored).  
- JSONL sidecars: `{frame_t, labels...}`.  
- Combat log parser → normalized events `{t, ability_id, source, target}`.  
- Consent file: recorder attestation + license for research use.

### 5.2 Vision

- ROI configs in `config/ui_layouts/default_1080p.yaml`.  
- Bar reader: color threshold + fill ratio; OCR fallback.  
- YOLO or template matching for action buttons (start with templates for speed).  
- Output pydantic/`dataclass` `GameState`.

### 5.3 Decision engine

- YAML priority list, e.g. `config/rotations/demo_spec.yaml`.  
- Pure functions: `recommend(state) -> Action`.  
- Unit tests with synthetic states (no video required).

### 5.4 Evaluation

- Align recommendations to next player cast within ±W ms.  
- Metrics: top-1 accuracy, top-3 accuracy, illegal-action rate, mean reaction lead/lag.  
- HTML or Markdown report generator.

### 5.5 UI

- Simple local web or desktop player: video scrubber, overlay text “Recommend: X”, green/red match marker.  
- No hooks into WoW.

---

## 6. Suggested label schema (PoC)

```json
{
  "frame_id": "clip01_000123",
  "timestamp_ms": 4100,
  "player_health_pct": 0.82,
  "target_health_pct": 0.56,
  "resource_pct": 0.40,
  "cast_bar": {"active": false, "name": null, "progress": 0.0},
  "cooldowns": {"ability_a": 0.0, "ability_b": 12.5},
  "enemy_location": null,
  "ground_hazard": false,
  "objective_marker": false,
  "death_state": false,
  "stuck_state": false,
  "current_ability": "AbilityA",
  "recommended_action": "AbilityB",
  "label_source": "expert|combat_log|model"
}
```

---

## 7. Data volume for PoC

| Asset | Quantity |
|-------|----------|
| Video | 10–30 minutes core; +10 minutes held-out |
| Manual expert labels | 500–2000 decision windows |
| Frame labels for bars/CDs | 500–1500 frames |
| Combat log | Full aligned export for same sessions |

Synthetic bar images (rendered in simulator later) can augment CD/HP readers.

---

## 8. Effort shape (engineering, not calendar)

| Workstream | Nature of work |
|------------|----------------|
| Log + video alignment | Parsers, clock skew handling |
| UI readers | Classical CV first, ML if needed |
| Priority engine | Small, test-heavy |
| Scoring + UI | Thin visualization |
| Compliance guards | CI tests, docs |

Avoid building a generic bot framework. Keep module names coaching/analysis oriented (`recommend`, `analyze`, never `send_key`).

---

## 9. Validation without the official live client

| Need | Method |
|------|--------|
| Perception | Prerecorded video |
| Decisions | Synthetic `GameState` unit tests + video replay |
| Latency | Offline profiling |
| Combat quality | Compare to log DPS/heal timelines only as **analysis**, not live optimization bots |

Optional later: replay the same recommender against **Architecture C** simulator states for controlled difficulty.

---

## 10. Exit criteria → next phase

Proceed to **custom simulator (Phase 4)** and richer IL experiments only after:

1. MVP acceptance metrics met on one spec.  
2. Compliance checklist still clean.  
3. Written decision that any live-overlay experiment remains recommendation-only.

**Do not** proceed to autonomous navigation against official servers.

---

## 11. Single sentence charter

> **We will analyze what a skilled player *should* press next from video and logs; we will not press it for them on official World of Warcraft.**
