# Learning Architecture V2

**Status:** Implemented on `main` (PR #4 + PR #5)  
**Intent:** Replace primary coarse tabular Q-learning with hierarchical, temporally aware skill learning while keeping legacy Q as fallback.

**Still local / ops (not missing code):** calibrate your game window/ROIs; record demos; train a BC checkpoint before hybrid beats scripted.

---

## 1. Current architecture (as of audit)

### Primary loop (`playmind/owned_loop.py`)
1. Capture game window → `latest.png`
2. Build dict observation (`vision_obs_from_frame` + `enrich_obs_from_screen` + OCR/UI memory)
3. Sticky `LifeFSM` + `ProgressTracker` + `StuckTracker` + `TravelMemory` + `ProcessMemory`
4. Choose **raw action string** via priority stack:
   - Life FSM death/ghost ownership
   - Progress force / process prevention
   - Occasional Screen-LLM / Ollama
   - **`OnlinePolicy` tabular Q** (`use_learned_policy=True` by default)
   - Heuristic fallback
5. Soft reject/scrub some invalid actions
6. Actuate key/mouse
7. Recapture aftermath → `reward_owned` + directive/progress bonuses
8. `policy.update(..., done=False)` always — **no real episode terminals**
9. Experience append + replay into Q; periodic save of `policy.json`

### Observation flow
Raw pixels → HP ROI / target bar / desaturation / OCR / UI OCR hits → mutable `dict` with many optional keys silently defaulting (e.g. hp→0.5, bools→False).

### Action flow
Unbounded-ish string space: `OWNED_ACTIONS` **plus all keys ever seen in Q-table** plus dynamic `key:` / `hold:` / `click_label:` / LLM inventions → actuator parser.

### Reward flow
Heavy speculative shaping in `reward_owned` + `progress.reward_bonus` + stuck penalties: pixel motion, having a target, pressing attack, OCR phrase clears, etc. Kill often inferred from **target loss after combat**.

### Persistence (retained)
| File | Role |
|------|------|
| `data/playmind/owned/policy.json` | Tabular Q (legacy) |
| `experience.jsonl` | Transition log |
| `ui_memory.json` / `ability_memory.json` | UI/ability memory |
| `process_memory.json` | Death pipeline / preventions |
| Travel / lessons / dryrun logs | As today |

---

## 2. Why coarse tabular Q aliases situations

`owned_state_key` collapses the world to roughly:

`hp_bin | has_target | in_combat | motion | hostiles | progress_stage`

So these share one Q-row:
- True elite fight vs sticky false target
- Walking into a wall vs walking to a camp
- Different zones / quests / abilities
- Momentary OCR flicker of “has_target”

Combined with:
- **Invented actions re-entering the action space** from Q keys
- **`done=False` always** (no episodic credit assignment)
- **Rewards for pressing buttons / motion / target presence**

…the policy learns “mash 1 / tab / walk” habits that look locally rewarded but are not competent play.

---

## 3. What we retain

- Window capture + actuators (dry-run / live gates)
- Death/ghost heuristics + LifeFSM ideas (re-expressed as skills)
- OCR, UI memory, ability memory, process memory, travel memory
- Teacher / teach tooling (feeds demos / skill labels later)
- GUI logging surfaces (extended, not discarded)
- OnlinePolicy / experience as **legacy baseline**

## 4. What we replace (as primary path)

| Old | New |
|-----|-----|
| Raw-action Q every tick | High-level **skill** selection |
| Dict-only obs | Typed `Observation` + confidence |
| Single-frame decisions | Temporal history (16 steps) |
| Speculative shaping | Confirmed-event rewards |
| Always `done=False` | Explicit episodes |
| Q keys inflate action space | Strict allowlist masking |
| LLM free-form strings | Validated actions only |

---

## 5. New architecture

```text
Capture → Sensors(+confidence) → Observation
                                ↓
                         TemporalHistory
                                ↓
              Emergency rules (death/modal/stuck)
                                ↓
         HighLevelPolicy → Skill (masked)
                                ↓
                    Skill.step → low-level action
                                ↓
                 ActionMask.validate → Actuator
                                ↓
              Aftermath sensors → Events → Rewards_v2
                                ↓
                     EpisodeManager (+ logs)
```

### Policy modes
1. **Scripted** — deterministic skills (default runnable)
2. **LegacyQ** — experimental raw-action Q bridge
3. **BehaviorClone** — skill classifier (after demos/training)
4. **Hybrid** — emergencies scripted; BC if confident; else scripted; LegacyQ opt-in last

### Data flow (learning)
Human demos / scripted rolls → demonstration store → BC dataset (episode-wise split) → checkpoint → HybridPolicy.  
Replay env evaluates offline without sending keys.

---

## 6. Migration plan

1. Ship Observation + History + Skills + Masking + ScriptedPolicy + Episodes + Events + Rewards_v2 **without requiring new training data**
2. Wire `owned_loop` behind config `policy_mode: scripted|hybrid|legacy_q` (default **scripted** or **hybrid** with BC absent → scripted)
3. Mark existing `policy.json` as legacy; disable inventing actions from Q keys by default
4. Add demo recorder + BC training when ready
5. Keep dry-run and owned-game gates unchanged

### Expected limitations
- BC needs human demos before it beats scripted
- Sensor confidence starts heuristic until labeled metrics exist
- Skills are initially scripted approximations of current heuristics
- CUDA optional; CPU must train small models slowly

---

## 7. Implementation checklist

- [x] Audit + this document
- [x] `playmind/observations.py`
- [x] `playmind/history.py`
- [x] `playmind/skills/` + runtime
- [x] `playmind/action_masking.py`
- [x] `playmind/policies/` (scripted, legacy, hybrid, BC / SkillPolicyV2)
- [x] `playmind/episodes.py`
- [x] `playmind/events.py` + `rewards_v2.py`
- [x] `playmind/learning_v2_controller.py` + `owned_loop` config gate
- [x] `playmind/models/policy_v2.py` + BC train/eval scripts
- [x] Sensor metrics + `scripts/review_sensor_frames.py`
- [x] Owned GUI V2 (policy mode, demos, episode reset, diagnostics)
- [x] Tests for new modules
- [x] `playmind/config_v2.py` (validated settings)
- [x] `playmind/migration.py` + `scripts/migrate_legacy_learning.py`
- [x] `playmind/diagnostics.py` + `scripts/export_diagnostics.py`
- [x] Docs: QUICKSTART_V2, DEMONSTRATION_RECORDING, TRAINING, EVALUATION, SENSOR_LABELING, SKILLS, MIGRATION
- [ ] Richer multi-tab GUI / heavier sensor labeling product polish (follow-on)

Enable via `learning_v2.enabled` in `owned_game.json` — see [QUICKSTART_V2.md](./QUICKSTART_V2.md).

---

## 8. Config sketch (`learning_v2` section)

Full validated settings: `playmind/config_v2.py` (`LearningV2Settings`).

```json
"learning_v2": {
  "enabled": true,
  "policy_mode": "hybrid",
  "legacy_q_fallback": false,
  "history_length": 16,
  "bc_checkpoint": null,
  "confidence_threshold": 0.45,
  "device": "cpu",
  "seed": 0
}
```
