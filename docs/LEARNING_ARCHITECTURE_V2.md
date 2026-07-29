# Learning Architecture V2

**Status:** Next phase implemented on `cursor/recurrent-policy-phase-3737`. Core behavior is covered by automated tests; useful learned play still needs real demonstrations and measured evaluation.
**Intent:** Replace primary coarse tabular Q-learning with hierarchical, temporally aware skill learning while keeping legacy Q as an explicit fallback.

No visual-learning or live-gameplay improvement claim is made.

---

## 1. Current architecture

### Legacy baseline (`playmind/owned_loop.py` when V2 is disabled)
1. Capture game window → `latest.png`
2. Build dict observation (`vision_obs_from_frame` + `enrich_obs_from_screen` + OCR/UI memory)
3. Sticky `LifeFSM` + `ProgressTracker` + `StuckTracker` + `TravelMemory` + `ProcessMemory`
4. Choose a raw action string via the original priority stack:
   - Life FSM death/ghost ownership
   - Progress force / process prevention
   - Occasional Screen-LLM / Ollama
   - `OnlinePolicy` tabular Q
   - Heuristic fallback

### V2 path (`learning_v2.enabled=true`)

Structured observations are encoded with [feature schema v2](./FEATURE_SCHEMA.md), accumulated into a last-16 history, and passed to a high-level skill policy. The recurrent BC policy uses a GRU; scripted handling remains available for emergencies and low-confidence fallback. [Skill commitment](./SKILL_COMMITMENT.md) gates switching, and the [episode lifecycle](./EPISODE_LIFECYCLE.md) separates gameplay from death recovery.

---

## 2. Why the legacy coarse tabular Q aliases situations

`owned_state_key` collapses the world to roughly:

`hp_bin | has_target | in_combat | motion | hostiles | progress_stage`

So these share one Q-row:
- True elite fight vs sticky false target
- Walking into a wall vs walking to a camp
- Different zones / quests / abilities
- Momentary OCR flicker of “has_target”

In the legacy path this combines with:
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
                   Commitment / hysteresis
                                ↓
                    Skill.step → low-level action
                                ↓
                 ActionMask.validate → Actuator
                                ↓
              Aftermath sensors → Events → Rewards_v2
                                ↓
              EpisodeLifecycleController (+ logs)
```

### Policy modes
1. **Scripted** — deterministic skills (default runnable)
2. **LegacyQ** — experimental raw-action Q bridge
3. **BehaviorClone** — recurrent skill classifier by default (after demos/training)
4. **Hybrid** — emergencies scripted; BC if confident; else scripted; LegacyQ opt-in last

### Data flow (learning)
Human demos / scripted rolls → demonstration store → BC dataset (episode-wise split) → checkpoint → HybridPolicy.  
Replay env evaluates offline without sending keys.

---

## 6. Next-phase implementation

Implemented on this branch:

- schema-v2 value/known/confidence features and train-only normalization
- episode-local recurrent windows and `RecurrentSkillPolicyV2`
- stateless bounded-history live inference and strict skill-logit masking
- skill commitment, hysteresis, emergency interrupts, and switch diagnostics
- gameplay/recovery episode lifecycle with controllability gating
- evidence-separated offline evaluation and baseline comparisons

Still needed for a useful deployment:

- calibrate sensors and controls for the owned game
- record sufficient labeled demonstrations
- train and evaluate on held-out episodes
- run controlled live trials before claiming gameplay improvement

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
- [x] `playmind/models/feature_schema.py` (`FEATURE_SCHEMA_VERSION=2`)
- [x] `playmind/models/recurrent_policy.py` + sequence-aware training
- [x] `playmind/skill_commitment.py` + controller gating
- [x] `playmind/life_episode.py` + recovery/gameplay boundaries
- [x] Outcome-section evaluation + scripted/legacy/random/human/checkpoint baselines
- [x] Sensor metrics + `scripts/review_sensor_frames.py`
- [x] Owned GUI V2 (policy mode, demos, episode reset, diagnostics)
- [x] Tests for new modules
- [x] `playmind/config_v2.py` (validated settings)
- [x] `playmind/migration.py` + `scripts/migrate_legacy_learning.py`
- [x] `playmind/diagnostics.py` + `scripts/export_diagnostics.py`
- [x] Docs: QUICKSTART_V2, DEMONSTRATION_RECORDING, TRAINING, EVALUATION, RECURRENT_POLICY, SKILL_COMMITMENT, EPISODE_LIFECYCLE, FEATURE_SCHEMA
- [ ] Richer multi-tab GUI / heavier sensor labeling product polish (follow-on)
- [ ] Real demonstration corpus, trained checkpoint results, and measured live trials
- [ ] Visual learning (encoder is currently a placeholder)

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
  "commitment_confidence_margin": 0.15,
  "minimum_commitment_seconds": 0.4,
  "maximum_commitment_seconds": 25.0,
  "controllable_frames": 3,
  "device": "cpu",
  "seed": 0
}
```
