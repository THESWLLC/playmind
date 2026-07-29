# Learning V2 — Next Phase Audit

**Date:** 2026-07-29  
**Branch intent:** Recurrent temporal BC, skill commitment, episode lifecycle, feature validity, outcome eval  
**Baseline:** `main` after PR #5 / #6 (97 passed, 1 skipped)

This audit was verified against source code, not documentation claims.

---

## 1. Current behavior (verified)

### Policy / features (`playmind/models/policy_v2.py`)
- Structured feature dim is **38** (20 obs + 7 life-phase one-hot + 11 temporal summary).
- Missing player HP → `0.5`; other missing scalars → `0.0`; unknown booleans → `0.0` (same as known `False`).
- `SkillPolicyNet` is a last-frame MLP. Direct 3-D input uses `features[:, -1, :]`.
- Public `predict()` flattens / truncates; it does not run a true sequence.
- Aux heads (`target_valid`, `combat`, `death`) exist but are **not trained**.
- Allowed skills are enforced by post-hoc replacement, not logit masking.
- Checkpoints: JSON metadata + sibling `.pt`; `trained=True` can be trusted even if weights missing.

### Dataset / training (`playmind/training/`)
- Dataset builds causal windows (default length 16) with left-pad-by-duplication, episode-hash splits.
- Dataset per-timestep features are **13-D**, not the policy’s 38-D layout.
- Train/eval **ignore** windows and rebuild a single-frame vector from the target row only.
- Temporal summary features are all zeros during training.
- Loss is skill CE only; aux unused. Bad/unlabeled rows included by default.
- Eval metrics = exact label accuracy + confusion matrix (no outcome metrics).

### History / observations
- `TemporalHistory` maxlen 16; `summarize()` exists but `current_skill_duration` is hardcoded `0.0`.
- `Observation` / `SensorValue` keep `None` for unknown; known = `value is not None`.
- Feature encoding collapses unknowns into numeric defaults (problem is in `policy_v2`, not typed obs).

### Controller / skills / episodes
- `choose_action()` queries the high-level policy **every tick**. Active skill continues only if the name matches.
- No minimum commitment, confidence margin, or hysteresis.
- On alive→dead: episode ends with `death`, then **immediately** starts `resurrected` — before resurrection/controllable confirmation.
- `KillConfirmed` can fire from `target_lost_after_combat` + `combat_ended` alone (correlated with `has_target`/`in_combat` coupling in owned loop).
- Skill success/timeout rewards require `nxt["skill_*"]` flags the controller does not set.
- Skill attempt counters double-count (start + terminal both call `note_skill_attempt`).
- Runtime: `start`/`step`/`cancel` exist; `retry_limit` not enforced; statuses are informal strings.

### Replay / evaluation
- `ReplayEnv` = per-frame `choose_skill` vs demo label; no runtime, events, rewards, or outcomes.
- Fixed allowlist; no live `mask_skills(obs)` or temporal history.
- `run_evaluation.py` compares scripted vs legacy stub on synthetic scenarios via label agreement.

---

## 2. Verified problems

1. Temporal BC is fake: windows discarded; MLP keeps last frame only.
2. Unknown sensors masquerade as known defaults in feature vectors.
3. Dataset feature schema ≠ policy feature schema (13 vs 38).
4. Aux heads untrained; no feature normalization stats in checkpoints.
5. Skill reselection every tick → churn / oscillation risk.
6. Death immediately opens a falsely labeled `resurrected` episode.
7. Ghost/loading/controllable gates missing from episode start rules.
8. Kill confirmation too weak (target-loss pair).
9. Offline eval is label matching, not gameplay outcomes.
10. No recurrent checkpoint type / feature schema versioning / legacy adapter clarity.
11. No modular observation encoder for future visual fusion.
12. Mask edge bugs: `stuck_hint="none"` truthy; `blocking_modal` ignored by `mask_skills`.

---

## 3. Files affected

| Area | Primary files |
|------|----------------|
| Audit | `docs/LEARNING_V2_NEXT_PHASE_AUDIT.md` |
| Features | `playmind/models/feature_schema.py` (new), `playmind/models/policy_v2.py`, `playmind/models/encoders.py` (new) |
| Dataset | `playmind/training/dataset.py` |
| Train/eval BC | `playmind/training/train_behavior_clone.py`, `evaluate_behavior_clone.py`, scripts |
| Commitment | `playmind/skill_commitment.py` (new), `learning_v2_controller.py` |
| Runtime | `playmind/skills/runtime.py`, `skills/base.py` |
| Episodes | `playmind/episodes.py`, `life_episode.py` (new or extend), controller |
| Events/rewards | `playmind/events.py`, `rewards_v2.py` |
| Live mask | `playmind/action_masking.py`, hybrid/controller inference |
| Eval | `replay_env.py`, `evaluation/*`, `scripts/run_evaluation.py` |
| Docs | README, QUICKSTART_V2, TRAINING, EVALUATION, new RECURRENT/COMMITMENT/EPISODE/FEATURE docs |
| Tests | new `tests/test_recurrent_*`, `test_skill_commitment.py`, `test_episode_lifecycle.py`, etc. |

---

## 4. Migration strategy

1. Introduce **feature schema v2** (value/known/confidence) with stable ordered names; keep v1 MLP path as `model_type: structured_mlp_legacy`.
2. Implement `RecurrentSkillPolicyNet` (GRU) consuming encoded embeddings; default live inference = **stateless last-16 window**.
3. Dataset emits `(T,F)` + length + padding mask; episode-separated splits; exclude bad by default.
4. Training saves normalization stats + full metadata (`checkpoint_schema_version=2`, `feature_schema_version=2`, `model_type=recurrent_skill_policy`).
5. Skill commitment gate in controller before BC query; emergency interrupts unchanged priority.
6. Episode lifecycle: end on death; recovery segment; start next gameplay only after controllable alive confirmation.
7. Strengthen kill evidence classes; confidence thresholds for rewards.
8. Outcome-based eval + baselines (scripted, legacy Q, legacy MLP, recurrent, hybrid).
9. Encoder abstraction stubs for future visual; no claim of visual learning.

---

## 5. Backward-compatibility strategy

- Config modes preserved: `scripted | hybrid | behavior_clone | legacy_q`.
- Old MLP checkpoints: load via explicit legacy adapter **or** refuse with clear error → scripted fallback.
- Never reinterpret MLP weights as GRU.
- Feature schema mismatch → error / safe fallback, never silent reorder.
- Scripted mode remains runnable with no demos/torch.
- Dry-run and owned-game keyboard gates unchanged.

---

## 6. Test plan

- Recurrent: earlier timestep changes output; padding ignored; variable length; single-step; checkpoint round-trip; schema reject; logit masking.
- Features: unknown≠known-false; schema metadata; train-only normalization.
- Commitment: persist across ticks; hysteresis; emergency interrupt + cancel; completed release.
- Episodes: death ends; no immediate resurrected gameplay; ghost/loading blocked; controllable starts next.
- Dataset: no cross-episode windows; no shared episodes across splits; padding masks; bad excluded.
- Eval: observed vs counterfactual separation; baselines; switch metrics.
- Full suite must stay green; no live game required.

---

## 7. Expected limitations

- Outcome replay is still offline; counterfactual success after alternate skills is estimated, not confirmed.
- Without real human demos, recurrent BC cannot outperform scripted in live play.
- Stateful GRU live inference is optional; initial path is safer last-16 stateless.
- Visual encoder is interface-only until synchronized frame datasets exist.
- Kill confirmation remains heuristic without UI/combat log APIs.
- CPU GRU training is slow; CUDA optional.

---

## 8. Implementation order (this branch)

1. This audit  
2. Feature schema V2 + masks + normalization  
3. Dataset windows  
4. Recurrent model + checkpoint migration  
5. Training  
6. Skill commitment + runtime  
7. Episode lifecycle + events  
8. Live inference  
9. Outcome eval + baselines  
10. Tests + docs + full suite
