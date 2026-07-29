# Skill commitment

## Status

`SkillCommitmentTracker` and its controller integration are implemented and unit-tested. Their effect on real gameplay stability still needs measured live runs; no improvement claim is implied.

## Commitment rules

When a skill runtime starts, the tracker records a commitment. The default gates are:

- minimum commitment: 0.4 seconds
- maximum commitment: 25 seconds
- confidence margin: 0.15

Before the minimum expires, an ordinary alternative cannot replace the active skill. Afterward, a different proposal must have confidence at least:

```text
confidence_at_commitment_start + confidence_margin
```

The current skill is reconsidered when it finishes (`success`, `failed`, `timeout`, or `cancelled`), its preconditions become invalid, its maximum duration expires, or a permitted interrupt occurs. A quick A→B→A pattern can be counted and blocked as oscillation.

## Emergencies

Critical reasons bypass minimum duration, confidence hysteresis, and even a non-interruptible commitment:

- confirmed death or ghost state
- blocking modal or loading
- critical health or severe stuck state
- lost window focus
- fatal sensor disagreement

The controller asks the scripted policy for emergency handling. If the selected skill changes, it cancels the running skill, records the cancellation, and starts a new commitment.

## What is tracked

Each active commitment stores the skill, start time/tick, minimum and maximum durations, interruptibility and allowed reasons, starting confidence, and decision reason.

Aggregate diagnostics include:

- commitments started and current active skill
- accepted switches and prevented switches
- interrupt counts by reason
- oscillation count
- configured confidence margin and duration bounds

## Controller use

`LearningV2Controller` evaluates emergency and runtime state before accepting a new high-level proposal. It avoids polling on every render tick, continues stepping the active runtime while held, and periodically queries the policy after the minimum gate. A candidate is then passed through hysteresis and oscillation checks before any cancellation or restart.

Settings live under `learning_v2`:

```json
{
  "commitment_confidence_margin": 0.15,
  "minimum_commitment_seconds": 0.4,
  "maximum_commitment_seconds": 25.0
}
```

These defaults are implemented behavior, not tuned gameplay optima. Tune only from recorded switch, interruption, and outcome measurements.
