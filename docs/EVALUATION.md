# Outcome-oriented offline evaluation

Evaluation is actuator-free: policies choose skills against saved demonstrations or synthetic scenarios, but no choice is executed in a game. Reports must keep observed evidence separate from policy-dependent estimates.

## Comparative report

```bash
PYTHONPATH=. python3 scripts/run_evaluation.py \
  --data-dir data/playmind/demonstrations \
  --checkpoints models/checkpoints/recurrent_skill_policy.json \
  --output-dir data/playmind/evaluation/recurrent-comparison
```

This writes `report.json`, `metrics.csv`, and `report.md`. If no demonstrations exist, fixed synthetic scenarios are used; these test policy plumbing and expected-label agreement, not gameplay competence.

For sequence-aware held-out recurrent classification:

```bash
PYTHONPATH=. python3 scripts/evaluate_behavior_clone.py \
  --data-dir data/playmind/demonstrations \
  --checkpoint models/checkpoints/recurrent_skill_policy.json \
  --history-length 16 --split test \
  --json-out data/playmind/evaluation/recurrent-test.json
```

## Report sections

- **Observed outcomes:** events and observation transitions recorded in demonstrations, such as explicit kill events, deaths, target/combat transitions, objective delta, recovery, and recorded reward proxies.
- **Label agreement:** top-1/per-skill agreement and confusion against demonstrated skills. Sequence-aware checkpoint evaluation additionally reports top-k and calibration.
- **Model predicted:** averages from auxiliary heads. These are predictions, not observations.
- **Counterfactual estimates:** marked `estimated_not_confirmed`; currently the primary proxy is agreement with the demonstrated action.
- **Decision validity:** invalid/masked proposals, fallback, emergency, and low-confidence rates.
- **Temporal:** switches, oscillations, commitment durations, prevented switches, premature interrupts, and repeated actions when evidence is available.

Observed demonstration outcomes are shared across policies replayed on the same data. A policy choosing a different skill does not prove that it would have changed a kill, death, or objective outcome.

## Baselines

`run_evaluation.py` includes:

- scripted policy
- empty CPU-only legacy-Q stub
- deterministic random-valid-skill reference
- hybrid (checkpoint-backed when supplied, otherwise scripted fallback)
- human-demonstration label reference

Each `--checkpoints` path adds a recurrent or legacy-MLP policy. A hybrid using the first recurrent checkpoint (otherwise the first legacy MLP) is also included. The human reference is an upper-bound label copier, not an independently evaluated agent.

## Episode KPI input

Use `--episodes-jsonl PATH` to aggregate stored episode records (kills/hour stub, skill success, invalid actions, death rate, reward, and duration). These values are only as reliable as the recorded fields and event evidence.

## Interpretation

Offline replay can test loading, masks, label fit, fallback behavior, and temporal diagnostics. It cannot establish visual learning or live improvement. Any live claim needs a predefined protocol, comparable baselines, enough gameplay episodes, and measured uncertainty.
