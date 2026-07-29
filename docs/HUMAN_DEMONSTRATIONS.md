# Human demonstrations

Planner quality depends on clean physical demonstrations from the owned game.
Synthetic and PlayMind-generated actions are useful for smoke tests but are
excluded from normal planner export.

## What is recorded

Starting a GUI demonstration creates:

```text
data/playmind/demonstrations/<session>/
├── meta.jsonl
├── session.json
└── frames/
```

Each schema-v2 row can contain a screenshot reference, structured observation,
physical keyboard/mouse events, generated events, source, lifecycle state,
sensor confidence, inferred skill, segmentation metadata, goal, and labels.
Keyboard events include down/up timestamps and hold duration. Mouse move,
button, wheel, coordinates, and optional normalized coordinates are supported.

Physical capture is optional and uses `pynput`. The recorder continues without
it, but such a session is not evidence of a physical demonstration.

## Recording procedure

1. Run System Doctor and confirm `capture_dependencies.pynput.available`.
2. Start the owned loop in `observe` or `shadow` with live keyboard unchecked.
3. In **Demonstrations**, set a narrow goal and start recording.
4. Focus the owned game and play one coherent segment yourself.
5. End at a meaningful boundary, mark `Success`, `Failure`, or `Bad`, and stop.
6. Review `session.json`, sampled frames, physical event counts, inferred
   skills, and source labels before export.

`F9` toggles recording globally when the optional listener starts. The GUI also
has Start/Stop controls. There are no implemented hotkeys for labels or manual
skill boundaries; use the GUI.

## Contamination prevention

- Never enable PlayMind live input while collecting human-only data.
- Stop recording before using chat, credentials, desktop shortcuts, another
  application, or the PlayMind GUI.
- Record short, goal-specific sessions and mark contaminated sessions `Bad`.
- Ensure rows intended for planner SFT have `input_source: "human"`,
  `human_training_eligible: true`, physical events, and a plausible inferred
  skill.
- Keep `playmind_generated`, `human`, and `unknown` sources separate. Mixed or
  empty physical-event windows become `unknown` or generated in GUI recording
  and should be reviewed.
- Do not pass `--include-ineligible` during normal export.

The capture class can ignore or label unfocused events when supplied a focus
provider. The current GUI starts it without a game-focus provider, so events
are normally labelled focused even after an Alt-Tab. This is a known
implementation limitation: use operational discipline and frame/event review,
not the focus field alone.

The listener heuristically marks likely chat/typing bursts and menu
interactions. These are review hints, not reliable automatic redaction.

## Rule-based skill segmentation

Implemented deterministic rules infer high-level skill spans:

- `Tab → forward → attack`: `acquire_target → approach_target → engage_target`
- low HP plus stop/food: `disengage → recover_health`
- death, ghost, or blocking modal: lifecycle recovery skills
- repeated zero motion or stagnation count: `unstuck`

Segments include event indices, confidence, rule IDs, eligibility, and optional
manual-override metadata. Rules are fixed-priority and intentionally narrow;
they do not understand arbitrary keymaps or infer every skill.

The default confidence threshold is `0.70`. Low-confidence segments remain
reviewable but are ineligible unless explicitly allowed. A library caller may
provide `manual_override` to produce a confidence-1.0 segment; the current GUI
does not expose this control.

## Episode boundaries

Splits are assigned by `episode_id`, so bad boundaries can leak nearly
identical context across examples or combine unrelated behavior. Start a new
episode after death/recovery, loading transitions, manual resets, or a genuinely
new task. Do not label death dialogs, ghost runback, or loading as ordinary
combat/navigation.

See [PLANNER_DATASET.md](./PLANNER_DATASET.md) for export eligibility and
[EPISODE_LIFECYCLE.md](./EPISODE_LIFECYCLE.md) for lifecycle boundaries.
