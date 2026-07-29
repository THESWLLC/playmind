# Episode lifecycle

## Status

`EpisodeLifecycleController` is implemented and unit-tested with explicit gameplay/recovery separation. Real-client sensor calibration and recovery timing still require local testing.

## States

Gameplay states:

- `alive_controllable`
- `combat`
- `alive_controllable_after_resurrection` (transition marker when recovery closes)

Non-gameplay or recovery-gated states:

- `unknown`, `loading`, `alive_not_controllable`
- `dead_dialog`, `release_confirm`
- `ghost`, `runback`, `resurrection_pending`

The controller classifies life/UI state, then applies temporal gating. A gameplay episode opens only after the configured number of consecutive frames are alive, UI-stable, and explicitly control-responsive (default: 3).

## Episode boundaries

Gameplay episodes:

- start as `new_run`, or as `controllable` after successful recovery
- end terminally on confirmed death or goal completion
- end terminally on session end/logout
- truncate on manual reset, fatal sensor failure, or maximum duration

Recovery episodes:

- start as `death_recovery` when death closes gameplay
- cover dialog, release, ghost/runback, loading, resurrection pending, and alive-but-not-controllable time
- end as `recovery_complete` only after the controllable-frame gate opens
- truncate on reset, sensor failure, or maximum duration; session end closes them terminally

Records link `previous_episode_id`, `death_event_id`, and `recovery_segment_id`, and can store death→resurrection and resurrection→controllable durations.

## Death and resurrection invariant

Death must **not** immediately open a resurrected gameplay episode:

```text
gameplay --death--> recovery --alive + stable UI + responsive controls
         + required consecutive frames--> new gameplay
```

The death update closes gameplay and opens a recovery segment. Seeing an alive frame can mark resurrection timing, but gameplay remains closed until controls are confirmed and the full frame gate passes. Ghost and loading observations never start normal gameplay.

## Recovery versus gameplay

Recovery is tracked as its own episode kind so death-handling time is not scored as ordinary controllable play. `EpisodeManager` persists completed records to `episodes.jsonl`; the lifecycle status also exposes current kind/state and start/end edges.

This boundary logic is tested with synthetic observations. It does not prove that a particular game's OCR/UI sensors identify resurrection or controllability correctly.
