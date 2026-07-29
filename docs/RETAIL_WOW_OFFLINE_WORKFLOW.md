# Retail World of Warcraft offline workflow

For retail World of Warcraft, PlayMind Studio is a post-recording analysis
tool. It is not a bot, live coach, capture service, or input logger.

## Allowed architecture

```text
recording file supplied by authorized user
  -> offline FFmpeg probe/frame extraction
  -> offline heuristic analysis
  -> human annotation/correction
  -> permission-gated dataset export
  -> offline training
  -> actuator-free held-out evaluation
```

The `retail_wow_offline_only` profile permits offline video import, frame
analysis, human annotation, dataset export, and evaluation. It disables live
perception/capture/planning, physical input logging, process access, and
generated input; it prohibits all live planning modes.

## Prohibited workflow

Do not:

- point PlayMind capture/owned-loop tools at the official WoW client;
- read its process memory, inject code, hook packets, or inspect processes for
  gameplay data;
- send keyboard/mouse/controller actions;
- log global gameplay input for imitation learning;
- build live shadow/assist/hybrid/autonomous planning around the client;
- evade anti-cheat or hide the tool;
- scrape videos without permission;
- treat an offline-trained model as authorization for live use.

`start_playmind.bat` opens the separate owned-game lab. Do not use it for
retail WoW. Start the file-only surface with `start_playmind_studio.bat`
(Windows) or `python scripts/start_studio.py`.

## Safe operator procedure

1. Record gameplay using ordinary user-controlled recording software with no
   PlayMind integration. The player performs every action.
2. Exit the client and work from the saved recording file.
3. Confirm ownership/permission, consent, training rights, privacy, retention,
   and redistribution limits.
4. Import under `retail_wow_offline_only`; prefer local `copy` mode only when
   retention is permitted.
5. Extract/analyze frames offline.
6. Review every segment/detection/correction; exclude ambiguous/private data.
7. Keep source-related examples in one split and reserve untouched projects
   for benchmark use.
8. Train and evaluate offline.
9. Keep derived registry records marked `live_use_prohibited` with allowed uses
   limited to offline evaluation/research.
10. Report results as offline benchmark evidence only.

## Current enforcement and limits

Implemented:

- Studio modules have a tested static boundary against `playmind.actuators`,
  `playmind.owned_loop`, and `pyautogui`;
- `StudioApp` does not import the actuator module;
- protected profile capabilities are permanently false;
- model registry rejects live use and promotion of
  `live_use_prohibited` records;
- smoke artifacts cannot be promoted or used as models.

Limits and non-automatic controls:

- the Studio GUI is a local path/form dashboard, not a playable video timeline;
- `detect_forbidden_live_context` evaluates a process-name list supplied by a
  caller; Studio deliberately does not inspect processes itself;
- provenance checkboxes cannot verify the truth of permission;
- a user can run unrelated owned-game scripts separately;
- exported files need operational access controls outside Python.

Safety therefore depends on both architecture and operator procedure.

## Handling footage

Retail footage may contain character/account names, guild chat, voice chat,
friend identities, private messages, server names, and third-party artwork or
audio. Minimize collection, redact where required, store locally, and do not
commit media/frames/derived datasets. Keep consent evidence outside Git.

If a participant revokes consent, follow the project/source hash through
frames, annotations, exports, benchmarks, and derived model runs. Stop sharing
before cleanup.

## Results language

Acceptable:

> On a held-out, permissioned set of 120 offline scenarios, candidate X
> improved planner benchmark score from A to B with illegal-skill rate C.

Not supported:

> The model plays WoW better, is safe to use in-game, or avoids detection.

See [Compliance Boundaries](./COMPLIANCE_BOUNDARIES.md),
[Data Provenance](./DATA_PROVENANCE_AND_PERMISSION.md), and
[Account Safety Architecture](./ACCOUNT_SAFETY_ARCHITECTURE.md).
