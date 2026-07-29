# Account safety architecture

PlayMind's account-safety boundary for protected games is architectural:
Offline Studio processes files and has no path to generated input. This reduces
risk but is not a guarantee against account action, policy changes, operator
misuse, or defects outside Studio.

## Separation of products

| Surface | Purpose | Retail WoW |
|---|---|---|
| Offline Studio (`playmind.studio`) | Imported-file analysis, annotation, dataset building, offline train/eval | The only documented PlayMind workflow |
| Owned-game lab (`start_playmind.bat`, owned loop/GUI) | Authorized games/test environments that the operator owns or may automate | Do not use |

The products share some offline planner/data code but not the Studio live/input
surface. The separate Studio launchers are `scripts/start_studio.py` and
`start_playmind_studio.bat`; the owned lab uses different entrypoints and port.

## Enforced Studio boundary

`retail_wow_offline_only` sets all of these false:

- live perception and capture;
- physical gameplay input logging;
- live planning;
- process access;
- generated input.

It prohibits `shadow`, `assist`, `hybrid`, and `autonomous` live planning modes.
Studio module tests reject imports of `playmind.actuators`,
`playmind.owned_loop`, and `pyautogui`, and verify importing `StudioApp` does
not load the actuator module.

The allowed dataflow is one-way:

```text
local media file -> offline artifacts -> datasets -> offline models/reports
```

There is no Studio output edge to a game client.

## Defense layers

1. **Entrypoint separation:** Studio must have a distinct launcher; the
   owned-game launcher is labeled separately.
2. **Capability profile:** protected profiles cannot enable live/input
   capability, even through construction parameters.
3. **Import boundary:** Studio code cannot import actuator/owned-loop modules.
4. **File-only APIs:** import, extraction, analysis, annotation, correction,
   export, and evaluation operate on paths/records.
5. **Provenance gates:** unknown/unconsented/unlicensed sources are excluded
   conservatively.
6. **Human review:** suggestions are ineligible until reviewed.
7. **Registry restrictions:** `smoke` and `live_use_prohibited` are hard
   promotion errors; allowed-use checks reject live/generated-input use.
8. **Audit artifacts:** manifests, evaluation reports, registry changes, and
   benchmark versions preserve evidence.

## Important implementation limits

- `assert_studio_safe()` currently asserts a static invariant; it is not an OS
  sandbox.
- `detect_forbidden_live_context()` does not enumerate processes. A caller may
  supply a process snapshot, but Studio itself intentionally has no process
  access.
- Python import tests do not prevent a user from modifying code or launching a
  separate tool.
- Filesystem permissions, encryption, backups, and network egress are outside
  the Studio package.
- Provenance statements are operator-provided and cannot verify consent.
- The Studio GUI binds to loopback by default and blocks known live/input API
  paths/options, but it is an application check rather than an OS sandbox.
- Model restrictions protect registry operations, not arbitrary copying of an
  adapter file.

## Operator controls

- Use a separate local directory/account for protected-game offline research.
- Keep media and model artifacts out of synced/public folders.
- Exit the game client before post-processing as a simple procedural control;
  the architecture does not require or inspect the client.
- Never run owned-game capture, teleop, or actuator scripts for retail WoW.
- Do not grant Studio administrator privileges.
- Verify every intended command and path before execution.
- Restrict outbound network access when handling sensitive footage.
- Keep base models and transcripts local unless permission explicitly allows
  an external service.
- Re-audit the current EULA/policies before changing scope toward any live
  feature.

## Model artifact policy

An artifact derived from the protected profile should be registered with:

```python
{
    "live_use_prohibited": True,
    "source_game_profile": "retail_wow_offline_only",
    "allowed_uses": ["offline_evaluation"],
}
```

The dataset bridge does not currently register models or automatically pass
these fields into SFT registration. Operators must ensure protected-profile
lineage is preserved; ordinary SFT training currently registers non-smoke
candidates without those restrictions by default. This is a real integration
gap—do not assume provenance automatically propagates into the model registry.

For protected data, either use `--no-register` and register the artifact
manually with restrictions, or update/rebuild the integration before relying
on registry enforcement. Do not promote such an artifact to general
production.

## Incident response

If any PlayMind process appears to interact with a protected live client:

1. stop PlayMind and the client;
2. preserve logs/configuration without publishing sensitive data;
3. verify which entrypoint and modules ran;
4. quarantine generated artifacts and credentials;
5. do not reproduce against a live account;
6. test only with synthetic fixtures or an original simulator;
7. document and close the architectural path before resuming offline work.

See [Retail WoW Offline Workflow](./RETAIL_WOW_OFFLINE_WORKFLOW.md) and
[Compliance Boundaries](./COMPLIANCE_BOUNDARIES.md).
