# WoW-Like AI Research Workspace

Feasibility and architecture study for **human-similar game decision research** related to World of Warcraft–*like* combat and UI understanding.

## Important

This repository is for **research, offline analysis, coaching prototypes, and custom simulators**.  
It is **not** a bot for official World of Warcraft servers.

Automated control of characters on Blizzard’s live service violates the [Blizzard EULA](https://www.blizzard.com/en-us/legal/08b946df-660a-40e4-a072-1fbde65173b1/blizzard-end-user-license-agreement) (bots / unauthorized automation). See `docs/COMPLIANCE_BOUNDARIES.md`.

## Documents

| Doc | Contents |
|-----|----------|
| [docs/FEASIBILITY_STUDY.md](docs/FEASIBILITY_STUDY.md) | Executive conclusion, task matrix, roadmap, build/research decision |
| [docs/COMPLIANCE_BOUNDARIES.md](docs/COMPLIANCE_BOUNDARIES.md) | Policy excerpts, permitted vs prohibited scope |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Architectures A/B/C, stack, folder plan |
| [docs/POC_PLAN.md](docs/POC_PLAN.md) | Offline combat coaching analyzer PoC |
| [docs/EVALUATION_PLAN.md](docs/EVALUATION_PLAN.md) | Metrics and automated test suite |

## Recommended next implementation

**Offline Combat Coaching Analyzer** — prerecorded video + combat log → UI detection → ability recommendation overlay → scoring. **No key presses.**

## Scaffold

```text
docs/ simulator/ vision/ decision_engine/ models/
data_collection/ evaluation/ ui/ tests/ config/
```

## Workspace note

This cloud-agent run started with **no attached git remote/repository**. Documentation and scaffold were created as a greenfield research tree on branch `cursor/wow-ai-feasibility-study-3737`.
