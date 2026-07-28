# World of Warcraft AI Agent — Feasibility Study

**Status:** Research / architecture study only  
**Date:** 2026-07-28  
**Scope:** Technical possibility, system design, and policy separation — **not** a plan to deploy autonomous play on official Blizzard servers.

---

## 0. Repository and codebase inspection

### Findings

This cloud-agent run was launched **without an attached source repository** (`repoUrl: null`). The workspace at `/agent` was empty at start.

| Item | Finding |
|------|---------|
| Programming language | None present — greenfield |
| Frameworks | None present |
| OS assumptions | Linux agent environment; target research stack should also support Windows (typical WoW client host) for any future offline capture tooling |
| Existing modules | None |
| Reusable components | None |

A proposed repository scaffold has been created (see [ARCHITECTURE.md](./ARCHITECTURE.md) § Repository plan). All recommendations below assume a **new** research codebase.

---

## 1. Executive conclusion

### Verdict

**Technically possible but unsuitable for official servers because of policy restrictions**, with a practical research path that is **possible only for limited gameplay** and **practical only in a controlled environment**.

| Question | Answer |
|----------|--------|
| Can modern AI/CV/planning systems approximate human-like WoW play in principle? | **Yes**, for constrained slices (combat recommendations, simple navigation in a simulator, offline video analysis). |
| Can a full autonomous raid/PvP/open-world agent match a skilled human end-to-end today? | **Not reliably** — high variance, brittle perception, long-horizon planning, and social/coordination gaps remain. |
| Can such a system be lawfully used to automate play on official WoW servers? | **No** — Blizzard’s EULA prohibits bots and unauthorized gameplay automation. |
| What is a useful, safe research path? | Offline video analysis + non-controlling coaching overlays + a **custom WoW-like simulator** for learning and evaluation. |

### Technical possibility vs allowed use

- **Technical possibility** asks: *Could software observe pixels/UI/audio and choose actions that look like human play?* For many subskills, yes — especially combat rotations and UI reading.
- **Allowed use** asks: *Does Blizzard authorize automated control of characters or gameplay-facilitating software on official clients/servers?* Official policy answer: **no** for bots and unauthorized automation; **yes (with constraints)** for sandbox addons and human-operated assistive UI within Blizzard’s APIs.

This study therefore separates:

1. **Safe experiments** — custom simulator, offline footage, authorized test environments, private development sandboxes with owned assets.
2. **Accessibility / assistive tools** — player remains in control; recommendations or overlays only; no automated input.
3. **Prohibited on official servers** — unattended play, automated input, memory reading, injection, packet manipulation, anti-cheat evasion, selling automation services.

See [COMPLIANCE_BOUNDARIES.md](./COMPLIANCE_BOUNDARIES.md) for source-backed policy analysis.

---

## 2. Approach analysis

### 2.1 Screen capture and computer vision only

| Aspect | Assessment |
|--------|------------|
| Feasibility | High for UI bars, nameplates, icons; medium for 3D world semantics |
| Strengths | No game-memory access; mirrors human sensory channel |
| Weaknesses | Resolution/UI skin variance; occlusion; lighting; camera motion; GCD timing jitter |
| Compliance | Capture of **user-owned recordings** for offline analysis is the safe path. Live capture driving **automated input** on official clients is prohibited botting. |

### 2.2 Reading permitted UI data through WoW addons

| Aspect | Assessment |
|--------|------------|
| Feasibility | High historically for display/telemetry; **sharply reduced for combat decision logic** under Midnight “secret values” |
| What addons **can** do | Aesthetic UI; display secret values via widgets; out-of-combat configuration; SavedVariables; sandbox Lua |
| What addons **cannot** do | Automate protected actions without hardware events; call protected APIs in combat lockdown; (Midnight+) freely branch on many combat state values |
| Compliance | Official addon sandbox is the permitted extension path; must follow UI Add-On Development Policy (free, visible code, etc.). Exporting telemetry for an **external auto-controller** crosses into prohibited automation. |

### 2.3 OCR

Useful for health text, cast names, quest text, combat floating text, AH prices. Expected latency 10–80 ms per ROI with modern OCR; brittle under stylized fonts and motion blur. Best as a **supplement** to template matching / detector heads, not sole state source.

### 2.4 Object detection

YOLO-class detectors can locate nameplates, cast bars, action-button highlights, loot sparkles, and some ground FX given labeled data. Ground hazards and transparent AoE remain hard. Reliability depends heavily on UI consistency and camera FOV.

### 2.5 Audio recognition

Feasible for distinct alert sounds, boss voice lines, and “spell ready” cues. Complements vision under occlusion. Latency typically 50–200 ms with streaming classifiers. Alone, insufficient for full state estimation.

### 2.6 Behavior trees

Excellent for hierarchical combat/defensive/loot logic with interpretable fallbacks. Preferred backbone for **deterministic combat** in simulator and for coaching rule engines. Weak at long-horizon quest planning without higher-level planners.

### 2.7 Rule-based decision systems

Strong for known rotations (priority lists), interrupt rules, and defensive thresholds. Transparent and fast (<1 ms). Fail on novel encounters and incomplete perception. Ideal Phase-3 coaching core.

### 2.8 Reinforcement learning

Viable **inside a custom simulator** with shaped rewards. Sample-inefficient for full WoW complexity; sim-to-real transfer to official client is both hard and **policy-prohibited** if used for live automation. Use RL for research benchmarks, not live-server agents.

### 2.9 Imitation learning from recorded sessions

Strong for learning “next ability” and simple movement policies from expert video + combat logs. Needs aligned labels and action timestamps. Safe when trained offline and evaluated as recommendations; unsafe when paired with input injection on live servers.

### 2.10 Vision-language models

Useful for sparse high-level descriptions (“player is dead”, “quest marker ahead”) and debugging. Too slow/expensive for GCD-level combat (often 200 ms–several seconds). Best as asynchronous planners or annotators.

### 2.11 LLMs for high-level planning

Good for quest decomposition, inventory reasoning, and post-combat critique. Poor for hard-real-time combat. Must be grounded with structured state; hallucination risk is high without tools/verification.

### 2.12 Navigation (landmarks / map data)

Visual landmark navigation is research-grade and brittle outdoors. Map-assisted navigation is powerful but **map/minimap memory scraping or unauthorized client hooks** are out of scope for compliant designs. In a **custom simulator**, grid/navmesh navigation is straightforward and recommended.

### 2.13 Human-in-the-loop

The compliant sweet spot: AI proposes; human executes. Enables coaching, accessibility overlays, and evaluation without automated control.

### 2.14 Hybrid deterministic combat + AI planning

**Recommended research architecture:** rule/BT combat micro-policy + LLM/VLM for sparse macro goals + vision for state estimation — evaluated offline or in simulator; recommendations only if ever near a live client.

---

## 3. Task feasibility matrix

Legend for **policy**: Safe test = can be tested without automating the official client. Official = likely permit/prohibit on live servers.

| Task | Diff (1–10) | Required inputs | Recommended approach | Expected reliability | Main failure modes | Response-time need | Safe offline/sim test? | Official servers |
|------|-------------|-----------------|----------------------|----------------------|--------------------|--------------------|------------------------|------------------|
| Move through open world | 8 | Frames, camera, rough pose/map | Sim navmesh; visual odometry research offline | Low–medium outside sim | Stuck geometry, camera loss, path loops | 50–150 ms control | Yes (sim/video) | **Prohibit** if autonomous |
| Avoid obstacles | 7 | Depth/segmentation or collision signals | Sim physics; offline semantic seg | Medium | Transparent props, steep terrain | 33–100 ms | Yes | **Prohibit** if autonomous |
| Follow roads/paths | 6 | Road segmentation, minimap (owned data) | IL + rules in sim; offline CV | Medium | Crossroads ambiguity | 100–300 ms | Yes | **Prohibit** if autonomous |
| Find/target enemies | 6 | Nameplates, targeting reticle, audio | Detector + OCR; addon display only | Medium–high with UI cues | Occlusion, crowded packs | 50–150 ms | Yes | Auto-target: **prohibit**; display aids: constrained |
| Combat rotation | 4 | CDs, resources, GCD, buffs | Priority list / BT; IL for variants | High in known specs if state accurate | Misread CDs, latency, procs | **≤100–300 ms** (GCD) | Yes | Auto-cast: **prohibit**; coaching display: grey→risk |
| React to health/resources | 3 | HP/power bars, combat text | Threshold rules + OCR/CV | High | Bar skinning, absorbs | ≤100 ms defensive | Yes | Auto-use: **prohibit** |
| Defensive abilities | 5 | Damage intake, boss casts, HP | Interrupt/defensive BT | Medium–high | Unseen one-shots, secret values | **≤50–150 ms** | Yes | Auto-defensive: **prohibit** |
| Loot enemies | 5 | Loot sparkles, loot frame OCR | Detector + rules | Medium | Camera angle, range | 200–500 ms | Yes | Auto-loot bots: **prohibit** |
| Accept/complete quests | 7 | Quest UI OCR/VLM, objective markers | LLM planner + UI parsers (offline/sim) | Low–medium | Ambiguous text, travel gates | Seconds OK | Yes | Quest bots: **prohibit** |
| Navigate dungeons | 8 | Layout memory, party state | Sim graphs; offline video study | Low–medium | Pull sizing, patrols | 100–300 ms | Yes | **Prohibit** if autonomous |
| Follow a party | 7 | Party frames, nameplates, pathing | Multi-agent tracking + follow policy (sim) | Medium | LOS breaks, teleports | 100–200 ms | Yes | **Prohibit** if autonomous |
| React to boss mechanics | 8 | Ground FX, cast bars, audio | Scripted timelines (sim) + CV alerts | Medium | Novel telegraphs, secret data | **≤100–500 ms** by mechanic | Yes | Auto-solve: **prohibit**; native boss warnings OK |
| Participate in raids | 9 | Full encounter state, voice/comms | Human coordination required | Low for full autonomy | Assignment errors, adaptivity | Mixed (GCD + seconds) | Partial (sim bosses) | **Prohibit** autonomous |
| Play PvP | 9 | Opponent intent, burst windows | IL + opponent models (sim only) | Low | Unpredictability, latency | ≤50–100 ms | Yes (custom arena sim) | **Prohibit** autonomous |
| Use auction house | 5 | AH UI OCR/API-like parsers | UI automation in **sim**; rules for pricing research | Medium | UI changes, sniping races | 200–1000 ms | Yes | AH bots: **prohibit** |
| Manage inventory | 4 | Bags OCR, item icons | Rules + LLM for sorting advice | Medium–high advice | Icon collisions | Seconds OK | Yes | Auto-vendoring bots: **prohibit**; advice OK offline |
| Learn from mistakes | 6 | Logs, death recaps, rewards | Offline IL/RL in sim; post-game analysis | Medium | Credit assignment | Offline | Yes | Learning that drives live bots: **prohibit** |
| Recover when stuck/dead | 7 | Death UI, corpse location, stuck detectors | Recovery BT in sim; coaching tips live | Medium | Pathing dead-ends | Seconds | Yes | Auto-recover bots: **prohibit** |

---

## 4. Technology recommendations (summary)

**Primary research stack:** Python 3.11+ · PyTorch · OpenCV · Ultralytics YOLO · PaddleOCR/EasyOCR · ONNX Runtime for deployment experiments · Gymnasium-compatible custom simulator · BehaviorTree.CPP or `py_trees` · optional local VLM for sparse annotation.

**Why:** fastest CV/ML iteration, rich dataset tooling, easy Gymnasium RL loops, and clear separation from any live-client input layer (which this project must not implement for official servers).

Full comparison: [ARCHITECTURE.md](./ARCHITECTURE.md) § Technology recommendations.

---

## 5. Data strategy (summary)

Lawful datasets: user-recorded gameplay (consent), screenshots/clips, combat logs, addon telemetry **for analysis only**, manual labels, **synthetic simulator data**.

Prototype scale: ~2–5 hours labeled combat video + aligned combat logs for one specialization can support a coaching PoC; navigation/RL needs large synthetic corpora in-sim.

Details: [POC_PLAN.md](./POC_PLAN.md) and § Data strategy below.

### Example labels

`player_health`, `target_health`, `current_ability`, `cooldowns[]`, `cast_bar`, `enemy_bbox`, `ground_hazard`, `objective_marker`, `death_state`, `stuck_state`, `recommended_action`

### Volume guidance (basic prototype)

| Goal | Rough data need |
|------|-----------------|
| UI bar/CD detector MVP | 1–3k frames, mixed resolutions |
| Ability recommendation (1 spec) | 5–20k labeled decision windows + combat log |
| Mechanic alert classifier | Hundreds of clips per mechanic |
| Navigation policy | Prefer **synthetic** millions of sim steps |

---

## 6. Build-versus-research decision

### Recommendation: **Build a non-controlling coaching assistant** (offline-first), plus a **custom simulator** for any learning experiments.

| Option | Decision |
|--------|----------|
| Do not build | Rejected — research value exists if scoped correctly |
| Build only the simulator | Strong secondary track; needed before RL/IL autonomy claims |
| **Build a non-controlling coaching assistant** | **Primary recommendation** |
| Build an authorized research prototype | Only if Blizzard/partner authorization exists (not assumed) |
| Proceed with limited PoC | **Yes** — prerecorded video → detect UI → recommend ability → no key presses |

**Reasoning:** Full live autonomy is both unreliable at raid/PvP scale and clearly prohibited. Offline coaching + simulator maximizes learning about perception and decision quality with minimal compliance risk.

---

## 7. Implementation roadmap

### Phase 1: Repository and policy audit

| | |
|--|--|
| **Objective** | Establish legal/tech boundaries and empty scaffold |
| **Deliverables** | This doc set; folder structure; CONTRIBUTING/policy checklist |
| **Dependencies** | None |
| **Main risks** | Scope creep toward live automation |
| **Completion criteria** | Docs merged; explicit “no live input automation” rule in repo |

### Phase 2: Prerecorded-video UI detection

| | |
|--|--|
| **Objective** | Detect player/target HP, resources, CDs from video |
| **Deliverables** | Dataset schema; YOLO/OCR pipelines; metrics dashboard |
| **Dependencies** | Phase 1; consented recordings |
| **Main risks** | UI skin variance; labeling cost |
| **Completion criteria** | ≥90% HP bar IoU/read accuracy on held-out default UI clips |

### Phase 3: Combat recommendation engine

| | |
|--|--|
| **Objective** | Recommend next ability; compare to combat log / expert labels |
| **Deliverables** | Priority-list engine; overlay UI for video playback; accuracy reports |
| **Dependencies** | Phase 2 |
| **Main risks** | Label ambiguity (multiple correct actions) |
| **Completion criteria** | Top-1 recommend ≥70% agreement with expert on scripted encounters; **zero** input injection code paths |

### Phase 4: Custom simulator

| | |
|--|--|
| **Objective** | WoW-like combat/movement/quests without official client |
| **Deliverables** | Gymnasium env; scripted mobs/boss; logging |
| **Dependencies** | Phase 1 (can parallelize with 2–3) |
| **Main risks** | Over-fidelity / IP lookalike issues — keep abstract art/mechanics |
| **Completion criteria** | Agent can complete a scripted “dungeon wing” in sim with logged metrics |

### Phase 5: Imitation-learning experiments

| | |
|--|--|
| **Objective** | Train policies from sim + offline demos |
| **Deliverables** | BC/IL baselines; comparison to rule agent |
| **Dependencies** | Phase 4 |
| **Main risks** | Covariate shift; overclaiming transfer |
| **Completion criteria** | IL beats random; approaches rule agent on sim combat suite |

### Phase 6: Navigation research

| | |
|--|--|
| **Objective** | Landmark/path following **in simulator** |
| **Deliverables** | Nav benchmarks; stuck/recovery metrics |
| **Dependencies** | Phase 4 |
| **Main risks** | Underestimating open-world complexity |
| **Completion criteria** | ≥80% success on sim waypoint courses |

### Phase 7: Controlled human-in-the-loop evaluation

| | |
|--|--|
| **Objective** | Humans use recommendations while they retain full control (offline video or explicit coaching UX) |
| **Deliverables** | Usability study protocol; false-action analysis |
| **Dependencies** | Phases 2–3 |
| **Main risks** | Accidental coupling to live input tools |
| **Completion criteria** | Study completed with no automated control of official client |

---

## 8. Risk summary

| Risk class | Mitigation |
|------------|------------|
| Policy violation | No automated input to official clients; no memory/packet/anti-cheat work |
| Overclaiming AI readiness | Publish task matrix with reliability bands |
| Dataset legality | Consent forms; no scraped private server assets; abstract sim art |
| Secret-value / Midnight API shift | Do not depend on addon combat branching for research agent state |

---

## 9. Final recommendation (single safest next PoC)

**Implement an offline Combat Coaching Analyzer:** ingest prerecorded gameplay + combat log → detect UI state → recommend next ability on a video timeline → score vs log/expert — **never press keys**.

Details: [POC_PLAN.md](./POC_PLAN.md).
