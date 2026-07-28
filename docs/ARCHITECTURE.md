# Architecture Options — WoW-Like AI Research

**Companion docs:** [FEASIBILITY_STUDY.md](./FEASIBILITY_STUDY.md) · [COMPLIANCE_BOUNDARIES.md](./COMPLIANCE_BOUNDARIES.md) · [POC_PLAN.md](./POC_PLAN.md)

---

## 1. Design constraints (all architectures)

1. No automated input to the official World of Warcraft client.  
2. No memory reading, injection, packet manipulation, or anti-cheat bypass design.  
3. Prefer offline datasets and custom simulators for learning.  
4. If a live-adjacent coaching UX exists, the **human must execute** all actions.

---

## 2. Architecture A — Vision-only agent

### Intent

The agent receives **screenshots or video frames** (and optionally audio). It does **not** read game memory or network traffic. In this repository, Architecture A is specified as **perception + recommendation** (and full autonomy **only inside a non-Blizzard simulator** that renders similar UI abstractions).

### Components

| Component | Role |
|-----------|------|
| Frame source | Video file / screen capture of recordings (not live control loop on official servers) |
| Preprocess | Letterbox, ROI crop, UI scale normalize |
| Detectors | YOLO (nameplates, buttons, hazards), bar readers, icon classifiers |
| OCR | Health text, cast names, quest text |
| Audio encoder (optional) | Alert / voice-line classifier |
| State estimator | Fusion → structured `GameState` |
| Decision engine | Behavior tree + priority lists; optional IL policy head |
| Planner (slow path) | LLM/VLM for quest-level suggestions |
| Output | Overlay recommendations, eval logs — **not** input injection |

### Data flow

```text
Video frames (+ optional audio)
    → Vision/OCR/Audio encoders
    → Structured GameState
    → Decision engine (BT / rules / policy)
    → RecommendedAction + confidence
    → UI timeline / metrics store
    → (Forbidden on official live client) Actuator
```

### Latency expectations

| Stage | Target |
|-------|--------|
| Capture/decode | 5–20 ms |
| Detection + OCR (GPU) | 15–60 ms |
| Decision (rules/BT) | <1–5 ms |
| VLM/LLM planning | 200–3000+ ms (async only) |
| End-to-end recommend | **≤100 ms** aspirational for GCD coaching on GPU |

### Hardware requirements

- **Dev:** NVIDIA GPU ≥8 GB VRAM (or cloud GPU) for training  
- **Inference PoC:** RTX-class GPU preferred; CPU-only possible for OCR+rules at lower FPS  
- **Storage:** 50–500 GB for raw video datasets  

### Training requirements

- Labeled UI frames for detectors  
- Decision windows labeled with `recommended_action`  
- Optional imitation dataset with aligned key timelines (for sim or offline metrics only)

### Security risks

- Storing gameplay video may include chat PII — redact  
- Accidental addition of an actuator module — gate with compliance checklist  

### Compliance risks

- **High** if vision loop is wired to live mouse/keyboard on official WoW  
- **Low** for offline analysis / coaching display without actuation  

### Engineering difficulty

**7/10** (perception robustness dominates).

### Advantages

- Human-analogous sensory channel  
- No dependency on addon secret values  
- Clear scientific story for CV research  

### Disadvantages

- Brittle to UI packs, resolutions, camera  
- Hard to recover exact cooldown internals from pixels alone  
- Full autonomy still policy-blocked on official servers  

---

## 3. Architecture B — Addon-assisted research agent

### Intent

Use **only information exposed through permitted WoW addon APIs** for research into state richness and UI affordances. Verify combat-time limits. External consumers may log **out-of-combat** or **display-oriented** telemetry for offline study — **not** drive automated play.

### Components

| Component | Role |
|-----------|------|
| WoW addon (Lua sandbox) | Display state; optional SavedVariables snapshots when permitted |
| Export path | Manual export / logout SavedVariables / player-initiated dump — not a hidden pipe to a bot |
| Schema mapper | Convert addon snapshots → `GameState` |
| Decision engine | Runs **offline** on snapshots or in parallel as coach UI |
| Secure UI | Human clicks secure buttons; addon does not auto-cast |

### What addons can and cannot access in combat (summary)

| Category | Typical status |
|----------|----------------|
| Casting spells / targeting / moving via unprotected calls | **Blocked** without hardware events; combat lockdown |
| Configuring secure frames | **Out of combat only** |
| Displaying health/power/aura widgets | **Allowed** (including secret values bound to UI in Midnight model) |
| Lua branching on many combat secrets (Midnight+) | **Restricted** — secrets displayable but not “known” |
| CLEU parsing for competitive live logic (Midnight direction) | **Heavily constrained / removed for addon use cases described officially** |
| Filesystem/network sockets | **Not available** in sandbox |

See official Midnight article on secret values and [COMPLIANCE_BOUNDARIES.md](./COMPLIANCE_BOUNDARIES.md).

### Data flow

```text
WoW addon sandbox
    → Widget display (human visible)
    → Optional player-initiated SavedVariables / log export
    → Offline analyzer / coach
    → Recommendations for next session
```

### Latency expectations

- In-game display: frame-rate bound (Blizzard UI)  
- Offline analysis: non-real-time  

### Hardware requirements

- Standard gaming PC for client; modest machine for offline analysis  

### Training requirements

- Low for pure telemetry schema work  
- Optional supervised models if snapshots are paired with expert labels  

### Security risks

- Malicious addon patterns (obfuscation) — forbidden by addon policy  
- Accidentally designing an external real-time bot pipe — **compliance fail**  

### Compliance risks

- **Medium** for aggressive live coaching addons that approach decision automation  
- **High** if addon + external program closes the loop on inputs  
- Must remain free, source-visible, ToU/EULA compliant  

### Engineering difficulty

**5/10** for display/telemetry research; **9/10** if attempting combat decision addons under Midnight constraints (intentionally hard).

### Advantages

- Clean structured data when API allows  
- Aligns with Blizzard’s extension model  
- Useful for non-combat systems (UI layout, collections, etc.)  

### Disadvantages

- Combat decision research is being **disarmed** by design  
- Cannot authorize automated control  
- Patch volatility  

---

## 4. Architecture C — Custom WoW-like simulator

### Intent

Build a **simplified environment** resembling combat, movement, targeting, quests, and boss mechanics so AI can be developed and trained **without** the official client/servers.

### Components

| Component | Role |
|-----------|------|
| Simulation core | Entities, GCDs, resources, threat-lite, collision |
| Scenario packs | Open-world waypoints, dungeon routes, boss timelines |
| Render/UI abstraction | Minimal bars/nameplates (original art) |
| Gymnasium API | `reset` / `step` / rewards |
| Scripted baseline agent | Priority rotation for benchmarking |
| RL/IL trainers | PyTorch |
| Evaluation harness | Deterministic seeds, metrics export |

### Data flow

```text
Env.reset(seed)
    → Agent.observe(state or pixels)
    → Agent.act
    → Env.step
    → Logs / rewards / videos
    → Evaluation suite
```

### Latency expectations

- Headless step: **≪1–5 ms**  
- Pixel render step: 5–20 ms  
- Training throughput: thousands of steps/sec headless  

### Hardware requirements

- CPU-heavy for parallel envs; GPU for pixel-based policies  

### Training requirements

- Curriculum: train dummy → pack AI → boss script → multi-agent follow  
- Millions of steps for RL; far fewer for BT baselines  

### Security risks

- Low (local sim)  
- Ensure no Blizzard client code/assets are copied  

### Compliance risks

- **Low** if original IP and no connection to official servers  
- Avoid trademark-infringing names/assets in public distributions; use clear “WoW-*like* research sim” branding  

### Engineering difficulty

**6/10** for combat slice; **8/10** for broad open-world fidelity.

### Advantages

- Best place for RL/IL, navigation, recovery, raid-mechanic research  
- Reproducible science  
- Aligns with policy constraints  

### Disadvantages

- Sim-to-real gap (and real transfer must not target official automation)  
- Engineering investment before fancy demos  

---

## 5. Hybrid recommendation (research default)

**Perception (A)** for offline video + **Decision rules/BT** + **Simulator (C)** for autonomy claims + **Addon (B)** only for optional non-automated telemetry experiments.

Never combine A/B with an official-client actuator in this project.

---

## 6. Repository plan

```text
/docs                 Feasibility, compliance, architecture, PoC, evaluation
/simulator            Gymnasium env, scenarios, scripted mobs/bosses
/vision               Detectors, OCR, audio hooks, state fusion
/decision_engine      Behavior trees, priority lists, planners
/models               Training scripts, checkpoints (git-lfs later)
/data_collection      Schemas, importers for video/combat-log/labels
/evaluation           Metrics, regression suites, report generators
/ui                   Coaching timeline viewer (recommendation-only)
/tests                Unit/integration/compliance tests
/config               Spec rotations, thresholds, model paths
```

### Folder purposes

| Folder | Purpose |
|--------|---------|
| `docs/` | Normative research and policy docs |
| `simulator/` | Controlled training/eval environment |
| `vision/` | CV/OCR/audio → `GameState` |
| `decision_engine/` | Actions as recommendations or sim actuations |
| `models/` | Learned weights and training entrypoints |
| `data_collection/` | Dataset build pipelines; consent metadata |
| `evaluation/` | Automated scoring vs baselines |
| `ui/` | Human-facing coaching visualization |
| `tests/` | Including tests that fail CI if input-injection modules appear |
| `config/` | Declarative rotation/spec YAML |

---

## 7. Technology recommendations

### Language / runtime comparison

| Tech | Pros | Cons | Role |
|------|------|------|------|
| **Python** | Best ML/CV ecosystem; Gymnasium | Slower control loops | **Primary** |
| C# | Strong on Windows tooling | Weaker RL ecosystem | Optional tooling |
| C++ | Max performance | Slower iteration | Optional BT / sim core later |
| Rust | Safety + speed | Smaller CV/ML ecosystem | Optional future rewrite of sim |

### Libraries / models

| Tech | Role |
|------|------|
| **OpenCV** | Preprocess, templates, geometry |
| **PyTorch** | Training IL/RL/detectors |
| **ONNX Runtime** | Portable inference experiments |
| **YOLO (Ultralytics)** | UI/world object detection MVP |
| **PaddleOCR / EasyOCR** | Text ROIs |
| **py_trees** or BehaviorTree.CPP | Interpretable combat logic |
| **Gymnasium** | Sim API compatibility |
| Local VLMs (e.g. small Qwen-VL class) | Offline annotation / rare queries |
| Cloud multimodal APIs | Optional labeling aid; watch privacy/ToS |

### Primary stack recommendation

**Python + PyTorch + OpenCV + YOLO + OCR + py_trees + Gymnasium simulator + ONNX export path.**

**Why:** Matches greenfield repo needs; optimizes for honest offline evaluation; avoids building anything that looks like a live bot framework (no input drivers for WoW in tree).

---

## 8. Estimated difficulty by architecture

| Architecture | Difficulty | Compliance posture |
|--------------|------------|--------------------|
| A Vision-only (offline coach) | 7 | Good if no actuator |
| B Addon-assisted | 5–9 | Good only if display/export; poor for combat AI post-Midnight |
| C Custom simulator | 6–8 | Best for autonomy research |
