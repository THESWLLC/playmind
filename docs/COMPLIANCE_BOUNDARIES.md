# Compliance Boundaries — World of Warcraft AI Research

**Purpose:** Separate safe research, assistive tools, and prohibited automation using **official Blizzard sources**.  
**Not legal advice.** Policies change; re-audit before any implementation near a live client.

---

## 1. Principle of this repository

This project may research:

- Offline computer vision on **user-consented** recordings
- Decision engines that **recommend** actions without executing them
- Custom, original **simulators** that are WoW-*like* but not derivatives of Blizzard client code/assets
- Human-in-the-loop coaching UX where the **player retains control**

This project must **not**:

- Automate control of characters on official World of Warcraft servers
- Bypass anti-cheat, read process memory, inject code, or manipulate packets
- Design evasion, process hiding, or input spoofing to avoid detection
- Sell or distribute gameplay automation services for official WoW

---

## 2. Official sources consulted

| Source | URL |
|--------|-----|
| Blizzard End User License Agreement | https://www.blizzard.com/en-us/legal/08b946df-660a-40e4-a072-1fbde65173b1/blizzard-end-user-license-agreement |
| Prohibitions on Third-party Software (Kaivax, Aug 2025) | https://us.forums.blizzard.com/en/wow/t/prohibitions-on-third-party-software/2142972 |
| Policy Update for Input Broadcasting (Kaivax, May 2021) | https://us.forums.blizzard.com/en/wow/t/policy-update-for-input-broadcasting-may-2021/956613/1 |
| UI Add-On Development Policy | https://us.forums.blizzard.com/en/wow/t/ui-add-on-development-policy/24534 |
| Combat Philosophy and Addon Disarmament in Midnight | https://worldofwarcraft.blizzard.com/en-us/news/24246290/combat-philosophy-and-addon-disarmament-in-midnight |

---

## 3. Brief excerpts (policy anchors)

### 3.1 Bots and cheating (EULA § License Limitations — Cheating)

From the Blizzard EULA, under license limitations regarding cheating, Blizzard prohibits creating/using/distributing:

> **bots;** i.e. any code and/or software, not expressly authorized by Blizzard, that allows the **automated control of a Game or part of a Game**, or any other feature of the Platform, e.g. the **automated control of a character in a Game**;

Also prohibited:

> **hacks;** i.e. accessing or modifying the software of the Platform in any manner not expressly authorized by Blizzard;

> any code and/or software, not expressly authorized by Blizzard, that can be used in connection with the Platform and/or any component or feature thereof which **changes and/or facilitates the gameplay** or other functionality;

And under Derivative Works:

> … reverse engineer, derive source code from, modify, disassemble, decompile, or create derivative works based on or related to the Platform.

**Implication for this study:** An AI agent that presses keys / moves the mouse / drives character actions on the official game without express authorization is a **bot** under the EULA. Memory reading, client modification, and similar techniques fall under **hacks** / unauthorized modification.

### 3.2 Third-party software that modifies the client (official forum, Aug 2025)

Kaivax (Blizzard):

> We regularly implement new and better cheat detection aimed at identifying those who reverse-engineer and bypass our security protocols to **modify the game client and/or control the game in ways that are prohibited**.

> The use of third-party software that **modifies the World of Warcraft game client** is against our Terms of Service.

**Implication:** Client modification and prohibited control methods are actively enforced, including permanent account closure.

### 3.3 Input broadcasting / automation-like multiboxing (May 2021)

Kaivax:

> … prohibit the use of all software and hardware mechanisms to **mirror commands** to multiple World of Warcraft accounts at the same time, or to **automate or streamline multi-boxing** in any way.

> … activities which effectively **replicate automated gameplay** are contrary to … the Blizzard End-User License Agreement (EULA).

**Implication:** Even partial automation / streamlining that replicates automated gameplay is treated as a policy violation.

### 3.4 Add-on policy (permitted sandbox, with rules)

From the UI Add-On Development Policy (official posting):

> 1) **Add-ons must be free of charge.** …

> 2) **Add-on code must be completely visible.** …

> 7) Add-ons must abide by World of Warcraft ToU and EULA.

> 8) Blizzard Entertainment has the right to **disable add-on functionality** as it sees fit.

**Implication:** Research may discuss **in-sandbox** addons for display/telemetry. Addons are not a license to build external bots. Paid/obfuscated addon distribution violates addon policy.

### 3.5 Midnight addon disarmament (combat data as “secret values”)

Ion Hazzikostas / official Midnight article:

> Information about the current combat state is designated as a **“secret value”** that can be **displayed** by addons, but not **“known”** by them.

> … addons … can’t “know” with certainty whether you or your target have a specific debuff currently active, or what the cooldown of a given ability is.

> **Addons should no longer offer a competitive advantage in WoW combat.**

Also notes native tools such as Assisted Highlight / One-Button Rotation, Boss Warnings, and Combat Audio Alerts as first-party accessibility/approachability features.

**Implication for Architecture B:** Addon-assisted agents that *compute* combat decisions from live combat state are increasingly blocked by design. Display-only and human-driven secure actions remain the intended model. External AI that screenshots the UI to drive automation is still **automated control** if it sends inputs.

---

## 4. Classification table

### 4.1 Experiments that can be performed safely (research scope)

| Experiment | Notes |
|------------|-------|
| Train detectors on consented screenshots/video | No live control |
| Align recommendations with combat logs offline | Analysis only |
| Build Gymnasium WoW-*like* combat/nav sims with original assets | Avoid Blizzard IP/assets |
| Behavior trees / RL / IL **inside the simulator** | No official client |
| Post-game raid-log analysis (e.g. public log platforms’ permitted uses) | Follow each platform’s ToS |
| Local VLM annotation of recorded frames | Offline |
| Human subjects study: show recommendations while user plays manually | User presses all keys |

### 4.2 Accessibility / assistive tools (player remains in control)

| Tool type | Boundary |
|-----------|----------|
| On-screen rotation **suggestion** overlay fed by vision of a recording | Safer if offline; live overlay that does not send inputs is lower risk than bots but may still be scrutinized if it “facilitates gameplay” via unauthorized third-party software — prefer Blizzard-native assists where possible |
| Blizzard Assisted Highlight / One-Button Rotation / Boss Warnings / Combat Audio Alerts | First-party; preferred assistive path |
| Colorblind filters, TTS, remapping within OS/accessibility APIs without gameplay automation | Generally assistive; do not add auto-cast |
| Addon UI rearrangements within sandbox | Must follow addon policy; Midnight limits combat computation |

**Hard rule for this repo:** assistive prototypes may **display** advice; they must **not** generate synthetic keystrokes/mouse events into the official client.

### 4.3 Features likely prohibited on official servers

| Feature | Why |
|---------|-----|
| Unattended grinding / questing / raiding | Automated character control (bot) |
| Automated combat rotation execution | Automated control |
| Auto-targeting, auto-loot, auto-pathing bots | Automated control |
| Memory reading / pixel bots that click for the user | Bot + often hack-adjacent |
| Process injection, DLL hooks, packet editing | Hacks / unauthorized modification |
| Anti-cheat bypass, hiding tools from Warden-like systems | Explicit evasion — out of scope and prohibited |
| Input broadcasting / multibox streamlining | Official prohibition |
| Selling botting or automation services | Create/offer/distribute bots; commercial cheating language in EULA |

---

## 5. Addon API capabilities vs limits (research-relevant)

### Can access / do (typical, sandbox)

- Create frames; display information (including binding secret values to widgets in Midnight model)
- Out-of-combat configuration of secure frames/attributes
- SavedVariables persistence
- Many unit existence / friendship / role style queries (non-secret categories evolve by patch)

### Cannot / must not rely on for an external agent

- Calling protected functions (cast, target, move, use action, etc.) without real hardware events; blocked in combat lockdown
- Synthetic “fake” hardware events to satisfy secure handlers
- (Midnight+) freely reading/branching on many combat secrets for decision addons
- Network/filesystem access outside SavedVariables / allowed addon messaging
- Exporting live combat state to an external program that then auto-plays (designs an unofficial bot pipeline)

### Combat lockdown (enduring model)

Protected actions and secure-frame mutation are restricted during combat so that **human hardware input** remains central. This is a deliberate anti-automation architecture inside the official client.

---

## 6. Safer alternatives (encouraged directions)

1. **Combat coaching** on prerecorded video  
2. **Post-game analysis** vs combat logs  
3. **Rotation recommendations** shown to a human (no auto input)  
4. **Raid-log analysis** and timeline review tools  
5. **Accessibility overlays** that do not act for the player; prefer first-party Blizzard accessibility features when available  
6. **Training in a custom simulator** with original mechanics/art  
7. **AI controlling a legally owned custom game environment** you create or license  

---

## 7. Engineering compliance checklist (required for contributors)

Before merging any code that touches capture, input, or networking:

- [ ] Does it send keyboard/mouse/controller events to WoW? → **Reject**
- [ ] Does it read WoW process memory or inject code? → **Reject**
- [ ] Does it parse/modify network packets? → **Reject**
- [ ] Does it discuss or implement anti-cheat evasion? → **Reject**
- [ ] Is it offline video / sim / recommendation-only? → **Allow under review**
- [ ] Does an addon remain inside Blizzard’s sandbox and addon policy? → **Allow under review**
- [ ] Are datasets consented and free of unauthorized Blizzard asset redistribution? → **Required**

---

## 8. Distinction reminder

| | Technical | Policy |
|--|-----------|--------|
| Vision-based full auto agent | Partially feasible | **Prohibited** on official servers |
| Offline recommender | Feasible | **Aligned** with this repo’s safe scope |
| Simulator agent | Feasible | **Aligned** if no Blizzard client derivative |
| Addon that displays CDs | Feasible within API | **Permitted** if policy-compliant; Midnight limits computation |
| Addon/external hybrid that auto-casts | Feasible in abstract | **Prohibited** |

---

## 9. Re-audit triggers

Re-read official EULA + WoW policy posts when:

- Expanding from offline → any live client adjacency
- Shipping an addon
- Monetizing any tool
- Midnight/API patches change secret-value rules
- Considering partnerships or “authorized” research access (obtain written authorization)
