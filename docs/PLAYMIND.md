# PlayMind overview

## Scope

PlayMind is an **owned-game** agent:
- Demo world included for offline development
- Learning + teach-mode built in
- Adapters later for your client / Parsec / vision

It is **not** a World of Warcraft bot.

## Components

| Module | Role |
|--------|------|
| `demo_world.py` | Grid quest/combat environment |
| `planner.py` | Heuristic planner + optional Ollama |
| `learning.py` | Experience buffer + epsilon-greedy Q policy |
| `agent.py` | Combines plan/learn/teach |
| `playmind_onefile.py` | Single-file shareable demo |

## Learning

1. Agent acts in the demo  
2. Rewards update tabular Q-values **automatically**  
3. Experience is logged  
4. Successful/teacher rows export to `finetune.jsonl` for later LLM fine-tunes  

Teach mode: when unsure, ask the human; store answer as a boosted label.

## Next adapters (not in MVP)

- Screen capture of **your** game window  
- OCR quest text from pixels  
- Keyboard emitter (local or Parsec)  
- Session scheduler (breaks/logout) for **your** game only  
