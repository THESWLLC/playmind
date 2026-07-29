"""PlayMind agent: planner + optional self-learning + teach mode + vision."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from playmind.actuators import Actuator, DemoActuator
from playmind.demo_world import ACTIONS, DemoWorld
from playmind.learning import ExperienceBuffer, OnlinePolicy
from playmind.planner import HeuristicPlanner, OllamaPlanner, Planner
from playmind.vision import VisionReading, read_frame, save_demo_ascii_frame


@dataclass
class AgentConfig:
    use_ollama: bool = False
    ollama_model: str = "dolphin-llama3"
    learn: bool = True
    # When False (default), planner acts while learning still trains in the background.
    use_learned_policy: bool = False
    teach_mode: bool = False
    use_vision: bool = False
    vision_frame_path: Path = Path("data/playmind/frames/latest.txt")
    ask_every_n_uncertain: int = 1
    data_dir: Path = Path("data/playmind")


@dataclass
class PlayMindAgent:
    world: DemoWorld = field(default_factory=DemoWorld)
    config: AgentConfig = field(default_factory=AgentConfig)
    directive: str | None = None
    pending_question: str | None = None
    last_vision: VisionReading | None = None
    actuator: Actuator = field(default_factory=DemoActuator)
    _planner: Planner = field(init=False)
    policy: OnlinePolicy = field(default_factory=OnlinePolicy)
    buffer: ExperienceBuffer = field(init=False)

    def __post_init__(self) -> None:
        if self.config.use_ollama:
            self._planner = OllamaPlanner(model=self.config.ollama_model)
        else:
            self._planner = HeuristicPlanner()
        self.config.data_dir.mkdir(parents=True, exist_ok=True)
        self.buffer = ExperienceBuffer(self.config.data_dir / "experience.jsonl")
        policy_path = self.config.data_dir / "policy.json"
        self.policy.load(policy_path)

    def set_directive(self, text: str) -> None:
        self.directive = text.strip() or None

    def observe(self) -> dict[str, Any]:
        obs = self.world.observe()
        if self.config.use_vision:
            # Emit an ascii frame the vision layer can parse (stand-in for screenshots).
            quest = obs.get("quest_text") or (
                f"Kill {obs['quest_kills_needed']} Wolves "
                f"({obs['quest_kills']}/{obs['quest_kills_needed']}). Talk to Mira."
            )
            save_demo_ascii_frame(self.world.render_ascii(), quest, self.config.vision_frame_path)
            reading = read_frame(self.config.vision_frame_path)
            self.last_vision = reading
            obs.update(reading.to_obs_patch())
            # If quest log closed, still allow vision-derived quest text for planning.
            if not obs.get("quest_text") and reading.quest_text:
                obs["quest_text"] = reading.quest_text
        return obs

    def propose_action(self, obs: dict[str, Any] | None = None) -> str:
        obs = obs or self.observe()
        if self.config.use_learned_policy and self.config.learn:
            return self.policy.choose(obs, list(ACTIONS))
        return self._planner.plan(obs, self.directive)

    def maybe_ask(self, obs: dict[str, Any], action: str) -> str | None:
        if not self.config.teach_mode:
            return None
        uncertain = (
            not obs.get("adjacent_enemies")
            and obs.get("quest_kills", 0) < obs.get("quest_kills_needed", 0)
            and obs.get("steps", 0) % max(1, self.config.ask_every_n_uncertain * 7) == 0
        )
        if uncertain:
            self.pending_question = (
                f"I'm thinking `{action}`. Better action? "
                f"Options: {', '.join(ACTIONS)} (or Enter to accept)"
            )
            return self.pending_question
        return None

    def answer_teach(self, human_action: str, obs: dict[str, Any]) -> None:
        action = human_action.strip().replace("-", "_")
        if action not in ACTIONS:
            return
        self.policy.teach(obs, action, boost=1.5)
        self.buffer.add(obs, action, reward=0.5, next_obs=obs, done=False, source="teacher")
        self.pending_question = None

    def tick(self, action: str | None = None) -> dict[str, Any]:
        obs = self.observe()
        chosen = action or self.propose_action(obs)
        self.actuator.send(chosen)
        next_obs, reward, done, info = self.world.step(chosen)
        if self.config.learn:
            self.policy.update(obs, chosen, reward, next_obs, done, list(ACTIONS))
            self.buffer.add(obs, chosen, reward, next_obs, done, source="self")
        return {
            "action": chosen,
            "reward": reward,
            "done": done,
            "info": info,
            "obs": next_obs,
            "vision": self.last_vision,
        }

    def save(self) -> None:
        self.policy.save(self.config.data_dir / "policy.json")
        self.buffer.save()
        n = self.buffer.export_finetune_jsonl(self.config.data_dir / "finetune.jsonl")
        (self.config.data_dir / "last_export_count.txt").write_text(str(n), encoding="utf-8")
