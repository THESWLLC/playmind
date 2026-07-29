"""Recovery, exploration, modal, and utility skills."""

from __future__ import annotations

from playmind.skills.base import Skill, SkillContext, SkillStepResult


class RecoverHealthSkill(Skill):
    name = "recover_health"
    timeout_s = 8.0

    def can_start(self, ctx: SkillContext) -> bool:
        return ctx.alive and ctx.hp < 0.45 and not ctx.modal_menu

    def step(self, ctx: SkillContext) -> SkillStepResult:
        self._steps += 1
        if ctx.hp >= 0.55:
            self._done = True
            return self._result(
                "wait",
                "recover:hp_ok",
                status="success",
                success_evidence=[f"hp={ctx.hp:.2f}"],
            )
        if ctx.in_combat and ctx.hp < 0.35:
            # Prefer backing off while recovering under fire.
            act = "hold:s:0.8"
        elif self._steps % 3 == 0:
            act = "key:5"  # common heal/consumable bind placeholder
        else:
            act = "wait"
        if self.timed_out(ctx):
            self._mark_failed("timeout")
            return self._result(act, "recover:timeout", status="timeout", timed_out=True)
        return self._result(act, f"recover:{act}", status="running")

    def allowed_actions(self) -> list[str]:
        return ["wait", "key:5", "hold:s:0.8", "hold:s:1.0"]


class ExploreSkill(Skill):
    name = "explore"
    timeout_s = 6.0

    def can_start(self, ctx: SkillContext) -> bool:
        return ctx.alive and not ctx.modal_menu and not ctx.confirm_pending

    def step(self, ctx: SkillContext) -> SkillStepResult:
        self._steps += 1
        if ctx.has_target or ctx.obs.get("hostiles_near"):
            self._done = True
            return self._result(
                "wait",
                "explore:found_content",
                status="success",
                success_evidence=["hostiles_or_target"],
            )
        motion = float(ctx.obs.get("motion") or 0)
        if self._steps >= 4 and motion >= 2.0:
            self._done = True
            return self._result(
                "hold:w:1.1",
                "explore:moved",
                status="success",
                success_evidence=["motion"],
            )
        if self.timed_out(ctx):
            self._mark_failed("timeout")
            return self._result("hold:w:1.1", "explore:timeout", status="timeout", timed_out=True)
        # Periodically turn to cover more ground.
        if self._steps % 5 == 0:
            return self._result("hold:d:0.6", "explore:turn", status="running")
        return self._result("hold:w:1.1", "explore:forward", status="running")

    def allowed_actions(self) -> list[str]:
        return [
            "hold:w:1.1",
            "hold:w:0.6",
            "hold:d:0.6",
            "hold:a:0.6",
            "move_north",
            "move_east",
            "wait",
        ]


class UnstuckSkill(Skill):
    name = "unstuck"
    timeout_s = 5.0

    _SEQ = ("hold:s:0.8", "hold:d:1.1", "hold:a:1.1", "hold:w:1.2", "key:tab")

    def can_start(self, ctx: SkillContext) -> bool:
        if not ctx.alive or ctx.modal_menu:
            return False
        return bool(ctx.obs.get("stuck_hint")) or int(ctx.obs.get("stagnant") or 0) >= 4

    def step(self, ctx: SkillContext) -> SkillStepResult:
        self._steps += 1
        act = self._SEQ[(self._steps - 1) % len(self._SEQ)]
        motion = float(ctx.obs.get("motion") or 0)
        if motion >= 4.0 and self._steps >= 2:
            self._done = True
            return self._result(
                act,
                "unstuck:freed",
                status="success",
                success_evidence=["motion"],
            )
        if self.timed_out(ctx):
            self._mark_failed("timeout")
            return self._result(act, "unstuck:timeout", status="timeout", timed_out=True)
        return self._result(act, f"unstuck:{act}", status="running")

    def allowed_actions(self) -> list[str]:
        return list(self._SEQ) + ["wait"]


class ClearModalSkill(Skill):
    name = "clear_modal"
    timeout_s = 4.0

    def can_start(self, ctx: SkillContext) -> bool:
        if ctx.confirm_pending or ctx.is_dead or ctx.is_ghost:
            # Death confirm is owned by DeathRecoverySkill.
            return False
        blob = ctx.ocr_blob
        return bool(ctx.modal_menu) or any(
            m in blob
            for m in ("options", "exit game", "key bindings", "macros", "addons", "logout")
        )

    def step(self, ctx: SkillContext) -> SkillStepResult:
        self._steps += 1
        blob = ctx.ocr_blob
        if not ctx.modal_menu and not any(
            m in blob for m in ("options", "exit game", "key bindings", "macros")
        ):
            self._done = True
            return self._result(
                "wait",
                "clear_modal:cleared",
                status="success",
                success_evidence=["modal_gone"],
            )
        act = "click_label:Close" if "close" in blob else "key:esc"
        if self.timed_out(ctx):
            self._mark_failed("timeout")
            return self._result(act, "clear_modal:timeout", status="timeout", timed_out=True)
        return self._result(act, f"clear_modal:{act}", status="running")

    def allowed_actions(self) -> list[str]:
        return ["key:esc", "key:escape", "click_label:Close", "wait"]


class InteractSkill(Skill):
    name = "interact"
    timeout_s = 3.0

    def can_start(self, ctx: SkillContext) -> bool:
        return ctx.alive and not ctx.modal_menu

    def step(self, ctx: SkillContext) -> SkillStepResult:
        self._steps += 1
        blob = ctx.ocr_blob
        if any(w in blob for w in ("accept", "continue", "complete", "quest")):
            act = "click_label:Accept"
        else:
            act = "interact"
        if self._steps >= 2:
            self._done = True
            return self._result(
                act,
                "interact:done",
                status="success",
                success_evidence=["interacted"],
            )
        if self.timed_out(ctx):
            self._mark_failed("timeout")
            return self._result(act, "interact:timeout", status="timeout", timed_out=True)
        return self._result(act, f"interact:{act}", status="running")

    def allowed_actions(self) -> list[str]:
        return ["interact", "click_label:Accept", "click_label:Continue", "wait"]


class WaitSkill(Skill):
    name = "wait"
    timeout_s = 2.0

    def can_start(self, ctx: SkillContext) -> bool:
        return True

    def step(self, ctx: SkillContext) -> SkillStepResult:
        self._steps += 1
        if self._steps >= 1:
            self._done = True
            return self._result("wait", "wait:tick", status="success", success_evidence=["waited"])
        return self._result("wait", "wait:running", status="running")

    def allowed_actions(self) -> list[str]:
        return ["wait"]
