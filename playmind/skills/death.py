"""Death and ghost recovery skills (release → confirm → rez → runback)."""

from __future__ import annotations

from playmind.skills.base import Skill, SkillContext, SkillStepResult


def _life_phase(ctx: SkillContext) -> str:
    phase = str(ctx.obs.get("life_phase") or "")
    if phase:
        return phase
    if ctx.is_ghost:
        return "ghost"
    if ctx.is_dead:
        blob = ctx.ocr_blob
        if "are you sure" in blob or ctx.confirm_pending:
            return "confirm"
        if "choose where" in blob or "closest town" in blob:
            return "rez_picker"
        return "dead_dialog"
    return "alive"


class DeathRecoverySkill(Skill):
    """Advance death UI: Release Spirit → Yes → Closest Town / Graveyard."""

    name = "death_recovery"
    timeout_s = 25.0

    def can_start(self, ctx: SkillContext) -> bool:
        phase = _life_phase(ctx)
        return phase in {"dead_dialog", "confirm", "rez_picker"} or ctx.is_dead

    def step(self, ctx: SkillContext) -> SkillStepResult:
        self._steps += 1
        phase = _life_phase(ctx)
        if phase == "ghost" or (ctx.is_ghost and not ctx.is_dead):
            self._done = True
            return self._result(
                "wait",
                "death_recovery:became_ghost",
                status="success",
                success_evidence=["ghost"],
            )
        if phase == "alive" and not ctx.is_dead:
            self._done = True
            return self._result(
                "wait",
                "death_recovery:alive",
                status="success",
                success_evidence=["alive"],
            )

        if phase == "confirm":
            act = "click_label:Yes"
            reason = "death_recovery:confirm_yes"
        elif phase == "rez_picker":
            act = "click_label:Closest Town"
            reason = "death_recovery:closest_town"
        else:
            # dead_dialog
            blob = ctx.ocr_blob
            if "release spirit" in blob:
                act = "click_label:Release Spirit"
            elif "safe zone" in blob or "resurrect" in blob:
                act = "click_label:Resurrect in a Safe Zone"
            else:
                act = "click_label:Return to Graveyard"
            reason = f"death_recovery:{act}"

        if self.timed_out(ctx):
            self._mark_failed("timeout")
            return self._result(act, "death_recovery:timeout", status="timeout", timed_out=True)
        return self._result(act, reason, status="running")

    def allowed_actions(self) -> list[str]:
        return [
            "release_spirit",
            "click_label:Release Spirit",
            "click_label:Return to Graveyard",
            "click_label:Resurrect in a Safe Zone",
            "click_label:Yes",
            "click_label:yes",
            "click_label:Closest Town",
            "click_label:Closest City",
            "key:enter",
            "wait",
        ]


class GhostRunbackSkill(Skill):
    name = "ghost_runback"
    timeout_s = 40.0

    def can_start(self, ctx: SkillContext) -> bool:
        return ctx.is_ghost or _life_phase(ctx) == "ghost"

    def step(self, ctx: SkillContext) -> SkillStepResult:
        self._steps += 1
        if ctx.alive and not ctx.is_ghost:
            self._done = True
            return self._result(
                "wait",
                "ghost_runback:rezzed",
                status="success",
                success_evidence=["alive_again"],
            )
        blob = ctx.ocr_blob
        if "spirit healer" in blob:
            act = "interact"
            reason = "ghost_runback:spirit_healer"
        elif self._steps % 6 == 0:
            act = "hold:d:0.4"
            reason = "ghost_runback:course_correct"
        else:
            act = "hold:w:1.0"
            reason = "ghost_runback:toward_corpse"
        if self.timed_out(ctx):
            self._mark_failed("timeout")
            return self._result(act, "ghost_runback:timeout", status="timeout", timed_out=True)
        return self._result(act, reason, status="running")

    def allowed_actions(self) -> list[str]:
        return ["hold:w:1.0", "hold:w:1.2", "hold:a:0.4", "hold:d:0.4", "interact", "wait"]
