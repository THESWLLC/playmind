"""Combat and targeting skills (tab → approach → engage → rotate → loot)."""

from __future__ import annotations

from playmind.skills.base import Skill, SkillContext, SkillStepResult


class AcquireTargetSkill(Skill):
    name = "acquire_target"
    timeout_s = 4.0
    retry_limit = 4

    def can_start(self, ctx: SkillContext) -> bool:
        return ctx.alive and not ctx.modal_menu and not ctx.has_target

    def step(self, ctx: SkillContext) -> SkillStepResult:
        self._steps += 1
        if ctx.has_target:
            self._done = True
            return self._result(
                "wait",
                "acquire_target:have_target",
                status="success",
                success_evidence=["has_target"],
            )
        if not ctx.alive:
            self._mark_failed("not_alive")
            return self._result(
                "wait",
                "acquire_target:not_alive",
                status="failed",
                failure_evidence=["dead_or_ghost"],
            )
        if self.timed_out(ctx):
            self._mark_failed("timeout")
            return self._result(
                "key:tab",
                "acquire_target:timeout",
                status="timeout",
                timed_out=True,
                failure_evidence=["no_target"],
            )
        return self._result("key:tab", "acquire_target:tab", status="running")

    def allowed_actions(self) -> list[str]:
        return ["key:tab", "target_nearest", "wait"]


class ValidateTargetSkill(Skill):
    """Confirm the current target looks real (bar / combat / not suspect)."""

    name = "validate_target"
    timeout_s = 2.0

    def can_start(self, ctx: SkillContext) -> bool:
        return ctx.alive and ctx.has_target

    def step(self, ctx: SkillContext) -> SkillStepResult:
        self._steps += 1
        if not ctx.has_target:
            self._mark_failed("lost_target")
            return self._result(
                "wait",
                "validate_target:lost",
                status="failed",
                failure_evidence=["no_target"],
            )
        suspect = bool(ctx.obs.get("target_suspect")) or int(ctx.obs.get("no_damage_casts") or 0) >= 3
        if suspect:
            self._mark_failed("suspect_target")
            return self._result(
                "key:tab",
                "validate_target:suspect",
                status="failed",
                failure_evidence=["target_suspect"],
            )
        thp = ctx.target_hp
        evidence = ["has_target"]
        if thp is not None:
            evidence.append(f"target_hp={thp:.2f}")
        if ctx.in_combat or (thp is not None and thp > 0):
            self._done = True
            return self._result(
                "wait",
                "validate_target:ok",
                status="success",
                success_evidence=evidence,
            )
        # Soft accept: having a target bar is enough for scripted path.
        self._done = True
        return self._result(
            "wait",
            "validate_target:accepted",
            status="success",
            success_evidence=evidence,
        )

    def allowed_actions(self) -> list[str]:
        return ["key:tab", "wait"]


class ApproachTargetSkill(Skill):
    name = "approach_target"
    timeout_s = 5.0

    def can_start(self, ctx: SkillContext) -> bool:
        return ctx.alive and ctx.has_target and not ctx.modal_menu

    def step(self, ctx: SkillContext) -> SkillStepResult:
        self._steps += 1
        if not ctx.has_target:
            self._mark_failed("lost_target")
            return self._result(
                "wait",
                "approach:lost_target",
                status="failed",
                failure_evidence=["no_target"],
            )
        motion = float(ctx.obs.get("motion") or 0)
        # After a few approach steps, consider range closed enough to engage.
        if self._steps >= 2 or motion >= 3.0 or ctx.in_combat:
            self._done = True
            return self._result(
                "hold:w:0.6",
                "approach:close_enough",
                status="success",
                success_evidence=["approached"],
            )
        if self.timed_out(ctx):
            self._mark_failed("timeout")
            return self._result(
                "hold:w:0.8",
                "approach:timeout",
                status="timeout",
                timed_out=True,
            )
        return self._result("hold:w:0.8", "approach:gap_close", status="running")

    def allowed_actions(self) -> list[str]:
        return ["hold:w:0.6", "hold:w:0.8", "hold:w:1.1", "wait"]


class EngageTargetSkill(Skill):
    name = "engage_target"
    timeout_s = 3.0

    def can_start(self, ctx: SkillContext) -> bool:
        return ctx.alive and ctx.has_target and not ctx.modal_menu

    def step(self, ctx: SkillContext) -> SkillStepResult:
        self._steps += 1
        if not ctx.has_target:
            self._mark_failed("no_target")
            return self._result(
                "wait",
                "engage:no_target",
                status="failed",
                failure_evidence=["no_target"],
            )
        if ctx.in_combat or self._steps >= 2:
            self._done = True
            return self._result(
                "key:1",
                "engage:opened",
                status="success",
                success_evidence=["combat_opened"],
            )
        if self.timed_out(ctx):
            self._mark_failed("timeout")
            return self._result("key:1", "engage:timeout", status="timeout", timed_out=True)
        return self._result("key:1", "engage:open_with_1", status="running")

    def allowed_actions(self) -> list[str]:
        return ["key:1", "attack", "wait"]


class BasicCombatRotationSkill(Skill):
    name = "basic_combat_rotation"
    timeout_s = 20.0

    _ROTATION = ("key:1", "key:1", "key:2", "key:1", "key:3")

    def can_start(self, ctx: SkillContext) -> bool:
        return ctx.alive and ctx.has_target and not ctx.modal_menu

    def step(self, ctx: SkillContext) -> SkillStepResult:
        self._steps += 1
        thp = ctx.target_hp
        if thp is not None and thp <= 0.02:
            self._done = True
            return self._result(
                "wait",
                "combat:target_dead",
                status="success",
                success_evidence=["target_hp_zero"],
            )
        if not ctx.has_target and self._steps > 1:
            # Target drop after casts often means kill.
            self._done = True
            return self._result(
                "wait",
                "combat:target_lost_after_casts",
                status="success",
                success_evidence=["target_cleared"],
            )
        if not ctx.has_target:
            self._mark_failed("no_target")
            return self._result(
                "wait",
                "combat:no_target",
                status="failed",
                failure_evidence=["no_target"],
            )
        if self.timed_out(ctx):
            self._mark_failed("timeout")
            act = self._ROTATION[(self._steps - 1) % len(self._ROTATION)]
            return self._result(act, "combat:timeout", status="timeout", timed_out=True)
        act = self._ROTATION[(self._steps - 1) % len(self._ROTATION)]
        return self._result(act, f"combat:cast:{act}", status="running")

    def allowed_actions(self) -> list[str]:
        return ["key:1", "key:2", "key:3", "key:4", "key:5", "attack", "wait"]


class LootTargetSkill(Skill):
    name = "loot_target"
    timeout_s = 4.0

    def can_start(self, ctx: SkillContext) -> bool:
        if not ctx.alive or ctx.modal_menu:
            return False
        thp = ctx.target_hp
        # Loot when target is dead (hp ~0) or we just lost a combat target.
        if thp is not None and thp <= 0.05:
            return True
        if ctx.has_target and thp is not None and thp > 0.05:
            return False
        return bool(ctx.obs.get("loot_available")) or (
            not ctx.has_target and bool(ctx.obs.get("recent_kill"))
        )

    def step(self, ctx: SkillContext) -> SkillStepResult:
        self._steps += 1
        thp = ctx.target_hp
        if ctx.has_target and thp is not None and thp > 0.05:
            self._mark_failed("target_alive")
            return self._result(
                "wait",
                "loot:target_still_alive",
                status="failed",
                failure_evidence=["target_alive"],
            )
        if self._steps >= 2 or self.timed_out(ctx):
            self._done = True
            status = "timeout" if self.timed_out(ctx) and self._steps < 2 else "success"
            if status == "timeout":
                self._mark_failed("timeout")
            return self._result(
                "loot",
                "loot:done",
                status=status,
                timed_out=status == "timeout",
                success_evidence=["loot_attempted"] if status == "success" else [],
            )
        return self._result("loot", "loot:attempt", status="running")

    def allowed_actions(self) -> list[str]:
        return ["loot", "interact", "wait"]


class DisengageSkill(Skill):
    name = "disengage"
    timeout_s = 4.0

    def can_start(self, ctx: SkillContext) -> bool:
        return ctx.alive and (ctx.in_combat or ctx.has_target or ctx.hp < 0.4)

    def step(self, ctx: SkillContext) -> SkillStepResult:
        self._steps += 1
        if self._steps >= 3 or (not ctx.in_combat and self._steps >= 2):
            self._done = True
            return self._result(
                "hold:s:1.2",
                "disengage:done",
                status="success",
                success_evidence=["backed_off"],
            )
        if self.timed_out(ctx):
            self._mark_failed("timeout")
            return self._result("hold:s:1.0", "disengage:timeout", status="timeout", timed_out=True)
        return self._result("hold:s:1.2", "disengage:flee", status="running")

    def allowed_actions(self) -> list[str]:
        return ["hold:s:1.0", "hold:s:1.2", "move_south", "wait"]
