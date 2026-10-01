"""Explicit goal state for the reactive experiments (refactor stage E).

Before stage E, goals were implicit in the behaviour rules. They are now a small explicit set that
appraisal can ask "which active goal could this outcome affect?". Not a planner.

  standing goals (always active in the reactive scenarios):
    remain_upright   1.0
    keep_distance    0.5   (preferred social distance; raised to 0.9 toward a feared person)
    explore          0.4
  from the behaviour currently executed:
    approach / inspect  → approach / inspect with that target (0.7)
    watch / retreat / freeze → retreat from that target (0.9) and keep_distance from it (0.9)
Priorities are NERVA design choices.
"""

from __future__ import annotations

from nerva.interfaces import Goal, GoalState

STANDING = (Goal("remain_upright", priority=1.0), Goal("keep_distance", priority=0.5), Goal("explore", priority=0.4))


def active_goals(mode: str = "", target: str | None = None, standing: tuple[Goal, ...] = STANDING) -> GoalState:
    goals = list(standing)
    tgt = target or ""
    if mode in ("approach", "inspect"):
        goals.append(Goal(mode, tgt, 0.7))
    elif mode in ("watch", "retreat", "freeze"):
        goals += [Goal("retreat", tgt, 0.9), Goal("keep_distance", tgt, 0.9)]
    return GoalState(tuple(goals))
