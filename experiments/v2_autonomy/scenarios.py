"""Small, reproducible v2 scenes; only world events are scripted.

NERVA's perception, memory, attention and motion decisions run closed-loop in
``experiments.reactive.scenario.run``.  The schedules below create a ball
encounter and two different observable interaction histories before an
identical later social meeting.
"""

from __future__ import annotations

from experiments.reactive.scenario import Agent


def curiosity_repeat_scenario() -> dict[str, Agent]:
    """A ball appears, is observed long enough to habituate, then returns."""
    return {
        "ball": Agent([
            (0.0, "hide", ()),
            (4.0, "appear", (1.3, 0.5)), (4.0, "wait", ()),
            (36.0, "hide", ()),
            (42.0, "appear", (1.3, 0.5)), (42.0, "wait", ()),
        ]),
        "person": Agent([(0.0, "hide", ())]),
    }


def social_history_scenario(history: str) -> dict[str, Agent]:
    """One visually identical person after positive or cautious evidence.

    ``positive`` contains measured gentle touches.  ``cautious`` contains a
    rendered-camera near miss (fast closing at short range), never a physical
    push.  Both end with the same person returning at 38 s and calmly
    approaching to 1.0 m.
    """
    if history not in ("positive", "cautious"):
        raise ValueError("history must be 'positive' or 'cautious'")
    if history == "positive":
        history_steps = [
            (4.0, "appear", (3.0, 0.15)), (4.0, "approach", (0.25, 0.9)),
            (13.0, "pet", (0.3,)), (23.0, "leave", (0.5,)), (30.0, "hide", ()),
        ]
    else:
        history_steps = [
            (4.0, "appear", (3.0, 0.15)), (4.0, "approach", (0.25, 1.4)),
            (12.0, "leave", (0.8,)), (13.5, "wait", ()),
            (15.0, "approach", (1.8, 0.45)), (17.0, "wait", ()),
            (22.0, "leave", (0.5,)), (30.0, "hide", ()),
        ]
    test_meeting = [
        (38.0, "appear", (3.0, 0.15)), (38.0, "approach", (0.25, 1.0)), (52.0, "wait", ()),
    ]
    return {"person": Agent([(0.0, "hide", ()), *history_steps, *test_meeting])}
