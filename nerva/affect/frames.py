"""Appraisal frames: every appraisal is about an explicit outcome hypothesis (refactor stage E).

Wraps an existing appraiser (v1 `ContextualAppraiser` or v2 `MemoryAppraiser`) and turns each of its
`AppraisalState`s into an `AppraisalFrame`:

1. **Hypothesis.** Each event kind is mapped to an explicit proposition (HYPOTHESES), e.g. a rapid
   approach → "collision" (would be confirmed by a `contact_impact` outcome; bears on remain_upright and
   keep_distance). The appraiser's likelihood becomes that hypothesis's probability, so "likelihood of
   what?" always has an answer. An event kind without a mapping is an error, not a silent default.
   Where an event's meaning depends on context (a person seen while remembered as a threat, or not),
   the sign of the contextual desirability selects the hypothesis.
2. **Goals.** relevance × NO_GOAL_RELEVANCE when none of the hypothesis's goals is active.
3. **Self state.** For physically adverse hypotheses, a high stability risk lowers controllability
   (× (1 − SELF_CONTROL_LOSS · risk)) and raises relevance (+ SELF_RELEVANCE · risk · (1 − relevance)):
   a collision matters more and is less controllable when the robot is already unsteady.

With every hypothesis's goals active and stability risk 0, the frame reproduces the wrapped appraiser's
numbers exactly (tested). All mappings and constants are NERVA design choices.
"""

from __future__ import annotations

import dataclasses

from nerva.interfaces import AppraisalFrame, AppraisalState, Event, GoalState, OutcomeHypothesis, SelfState

NO_GOAL_RELEVANCE = 0.3
TTC_SAFE_S = 3.0  # reaction margin at which an approaching agent no longer reduces control
C_MIN, C_MAX = 0.1, 0.8
SELF_CONTROL_LOSS = 0.5
SELF_RELEVANCE = 0.5

# hypothesis kind → (predicted outcome kinds, goal kinds it bears on, physically adverse?)
HYPOTHESIS_KINDS = {
    "collision": (("contact_impact",), ("remain_upright", "keep_distance", "retreat"), True),
    "adverse_interaction": (("contact_impact",), ("keep_distance", "retreat"), True),
    "benign_interaction": (("benign_contact",), ("approach", "keep_distance", "explore"), False),
    "threat_recedes": ((), ("keep_distance", "retreat"), False),
    "novel_stimulus": ((), ("explore", "inspect"), False),
    "stability_loss": (("stability_loss",), ("remain_upright",), True),
    "locomotion_progress": ((), ("follow_command", "remain_upright"), False),
    "goal_blocked": ((), ("follow_command",), False),
}


def _social(a: AppraisalState) -> str:
    return "adverse_interaction" if a.desirability < 0 else "benign_interaction"


# event kind → hypothesis kind (str) or a function of the contextual appraisal
HYPOTHESES = {
    "person_approaching_rapidly": "collision",
    "person_close": lambda a: "collision" if a.desirability < 0 else "benign_interaction",
    "person_appeared": _social,
    "person_approaching_slowly": _social,
    "person_in_view": _social,
    "touch_gentle": "benign_interaction",
    "person_lost": "threat_recedes",
    "ball_appeared": "novel_stimulus",
    "ball_in_view": "novel_stimulus",
    "near_fall": "stability_loss",
    "successful_walking": "locomotion_progress",
    "obstacle_blocking_goal": "goal_blocked",
}


def hypothesis_kind(event_kind: str, appraisal: AppraisalState) -> str:
    try:
        h = HYPOTHESES[event_kind]
    except KeyError:
        raise ValueError(f"no outcome hypothesis defined for event {event_kind!r}") from None
    return h(appraisal) if callable(h) else h


def frame_from(event_kind: str, a: AppraisalState, subject: str, self_state: SelfState | None,
               goals: GoalState | None, persistent: bool = False) -> AppraisalFrame:
    kind = hypothesis_kind(event_kind, a)
    predicts, goal_kinds, adverse = HYPOTHESIS_KINDS[kind]
    relevance, controllability = a.relevance, a.controllability
    active = goal_kinds if goals is None else tuple(g for g in goal_kinds if goals.priority((g,)) > 0)
    if goals is not None and not active:
        relevance *= NO_GOAL_RELEVANCE
    risk = self_state.stability_risk if self_state is not None else 0.0
    if adverse and risk > 0:
        controllability *= 1.0 - SELF_CONTROL_LOSS * risk
        relevance += SELF_RELEVANCE * risk * (1.0 - relevance)
    hyp = OutcomeHypothesis(kind, subject=subject, probability=a.likelihood, predicts=predicts)
    return AppraisalFrame(hyp, relevance=relevance, desirability=a.desirability, expectedness=a.expectedness,
                          controllability=controllability, goals=active, persistent=persistent)


def agent_controllability(track, self_state: SelfState | None) -> float:
    """Controllability of an interaction with an autonomous agent = the robot's reaction margin:
    clip(TTC / TTC_SAFE_S, C_MIN, C_MAX), TTC = distance / closing speed (inf when not closing), lowered by
    the robot's own instability (development log 2026-10-06; NERVA design)."""
    closing = track.approach_speed
    ttc = track.distance / closing if closing > 1e-6 else float("inf")
    c = min(C_MAX, max(C_MIN, ttc / TTC_SAFE_S))
    risk = self_state.stability_risk if self_state is not None else 0.0
    return c * (1.0 - 0.5 * risk)


class FrameAppraiser:
    """Same calls as the wrapped appraiser, but returns AppraisalFrames and takes self state and goals.
    Other attributes (identity, novelty(), memory, ...) are those of the wrapped appraiser."""

    def __init__(self, inner):
        object.__setattr__(self, "inner", inner)

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def __setattr__(self, name, value):  # e.g. appraiser.identity = ...: state lives in the wrapped appraiser
        setattr(self.inner, name, value)

    def appraise(self, event: Event, t: float, tracks, self_state: SelfState | None = None,
                 goals: GoalState | None = None) -> AppraisalFrame | None:
        a = self.inner.appraise(event, t, tracks)
        if a is None:
            return None
        return frame_from(event.kind, a, event.source, self_state, goals)

    def observe(self, t: float, tracks, dt: float, self_state: SelfState | None = None,
                goals: GoalState | None = None) -> list[tuple[str, AppraisalFrame]]:
        # in-view appraisals re-evaluate an ongoing situation: marked persistent. For a PERSON, controllability
        # is at most the reaction margin (agent_controllability); objects do not act and keep theirs.
        by_tid = {tr.tid: tr for tr in tracks}
        out = []
        for kind, a, subject in self.inner.observe_with_subjects(t, tracks, dt):
            tr = by_tid.get(subject)
            if tr is not None and tr.kind == "person":
                a = dataclasses.replace(a, controllability=min(a.controllability, agent_controllability(tr, self_state)))
            out.append((kind, frame_from(kind, a, subject, self_state, goals, persistent=True)))
        return out
