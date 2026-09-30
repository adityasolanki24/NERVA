"""Appraisal layer: what does an event mean for the robot?

EMA-inspired (Marsella & Gratch 2009): events are not mapped to emotions
directly. Each event is first appraised along EMA's appraisal variables; the
affect layer derives emotions from the appraisal.

The appraisal VALUES below are NERVA design choices for synthetic events, not
measurements or psychology. Reasoning per event: docs/affect_model.md §1.
Replace this table (or `appraise`) to change how events are interpreted.

This is appraisal v0: a context-free lookup. It is NOT the target design. Long term,
appraisal = f(perception, goals, self/physical state, expectations, history,
available actions), so the same event can mean different things in different
contexts. Keep callers depending only on `appraise(Event) -> AppraisalState`.
"""

from __future__ import annotations

import dataclasses

from nerva.interfaces import AppraisalState, Event

EVENT_APPRAISALS: dict[str, AppraisalState] = {
    # Routine walking progress: expected and low-stakes. v0.1: relevance 0.5 → 0.2 so routine
    # success does not dominate affect (a significant success/recovery would be a separate event).
    "successful_walking": AppraisalState(
        relevance=0.2, desirability=0.4, likelihood=1.0, expectedness=0.9, controllability=0.9),
    "near_fall": AppraisalState(
        relevance=1.0, desirability=-0.8, likelihood=0.5, expectedness=0.1, controllability=0.3),
    "person_approaching_slowly": AppraisalState(
        relevance=0.4, desirability=0.2, likelihood=0.6, expectedness=0.7, controllability=0.7),
    "person_approaching_rapidly": AppraisalState(
        relevance=0.9, desirability=-0.6, likelihood=0.6, expectedness=0.2, controllability=0.3),
    "obstacle_blocking_goal": AppraisalState(
        relevance=0.8, desirability=-0.5, likelihood=1.0, expectedness=0.5, controllability=0.6),
}


def appraise(event: Event, table: dict[str, AppraisalState] = EVENT_APPRAISALS) -> AppraisalState:
    """Look up the event's appraisal; `event.magnitude` scales its desirability."""
    try:
        base = table[event.kind]
    except KeyError:
        raise ValueError(f"no appraisal defined for event {event.kind!r}; known: {sorted(table)}") from None
    return dataclasses.replace(base, desirability=base.desirability * event.magnitude)


# ── Appraisal v1: context-aware (docs/reactive_behaviour_design.md §3) ────────
#
# The same event kind is appraised from the current tracks and from memory:
#   - novelty habituates with exposure: novelty = exp(-exposure_s / HABITUATION_S[kind])
#   - a fast approach is worse and less controllable the closer and faster it is
#   - a threat in the recent past (a fast approach or near fall within THREAT_MEMORY_S) turns an
#     otherwise friendly slow approach or closeness into a threat, and a threat leaving into relief
#   - repeated harmless fast approaches become more expected (surprise and fear habituate)
# All numbers are NERVA design choices for simulated events, not psychology.

import math  # noqa: E402

from nerva.interfaces import Track  # noqa: E402

HABITUATION_S = {"person": 30.0, "ball": 30.0}
THREAT_MEMORY_S = 45.0
IN_VIEW_PERIOD_S = 2.0  # re-appraise a visible stimulus this often ("still looking at it")


class ContextualAppraiser:
    def __init__(self):
        self.exposure_s: dict[str, float] = {}
        self.fast_approaches = 0
        self.last_threat_t = -math.inf
        self._last_in_view: dict[str, float] = {}

    def novelty(self, kind: str) -> float:
        return math.exp(-self.exposure_s.get(kind, 0.0) / HABITUATION_S.get(kind, 20.0))

    def threat_recent(self, t: float) -> bool:
        return t - self.last_threat_t < THREAT_MEMORY_S

    def observe(self, t: float, tracks: tuple[Track, ...], dt: float) -> list[tuple[str, AppraisalState]]:
        """Accumulate exposure; every IN_VIEW_PERIOD_S appraise each visible stimulus ("<kind>_in_view")."""
        out = []
        for tr in tracks:
            if not tr.visible:
                continue
            self.exposure_s[tr.kind] = self.exposure_s.get(tr.kind, 0.0) + dt
            if t - self._last_in_view.get(tr.kind, -math.inf) >= IN_VIEW_PERIOD_S:
                self._last_in_view[tr.kind] = t
                n = self.novelty(tr.kind)
                # desirability 0: looking elicits no joy/hope/fear by itself; expectedness ≥ 0.3: no surprise
                out.append((f"{tr.kind}_in_view", AppraisalState(
                    relevance=0.2 + 0.4 * n, desirability=0.0, likelihood=0.5,
                    expectedness=max(0.3, 1.0 - n), controllability=0.8)))
        return out

    def appraise(self, event: Event, t: float, tracks: tuple[Track, ...]) -> AppraisalState | None:
        """Appraise one perception/proprioception event in context; None = not relevant."""
        track = {tr.kind: tr for tr in tracks}
        kind = event.kind
        if kind in ("person_appeared", "ball_appeared"):
            who = kind.split("_")[0]
            n = self.novelty(who)
            threat = who == "person" and self.threat_recent(t)
            return AppraisalState(relevance=0.4 + 0.3 * n, desirability=-0.3 if threat else 0.1,
                                  likelihood=0.6, expectedness=1.0 - 0.8 * n,
                                  controllability=0.5 if threat else 0.8)
        if kind == "person_approaching_rapidly":
            tr = track.get("person")
            dist = tr.distance if tr else 1.5
            expected = min(0.6, 0.15 + 0.15 * self.fast_approaches)  # habituation to harmless lunges
            self.fast_approaches += 1
            self.last_threat_t = t
            return AppraisalState(relevance=0.9, desirability=-min(1.0, 0.4 + 0.4 * event.magnitude + 0.3 / max(dist, 0.3)),
                                  likelihood=0.7, expectedness=expected,
                                  controllability=float(min(0.7, max(0.1, dist / 2.5))))
        if kind == "person_approaching_slowly":
            if self.threat_recent(t):
                return AppraisalState(relevance=0.6, desirability=-0.4, likelihood=0.6,
                                      expectedness=0.5, controllability=0.5)
            return AppraisalState(relevance=0.5, desirability=0.3, likelihood=0.6, expectedness=0.6,
                                  controllability=0.7)
        if kind == "person_close":
            if self.threat_recent(t) and t - self.last_threat_t < 5.0:
                return AppraisalState(relevance=0.8, desirability=-0.7, likelihood=0.8, expectedness=0.4,
                                      controllability=0.2)
            return AppraisalState(relevance=0.4, desirability=0.2, likelihood=0.7, expectedness=0.7,
                                  controllability=0.7)
        if kind == "ball_close":
            return None  # reaching the ball is the end of an approach, handled by behaviour
        if kind == "person_lost":
            if self.threat_recent(t):  # the threat went away: relief
                return AppraisalState(relevance=0.6, desirability=0.4, likelihood=1.0, expectedness=0.6,
                                      controllability=0.7)
            return None
        if kind == "ball_lost":
            return None
        if kind == "near_fall":
            self.last_threat_t = t
            return EVENT_APPRAISALS["near_fall"]
        return appraise(event)
