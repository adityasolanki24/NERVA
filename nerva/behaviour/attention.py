"""Target selection for NERVA's visual attention.

This module sits between perception and the existing behaviour selector.  It is
deliberately small and designed rather than learned: it scores *measured*
tracks using novelty, relative motion, recent unexpected events, visibility
uncertainty and an optional current goal.  The score is only a reason to look
or inspect; it never changes the safety supervisor or locomotion envelope.

The manager has a dwell time and a switch margin.  That makes a target persist
through a short detector dropout and prevents the head/behaviour controller
from flickering between similarly interesting objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from nerva.interfaces import Event, GoalState, Track

MIN_FIXATION_S = 1.2
SWITCH_MARGIN = 0.18
MIN_INTEREST = 0.20
MOTION_FULL_M_S = 0.7


@dataclass(frozen=True)
class AttentionDecision:
    """Current target and the auditable components that selected it."""

    target: str | None
    reason: str
    score: float = 0.0
    scores: dict[str, float] = field(default_factory=dict)


@dataclass
class AttentionManager:
    """Choose one visible target from robot-camera tracks.

    ``novelty`` is supplied by entity memory/appraisal and must be in [0, 1].
    Track IDs remain temporary perceptual identities; this class never treats
    a track ID as a persistent person identity.
    """

    target: str | None = None
    target_since: float = 0.0
    last_decision: AttentionDecision = field(default_factory=lambda: AttentionDecision(None, "explore"))

    @staticmethod
    def _unexpected(events: tuple[Event, ...] | list[Event], tid: str) -> float:
        kinds = {ev.kind for ev in events if ev.source == tid}
        if any(k.endswith("_appeared") or "rapidly" in k or k.endswith("_close") for k in kinds):
            return 1.0
        if any("approaching" in k for k in kinds):
            return 0.55
        return 0.0

    @staticmethod
    def _goal_bonus(goals: GoalState | None, tid: str) -> float:
        if goals is None:
            return 0.0
        return max((goal.priority for goal in goals.goals if goal.target == tid), default=0.0)

    def update(self, t: float, tracks: tuple[Track, ...], events: tuple[Event, ...] | list[Event] = (),
               novelty: dict[str, float] | None = None, goals: GoalState | None = None) -> AttentionDecision:
        """Update attention from perception only and return the selected target.

        Invisible tracks are retained only while the tracker considers them
        recent, with a score penalty.  A target is dropped once it is no longer
        tracked, rather than inventing object permanence beyond that bounded
        tracker memory.
        """
        novelty = novelty or {}
        by_id = {tr.tid or tr.kind: tr for tr in tracks}
        scores: dict[str, float] = {}
        reasons: dict[str, str] = {}
        for tid, tr in by_id.items():
            n = min(1.0, max(0.0, float(novelty.get(tid, 1.0))))
            motion = min(1.0, abs(float(tr.approach_speed)) / MOTION_FULL_M_S)
            unexpected = self._unexpected(events, tid)
            uncertainty = min(1.0, max(0.0, float(tr.unseen_for_s) / MIN_FIXATION_S))
            visible = 1.0 if tr.visible else 0.35
            goal = self._goal_bonus(goals, tid)
            score = visible * (0.52 * n + 0.24 * motion + 0.30 * unexpected + 0.10 * uncertainty + 0.16 * goal)
            scores[tid] = round(score, 6)
            signals = []
            if unexpected:
                signals.append("unexpected")
            if n >= 0.5:
                signals.append("novel")
            if motion >= 0.25:
                signals.append("moving")
            if goal:
                signals.append("goal")
            reasons[tid] = "+".join(signals) or "familiar"

        if self.target not in by_id:
            self.target = None
        candidate = max(scores, key=scores.get) if scores else None
        current_score = scores.get(self.target, 0.0)
        candidate_score = scores.get(candidate, 0.0)
        urgent = candidate is not None and self._unexpected(events, candidate) >= 1.0
        can_switch = self.target is None or t - self.target_since >= MIN_FIXATION_S
        should_switch = candidate is not None and candidate_score >= MIN_INTEREST and (
            self.target is None or (can_switch and candidate != self.target and candidate_score > current_score + SWITCH_MARGIN)
            or (urgent and candidate != self.target and candidate_score > current_score)
        )
        if should_switch:
            self.target, self.target_since = candidate, t

        if self.target is None:
            decision = AttentionDecision(None, "explore", 0.0, scores)
        else:
            decision = AttentionDecision(self.target, reasons[self.target], scores[self.target], scores)
        self.last_decision = decision
        return decision
