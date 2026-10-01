"""Behaviour layer v0 for the style-conditioned policy S1: PAD → (what, how).

    AFFECT ──PADState──▶ S1Behaviour ──BehaviourCommand(vx, style_vector=e)──▶ S1 policy

WHAT (the task) and HOW (the style) stay separate; PAD never reaches joints.

HOW, e = (tempo, step height, torso pitch)   [NERVA design choice, NOT validated]
  tempo       e1 = clip(TEMPO_GAIN · arousal)
  step height e2 = 0 always: S1 does not express it (docs/development_log.md, 2026-09-30)
  torso pitch e3 = clip(−LEAN_GAIN · (valence + dominance) / 2)     (+ = forward lean)

Why these directions: studies of emotional gait report faster, higher-energy walking
with high-arousal emotions and slower walking with a flexed, head-down posture for
sadness (e.g. Montepare et al. 1987; Roether et al. 2009) [literature informs the
DIRECTION only]. The gains are chosen so that the PAD range affect v0.1 produces in
the demo scenario (|V|, |D| ≲ 0.4, A ≲ 0.35) spans S1's trained range; they are not
fitted to data. Whether observers read the result as intended is untested (human
evaluation, roadmap stage 15).

WHAT (demo task): walk forward at WALK_VX; stop while the goal is blocked.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nerva.interfaces import BehaviourCommand, PADState, StyleVector

WALK_VX = 0.13  # m/s, inside the trained command range
TEMPO_GAIN = 3.0
LEAN_GAIN = 3.0


@dataclass(frozen=True)
class BehaviourDecision:
    command: BehaviourCommand
    reason: str


def pad_to_style_vector(pad: PADState, tempo_gain: float = TEMPO_GAIN,
                        lean_gain: float = LEAN_GAIN) -> StyleVector:
    tempo = float(np.clip(tempo_gain * pad.arousal, -1.0, 1.0))
    lean = float(np.clip(-lean_gain * 0.5 * (pad.valence + pad.dominance), -1.0, 1.0))
    return StyleVector(tempo=tempo, step_height=0.0, torso_pitch=lean)


def s1_behaviour(pad: PADState, goal_blocked: bool) -> BehaviourDecision:
    e = pad_to_style_vector(pad)
    if goal_blocked:
        return BehaviourDecision(BehaviourCommand(vx=0.0, style_vector=e), "stop: goal blocked")
    return BehaviourDecision(BehaviourCommand(vx=WALK_VX, style_vector=e), "walk forward")
