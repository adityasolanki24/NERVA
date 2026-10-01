"""Mapping from the expressive style variable to locomotion parameters.

Phase 5, method A: style changes only the rate of the walking policy's gait-phase
clock (Open Duck's `phase_frequency_factor`). No retraining.

    phase_factor = 1 + PHASE_GAIN * style        style ∈ [-1, 1]

With PHASE_GAIN = 0.3, the factor spans 0.7-1.3. That range was chosen because
an exploratory probe (development_log.md, 2026-09-28) showed stable walking for
factors 0.6-1.4 on flat ground. This is a design choice, not a model of emotion.
Factors ≠ 1.0 are outside what the policy saw in training.
"""

from nerva.interfaces import ExpressiveStyle

PHASE_GAIN = 0.3


def style_to_phase_factor(style: ExpressiveStyle, gain: float = PHASE_GAIN) -> float:
    return 1.0 + gain * style.style


# ── S1: style-conditioned policy (docs/style_policy_design.md) ──────────────

S1_NEUTRAL_PERIOD_S = 0.54
S1_TEMPO_GAIN = 0.25  # single_support_duration × (1 − 0.25·e1), cloud/generate_styled_references.py
REFERENCE_FPS = 50


def s1_nb_steps_in_period(e1: float) -> int:
    """Control steps per gait cycle for tempo e1, as upstream PolyReferenceMotion computes it.

    Measured on the R1 pilot (2026-09-29): the generator's period is 3 × single-support
    duration (0.405 / 0.54 / 0.675 s for e1 = +1 / 0 / −1), i.e. linear in e1. For e1
    between the trained values (±1, 0) this interpolates; the policy never saw them.
    """
    period = S1_NEUTRAL_PERIOD_S * (1.0 - S1_TEMPO_GAIN * e1)
    return int(period * REFERENCE_FPS)


S1_NEUTRAL_FOOT_HEIGHT_M = 0.04  # upstream placo preset medium.json walk_foot_height
S1_FOOT_HEIGHT_GAIN = 0.5  # walk_foot_height × (1 + 0.5·e2), cloud/generate_styled_references.py


def s1_foot_height(e2: float) -> float:
    """The walk_foot_height [m] the R1 references were generated with, for step-height e2."""
    return S1_NEUTRAL_FOOT_HEIGHT_M * (1.0 + S1_FOOT_HEIGHT_GAIN * e2)
