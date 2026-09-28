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
