"""Affect layer: appraisal → emotion instances → persistent PAD state.

Sources and design choices are documented in docs/affect_model.md. In short:
  - categorise():  EMA appraisal→emotion rules (Marsella & Gratch 2009, Table 2)
                   with EMA intensities |desirability × likelihood|
                   (Gratch & Marsella 2004, Table 3).
  - EMOTION_PAD:   emotion → PAD point from ALMA (Gebhard 2005, Table 2);
                   'surprise' from WASABI (Becker-Asano & Wachsmuth 2010, Table 1).
  - AffectModel:   ALMA-style dynamics — decaying emotions pull the PAD state
                   toward their intensity-weighted centre; the state relaxes to
                   a baseline. Parameters are NERVA choices (AffectConfig).
  - Controllability → dominance blend: NERVA hypothesis.

This is an engineering model, not a validated model of human emotion.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from nerva.interfaces import AppraisalState, PADState

# (pleasure/valence, arousal, dominance)
EMOTION_PAD: dict[str, tuple[float, float, float]] = {
    "joy": (0.40, 0.20, 0.10),  # ALMA Table 2
    "hope": (0.20, 0.20, -0.10),  # ALMA Table 2
    "fear": (-0.64, 0.60, -0.43),  # ALMA Table 2
    "distress": (-0.40, -0.20, -0.50),  # ALMA Table 2 (EMA 2009 label: "sadness")
    "surprise": (0.10, 0.80, 0.00),  # WASABI Table 1 (10, 80, ±100)/100; D sign unspecified → 0 (NERVA)
}


@dataclass(frozen=True)
class AffectConfig:
    """Every tunable number of the affect model. All are NERVA choices unless noted."""

    surprise_expectedness_threshold: float = 0.3  # EMA: surprise when expectedness is "low"
    controllability_weight: float = 0.5  # w in D = (1-w)·D_emotion + w·(2c-1)
    tau_emotion_s: float = 4.0  # emotion intensity decay time constant
    k_pull_per_s: float = 1.0  # attraction rate toward the emotion centre at intensity 1
    tau_return_s: float = 20.0  # relaxation time constant back to baseline
    min_intensity: float = 0.01  # emotions below this are dropped
    baseline: tuple[float, float, float] = (0.0, 0.0, 0.0)
    emotion_pad: dict[str, tuple[float, float, float]] = field(default_factory=lambda: dict(EMOTION_PAD))


@dataclass
class EmotionInstance:
    label: str
    intensity: float  # [0, 1], decays over time
    pad: np.ndarray  # (3,) PAD point after the controllability adjustment


def categorise(a: AppraisalState, cfg: AffectConfig = AffectConfig()) -> list[tuple[str, float]]:
    """EMA rules: appraisal → [(emotion label, intensity)]. Can yield zero, one or two emotions."""
    if a.relevance == 0.0:
        return []
    out = []
    intensity = abs(a.desirability * a.likelihood)
    if a.desirability > 0:
        out.append(("joy" if a.likelihood >= 1.0 else "hope", intensity))
    elif a.desirability < 0:
        out.append(("distress" if a.likelihood >= 1.0 else "fear", intensity))
    if a.expectedness < cfg.surprise_expectedness_threshold:
        out.append(("surprise", 1.0 - a.expectedness))
    return [(label, i) for label, i in out if i > 0.0]


def emotion_pad_point(label: str, controllability: float, cfg: AffectConfig = AffectConfig()) -> np.ndarray:
    """Emotion's PAD point with dominance blended toward appraised controllability (NERVA hypothesis)."""
    p, a, d = cfg.emotion_pad[label]
    w = cfg.controllability_weight
    return np.array([p, a, (1.0 - w) * d + w * (2.0 * controllability - 1.0)])


class AffectModel:
    """Persistent PAD state driven by appraised events.

    Usage per time step:  model.add(appraisal)  (zero or more times),  then  model.step(dt).
    """

    def __init__(self, cfg: AffectConfig = AffectConfig()):
        self.cfg = cfg
        self.x = np.array(cfg.baseline, dtype=float)
        self.emotions: list[EmotionInstance] = []

    @property
    def pad(self) -> PADState:
        return PADState(*(float(v) for v in self.x))

    def add(self, appraisal: AppraisalState) -> list[EmotionInstance]:
        """Elicit the emotions an appraisal produces; returns the new instances."""
        new = [EmotionInstance(label, min(i, 1.0), emotion_pad_point(label, appraisal.controllability, self.cfg))
               for label, i in categorise(appraisal, self.cfg)]
        self.emotions.extend(new)
        return new

    def emotion_centre(self) -> tuple[np.ndarray, float]:
        """ALMA 'virtual emotion center': intensity-weighted mean PAD point, and mean intensity."""
        if not self.emotions:
            return np.zeros(3), 0.0
        w = np.array([e.intensity for e in self.emotions])
        pts = np.array([e.pad for e in self.emotions])
        return (w[:, None] * pts).sum(0) / w.sum(), float(w.mean())

    def step(self, dt: float) -> PADState:
        """Advance dt seconds. Exact integration of the linear ODE with I, E held over the step."""
        c = self.cfg
        centre, intensity = self.emotion_centre()
        a = c.k_pull_per_s * intensity  # pull toward emotion centre
        b = 1.0 / c.tau_return_s  # return to baseline
        target = (a * centre + b * np.array(c.baseline)) / (a + b)
        self.x = np.clip(target + (self.x - target) * math.exp(-(a + b) * dt), -1.0, 1.0)

        decay = math.exp(-dt / c.tau_emotion_s)
        for e in self.emotions:
            e.intensity *= decay
        self.emotions = [e for e in self.emotions if e.intensity >= c.min_intensity]
        return self.pad
