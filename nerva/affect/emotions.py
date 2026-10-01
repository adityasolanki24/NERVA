"""Affect model v0.1 ("Model A"): appraisal → discrete emotions → PAD anchors → persistent PAD.

This is ONE implementation of `nerva.interfaces.AffectSystem`, not a required
layer of NERVA: a future model may map appraisal to PAD directly. Sources and
design choices are documented in docs/affect_model.md. In short:

  categorise()   simplified EMA-inspired rules. Labels follow Marsella & Gratch
                 2009 (Table 2); base intensity |desirability × likelihood| follows
                 Gratch & Marsella 2004 (Table 3). EMA's mood-adjusted intensity,
                 appraisal frames and coping are NOT implemented.
                 NERVA extension: intensity is scaled by relevance.
  EMOTION_PAD    emotion → PAD anchor: ALMA (Gebhard 2005, Table 2). Surprise is a
                 NERVA choice: arousal only (0.8, from WASABI Table 1); it does not
                 take part in the valence/dominance averages.
  dynamics       ALMA-style: decaying emotions pull the PAD state toward their
                 intensity-weighted centre, computed per dimension; each dimension
                 relaxes to its baseline with its own time constant. All
                 parameters are NERVA engineering choices (AffectConfig).
  controllability → dominance blend: NERVA hypothesis.
  interest       v0.2 NERVA extension (curiosity; EMA/ALMA define none): elicited by a novel
                 (expectedness < 0.5), non-harmful (desirability ≥ 0) appraisal; intensity
                 relevance × (1 − expectedness). Anchor: positive valence, raised arousal,
                 slight dominance, the "interested/alert" region of the affect circumplex
                 (qualitative placement; the numbers are ours). Events without novelty elicit
                 exactly what v0.1 did.

This is an engineered internal state, not a validated model of human emotion.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from nerva.interfaces import ActionTendencyState, AppraisalFrame, AppraisalState, PADState

NA = float("nan")  # "this emotion does not act on this PAD dimension"

# (valence/pleasure, arousal, dominance); NA = dimension not affected
EMOTION_PAD: dict[str, tuple[float, float, float]] = {
    "joy": (0.40, 0.20, 0.10),  # ALMA Table 2
    "hope": (0.20, 0.20, -0.10),  # ALMA Table 2
    "fear": (-0.64, 0.60, -0.43),  # ALMA Table 2
    "distress": (-0.40, -0.20, -0.50),  # ALMA Table 2 (EMA 2009 label: "sadness")
    "surprise": (NA, 0.80, NA),  # arousal from WASABI Table 1; V/D excluded (NERVA v0.1)
    "interest": (0.30, 0.40, 0.10),  # NERVA v0.2 choice, see module docstring
}


@dataclass(frozen=True)
class AffectConfig:
    """Every tunable number of the affect model. All are NERVA choices unless noted."""

    surprise_expectedness_threshold: float = 0.3  # EMA: surprise when expectedness is "low"
    interest_expectedness_threshold: float = 0.5  # NERVA v0.2: novelty needed for interest
    scale_by_relevance: bool = True  # NERVA extension of EMA (v0.1)
    controllability_weight: float = 0.5  # w in D = (1-w)·D_emotion + w·(2c-1)
    tau_emotion_s: float = 4.0  # default emotion intensity decay time constant
    tau_emotion_overrides_s: dict[str, float] = field(default_factory=lambda: {"surprise": 1.0})
    k_pull_per_s: float = 1.0  # attraction rate toward the emotion centre at intensity 1
    tau_return_s: tuple[float, float, float] = (20.0, 6.0, 20.0)  # V, A, D relaxation to baseline
    min_intensity: float = 0.01  # emotions below this are dropped
    baseline: tuple[float, float, float] = (0.0, 0.0, 0.0)
    emotion_pad: dict[str, tuple[float, float, float]] = field(default_factory=lambda: dict(EMOTION_PAD))


DEFAULT_CONFIG = AffectConfig()


@dataclass
class EmotionInstance:
    label: str
    intensity: float  # [0, 1], decays over time
    pad: np.ndarray  # (3,) anchor after the controllability adjustment; NaN = dimension not affected
    tau_s: float  # this instance's decay time constant


def categorise(a: AppraisalState, cfg: AffectConfig | None = None) -> list[tuple[str, float]]:
    """Simplified EMA-inspired rules: appraisal → [(emotion label, intensity)]."""
    cfg = cfg or DEFAULT_CONFIG
    if a.relevance == 0.0:
        return []
    scale = a.relevance if cfg.scale_by_relevance else 1.0
    out = []
    base = abs(a.desirability * a.likelihood)  # Gratch & Marsella 2004, Table 3
    if a.desirability > 0:
        out.append(("joy" if a.likelihood >= 1.0 else "hope", scale * base))
    elif a.desirability < 0:
        out.append(("distress" if a.likelihood >= 1.0 else "fear", scale * base))
    if a.expectedness < cfg.surprise_expectedness_threshold:
        out.append(("surprise", scale * (1.0 - a.expectedness)))  # intensity rule: NERVA
    if a.desirability >= 0 and a.expectedness < cfg.interest_expectedness_threshold:
        out.append(("interest", scale * (1.0 - a.expectedness)))  # NERVA v0.2
    return [(label, i) for label, i in out if i > 0.0]


def emotion_pad_point(label: str, controllability: float, cfg: AffectConfig | None = None) -> np.ndarray:
    """Emotion's PAD anchor; dominance (if the emotion has one) is blended toward controllability."""
    cfg = cfg or DEFAULT_CONFIG
    p, a, d = cfg.emotion_pad[label]
    w = cfg.controllability_weight
    if not math.isnan(d):
        d = (1.0 - w) * d + w * (2.0 * controllability - 1.0)
    return np.array([p, a, d])


class CategoricalAffectModel:
    """Affect model v0.1 (Model A). Implements nerva.interfaces.AffectSystem.

    Per time step:  model.add(appraisal)  (zero or more times),  then  model.step(dt).
    """

    def __init__(self, cfg: AffectConfig | None = None):
        self.cfg = cfg = cfg or DEFAULT_CONFIG
        self.x = np.array(cfg.baseline, dtype=float)
        self.emotions: list[EmotionInstance] = []

    @property
    def pad(self) -> PADState:
        return PADState(*(float(v) for v in self.x))

    def intensities(self) -> dict[str, float]:
        """Summed intensity of the active emotions, per label (for logging and the tendency adapter)."""
        out: dict[str, float] = {}
        for e in self.emotions:
            out[e.label] = out.get(e.label, 0.0) + e.intensity
        return out

    @property
    def tendencies(self) -> ActionTendencyState:
        """Action tendencies from the active emotions (nerva/affect/tendencies.py)."""
        from nerva.affect.tendencies import tendencies_from_emotions

        return tendencies_from_emotions(self.intensities())

    def add(self, appraisal: AppraisalState | AppraisalFrame) -> list[EmotionInstance]:
        """Elicit the emotions an appraisal produces; returns the new instances."""
        if isinstance(appraisal, AppraisalFrame):
            appraisal = appraisal.as_appraisal_state()
        c = self.cfg
        new = [EmotionInstance(label, min(i, 1.0), emotion_pad_point(label, appraisal.controllability, c),
                               c.tau_emotion_overrides_s.get(label, c.tau_emotion_s))
               for label, i in categorise(appraisal, c)]
        self.emotions.extend(new)
        return new

    def emotion_centre(self) -> tuple[np.ndarray, np.ndarray]:
        """ALMA 'virtual emotion center', per dimension.

        For each PAD dimension, over the emotions that act on it: intensity-weighted
        mean anchor, and mean intensity. Dimensions no active emotion acts on get
        centre 0 and intensity 0 (no pull).
        """
        centre, strength = np.zeros(3), np.zeros(3)
        if not self.emotions:
            return centre, strength
        w = np.array([e.intensity for e in self.emotions])
        pts = np.array([e.pad for e in self.emotions])
        for k in range(3):
            acts = ~np.isnan(pts[:, k])
            if acts.any():
                centre[k] = (w[acts] * pts[acts, k]).sum() / w[acts].sum()
                strength[k] = w[acts].mean()
        return centre, strength

    def step(self, dt: float) -> PADState:
        """Advance dt seconds. Exact integration per dimension with centre/strength held over the step."""
        c = self.cfg
        centre, strength = self.emotion_centre()
        a = c.k_pull_per_s * strength  # pull toward the emotion centre
        b = 1.0 / np.array(c.tau_return_s)  # return to baseline, per dimension
        target = (a * centre + b * np.array(c.baseline)) / (a + b)
        self.x = np.clip(target + (self.x - target) * np.exp(-(a + b) * dt), -1.0, 1.0)

        for e in self.emotions:
            e.intensity *= math.exp(-dt / e.tau_s)
        self.emotions = [e for e in self.emotions if e.intensity >= c.min_intensity]
        return self.pad
