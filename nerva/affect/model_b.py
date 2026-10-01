"""Affect Model B: appraisal → PAD and action tendencies directly, without discrete emotion categories.

Refactor stage G (docs/architecture.md; RQ8 in docs/research_questions.md: are the categories necessary
for the behaviours we test?). Same external contract as Model A (`AffectSystem`): `add(appraisal)`,
`step(dt)`, `pad`, `tendencies`.

Appraisal features, per added appraisal (r relevance, d desirability, l likelihood, e expectedness,
c controllability; d⁺ = max(d, 0), d⁻ = max(−d, 0)):

  u = [ pos = r·d⁺·l,  neg = r·d⁻·l,  unexpected = r·(1 − e)²,  novelty = r·(1 − e)·(1 − d⁻),  control = r·(2c − 1) ]

Each feature is added to a drive trace z that decays with tau_drive_s (unexpected: tau_unexpected_s).
PAD is a transparent linear leaky system on the drive:

  dx/dt = −Λ (x − x0) + W z,      Λ = diag(1/τ_V, 1/τ_A, 1/τ_D),   x clipped to [−1, 1]

integrated exactly over each step with z held constant. Tendencies are continuous functions of the
same traces (no thresholds, unlike Model A's categories):

  approach = pos,  explore = novelty,  avoid = neg,  orient = unexpected,
  withdraw = neg · (1 − c̄)  (harm the robot cannot control → disengage; c̄ = control trace mapped to [0, 1]),
  freeze   = orient · avoid

W, Λ and the time constants are NERVA design parameters (AffectBConfig). They were set in one
calibration pass so that two reference appraisals (a strong rapid approach and a gentle touch) give PAD
peaks of the same order as Model A (development log, 2026-10-02); nothing was tuned on the scenarios
used to compare the models, and nothing is fitted to human data. This is not a model of human emotion.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from nerva.interfaces import MAX_TENDENCY, ActionTendencyState, AppraisalFrame, AppraisalState, PADState

FEATURES = ("pos", "neg", "unexpected", "novelty", "control")


@dataclass(frozen=True)
class AffectBConfig:
    tau_drive_s: float = 4.0
    tau_unexpected_s: float = 1.0
    tau_pad_s: tuple[float, float, float] = (8.0, 4.0, 8.0)  # V, A, D relaxation to baseline
    baseline: tuple[float, float, float] = (0.0, 0.0, 0.0)
    # rows V, A, D; columns pos, neg, unexpected, novelty, control (per second, per unit drive)
    w: tuple[tuple[float, ...], ...] = field(default_factory=lambda: (
        (0.25, -0.47, 0.00, 0.08, 0.00),
        (0.05, 0.25, 0.40, 0.10, 0.00),
        (0.00, -0.10, 0.00, 0.00, 0.22),
    ))


DEFAULT_B = AffectBConfig()


def appraisal_features(a: AppraisalState) -> np.ndarray:
    dpos, dneg = max(a.desirability, 0.0), max(-a.desirability, 0.0)
    r = a.relevance
    return np.array([r * dpos * a.likelihood, r * dneg * a.likelihood, r * (1.0 - a.expectedness) ** 2,
                     r * (1.0 - a.expectedness) * (1.0 - dneg), r * (2.0 * a.controllability - 1.0)])


class DimensionalAffectModel:
    """Model B. Implements nerva.interfaces.AffectSystem."""

    def __init__(self, cfg: AffectBConfig | None = None):
        self.cfg = cfg = cfg or DEFAULT_B
        self.x = np.array(cfg.baseline, dtype=float)
        self.z = np.zeros(len(FEATURES))
        self.z_by_source: dict[str, np.ndarray] = {}  # the same traces, split by the track they are about
        self._w = np.array(cfg.w, dtype=float)
        self._lam = 1.0 / np.array(cfg.tau_pad_s, dtype=float)
        self._tau_z = np.array([cfg.tau_drive_s, cfg.tau_drive_s, cfg.tau_unexpected_s, cfg.tau_drive_s,
                                cfg.tau_drive_s])

    @property
    def pad(self) -> PADState:
        return PADState(*(float(v) for v in self.x))

    def add(self, appraisal: AppraisalState | AppraisalFrame, source: str = "") -> np.ndarray:
        if isinstance(appraisal, AppraisalFrame):
            source = source or appraisal.hypothesis.subject
            appraisal = appraisal.as_appraisal_state()
        u = appraisal_features(appraisal)
        self.z = self.z + u
        self.z_by_source[source] = self.z_by_source.get(source, np.zeros(len(FEATURES))) + u
        return u

    def step(self, dt: float) -> PADState:
        x0 = np.array(self.cfg.baseline)
        decay = np.exp(-self._lam * dt)
        drive = self._w @ self.z
        self.x = np.clip(x0 + decay * (self.x - x0) + (1.0 - decay) * drive / self._lam, -1.0, 1.0)
        k = np.exp(-dt / self._tau_z)
        self.z = self.z * k
        self.z_by_source = {s: z * k for s, z in self.z_by_source.items() if float(np.max(np.abs(z))) > 1e-4}
        return self.pad

    def intensities(self) -> dict[str, float]:
        """Drive traces by feature name (for logging; these are not emotion labels)."""
        return {k: float(v) for k, v in zip(FEATURES, self.z)}

    @property
    def tendencies(self) -> ActionTendencyState:
        return self._tendencies(self.z)

    def tendencies_for(self, source: str) -> ActionTendencyState:
        return self._tendencies(self.z_by_source.get(source, np.zeros(len(FEATURES))))

    @staticmethod
    def _tendencies(z) -> ActionTendencyState:
        pos, neg, unexpected, novelty, control = (float(v) for v in z)
        c_bar = 0.5 + 0.5 * math.tanh(control)  # control trace → [0, 1]

        def b(v):
            return min(max(v, 0.0), MAX_TENDENCY)

        return ActionTendencyState(approach=b(pos), explore=b(novelty), avoid=b(neg), orient=b(unexpected),
                                   freeze=b(unexpected * neg), withdraw=b(neg * (1.0 - c_bar)))
