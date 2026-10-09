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


class AttractorAffectModel(DimensionalAffectModel):
    """Model B v2 (`affect_model="Bv2"`): appraisal sets a bounded PAD *target*; PAD relaxes toward it.

    Fixes Model B's saturation (development log 2026-10-02): re-appraising an unchanged, ongoing
    situation (the persistent `*_in_view` frames, every 2 s) used to add a new impulse each time, so PAD
    grew with the re-appraisal rate rather than with the situation.

      contextual  one slot per source track holds the features of its LATEST persistent appraisal
                  (replaced, not added); full weight while it keeps being re-appraised (age since the last
                  refresh ≤ HOLD_S, longer than the 2 s re-appraisal period), then fading as
                  exp(−(age − HOLD_S) / TAU_CONTEXT_S); so the re-appraisal rate does not change the input
      phasic      discrete events (non-persistent appraisals) add decaying traces, as in Model B
      target      x* = x0 + tanh(G·W·(u_context + z_phasic)),  G = diag(τ_V, τ_A, τ_D)
                  (G converts Model B's per-second weights to the same steady-state gain; W is unchanged)
      dynamics    dx/dt = −Λ (x − x*); with no input x* = x0, so PAD returns to baseline

    Bounded by construction (|tanh| < 1, no clipping needed while x0 = 0). Repeating an identical
    persistent appraisal converges, at any re-appraisal rate. Action tendencies are computed exactly as in
    Model B (unchanged, so behaviour sees the same tendency signal).
    """

    TAU_CONTEXT_S = 3.0
    HOLD_S = 2.5  # > IN_VIEW_PERIOD_S (2 s) of the appraisers

    def __init__(self, cfg: AffectBConfig | None = None, tau_context_s: float | None = None):
        super().__init__(cfg)
        self.tau_ctx = tau_context_s or self.TAU_CONTEXT_S
        self.t = 0.0
        self.context: dict[str, tuple[np.ndarray, float]] = {}  # source -> (features, time of last refresh)
        self.z_phasic = np.zeros(len(FEATURES))
        self._gain = np.array(self.cfg.tau_pad_s, dtype=float)

    def add(self, appraisal: AppraisalState | AppraisalFrame, source: str = "") -> np.ndarray:
        persistent = isinstance(appraisal, AppraisalFrame) and appraisal.persistent
        if isinstance(appraisal, AppraisalFrame):
            source = source or appraisal.hypothesis.subject
        u = super().add(appraisal, source)  # tendency traces, unchanged from Model B
        if persistent:
            self.context[source] = (u, self.t)
        else:
            self.z_phasic = self.z_phasic + u
        return u

    def _weight(self, t_last: float) -> float:
        age = self.t - t_last
        return 1.0 if age <= self.HOLD_S else math.exp(-(age - self.HOLD_S) / self.tau_ctx)

    def context_input(self) -> np.ndarray:
        total = np.zeros(len(FEATURES))
        for u, t_last in self.context.values():
            total += u * self._weight(t_last)
        return total

    def target(self) -> np.ndarray:
        x0 = np.array(self.cfg.baseline)
        return x0 + np.tanh(self._gain * (self._w @ (self.context_input() + self.z_phasic)))

    def step(self, dt: float) -> PADState:
        x_star = self.target()
        self.x = np.clip(x_star + (self.x - x_star) * np.exp(-self._lam * dt), -1.0, 1.0)
        self.t += dt
        k = np.exp(-dt / self._tau_z)
        self.z = self.z * k
        self.z_phasic = self.z_phasic * k
        self.z_by_source = {s: z * k for s, z in self.z_by_source.items() if float(np.max(np.abs(z))) > 1e-4}
        self.context = {s: (u, tl) for s, (u, tl) in self.context.items() if self._weight(tl) > 1e-3}
        return self.pad


class OnsetAttractorAffectModel(AttractorAffectModel):
    """Model B v3 (`affect_model="Bv3"`): Model B v2 with fast onset, slow return.

    Bv2 relaxed toward its target with one time constant per dimension (8 / 4 / 8 s), so a sudden threat
    could not move dominance within seconds (development log 2026-10-06). Here, per dimension, the rate is
    1/TAU_ONSET_S while the response is building (the target lies further from baseline than x, on the
    side x is moving to) and 1/τ (Bv2's) otherwise. Same structure as Model A: fast pull, slow return.
    """

    TAU_ONSET_S = 1.0

    def add(self, appraisal: AppraisalState | AppraisalFrame, source: str = "") -> np.ndarray:
        """As Bv2, but context slots are keyed by (source, hypothesis kind), so one person's in-view context
        and ongoing-contact context coexist (2026-10-06)."""
        persistent = isinstance(appraisal, AppraisalFrame) and appraisal.persistent
        key_kind = appraisal.hypothesis.kind if isinstance(appraisal, AppraisalFrame) else ""
        if isinstance(appraisal, AppraisalFrame):
            source = source or appraisal.hypothesis.subject
        u = DimensionalAffectModel.add(self, appraisal, source)
        if persistent:
            self.context[f"{source}|{key_kind}"] = (u, self.t)
        else:
            self.z_phasic = self.z_phasic + u
        return u

    def step(self, dt: float) -> PADState:
        x_star = self.target()
        x0 = np.array(self.cfg.baseline)
        building = (np.abs(x_star - x0) > np.abs(self.x - x0)) & (np.sign(x_star - x0) == np.sign(x_star - self.x))
        lam = np.where(building, 1.0 / self.TAU_ONSET_S, self._lam)
        self.x = np.clip(x_star + (self.x - x_star) * np.exp(-lam * dt), -1.0, 1.0)
        self.t += dt
        k = np.exp(-dt / self._tau_z)
        self.z = self.z * k
        self.z_phasic = self.z_phasic * k
        self.z_by_source = {s: z * k for s, z in self.z_by_source.items() if float(np.max(np.abs(z))) > 1e-4}
        self.context = {s: (u, tl) for s, (u, tl) in self.context.items() if self._weight(tl) > 1e-3}
        return self.pad


class FacetAttractorAffectModel(OnsetAttractorAffectModel):
    """Bv4: combine context facets per source (docs/context_facet_experiment.md).

    Each source contributes a fade-weighted mean, scaled by its freshest weight.
    Distinct sources still add; phasic input and tendency traces are unchanged.
    """

    def add(self, appraisal: AppraisalState | AppraisalFrame, source: str = "") -> np.ndarray:
        persistent = isinstance(appraisal, AppraisalFrame) and appraisal.persistent
        kind = appraisal.hypothesis.kind if isinstance(appraisal, AppraisalFrame) else ""
        if isinstance(appraisal, AppraisalFrame):
            source = source or appraisal.hypothesis.subject
        u = DimensionalAffectModel.add(self, appraisal, source)
        if persistent:
            self.context[(source, kind)] = (u, self.t)
        else:
            self.z_phasic = self.z_phasic + u
        return u

    def context_input(self) -> np.ndarray:
        grouped = {}
        for (source, _kind), (u, t_last) in self.context.items():
            weight = self._weight(t_last)
            numerator, denominator, freshest = grouped.get(source, (np.zeros(len(FEATURES)), 0.0, 0.0))
            grouped[source] = (numerator + weight * u, denominator + weight, max(freshest, weight))
        total = np.zeros(len(FEATURES))
        for numerator, denominator, freshest in grouped.values():
            total += freshest * numerator / denominator
        return total
