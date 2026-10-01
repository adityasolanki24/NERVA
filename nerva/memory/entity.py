"""Entity memory (docs/memory_design.md, phase M1): who/what the robot knows and how it feels about them.

One record per entity (a person identity or an object), found again by appearance:
  identity      running mean of an appearance embedding (vision: hue histogram; hardware: a face
                embedding); a new entity is created when nothing matches (cosine < MATCH_COSINE)
  familiarity   exposure time, which slowly fades while the entity is absent → novelty = exp(-exposure / τ)
  association   somatic-marker-like expectations learned by prediction error
                  threat   T ← T + α·(fear − T)          warmth  W ← W + α·(valence − W)
                  trust    R ← R + α·(benign − R), benign = 0 on a negative surprise, 1 otherwise
                with α = ALPHA·(1 + arousal)·confidence: emotional events are learned faster (McGaugh),
                and learning is proportional to the error, so later gentle encounters gradually
                outweigh an early fright rather than erasing it (extinction as new learning)
  episodes      a short list of salient events with this entity (M2 turns this into the episodic store)
Threat and warmth also relax slowly toward neutral while the entity is absent (DRIFT_S).

Two learning modes (refactor stage D, docs/architecture.md §1.3):
  "legacy"    the above: associations learned from the emotions Model A elicited. Kept as the baseline.
              Known problem: a remembered threat re-elicits fear, which is learned as more threat.
  "grounded"  associations learned only from measured outcomes (OutcomeSignal, nerva/world/outcomes.py):
                adverse  A ← A + α·(m − A)   on an actual adverse outcome (contact_impact, stability_loss)
                risk     K ← K + α·(m − K)   on a RiskEstimate (collision_risk: a near miss), kept separate
                benign   B ← B + α·(m − B)   on benign_contact; also A, K ← · + α·EXTINCTION·(0 − ·)
              with α = OUTCOME_ALPHA·confidence (no arousal scaling: affect does not set the teaching
              signal). learn() then only records the episode (emotions kept as history).
              Sleep replay (replay=True) adds NO evidence: it only restores A and B toward the levels the
              last real outcomes set, undoing absence drift, never beyond (one lunge stays one lunge).
              threat = max(A, K) and warmth = B − threat are DERIVED summaries in this mode; trust is not
              defined in this mode and stays at its prior (surprise is not evidence of untrustworthiness).
All constants are NERVA design choices (docs/memory_design.md §3.3); nothing here is fitted to data.
Identity from modular sensor evidence (refactor stage F): `resolve_evidence(kind, [SensorEvidence, ...])`
combines whichever modalities are present in both the observation and the record, weighted by the
evidence confidence (cosine similarity per modality); missing modalities contribute nothing. Today only
"vision.appearance" exists (stored in `appearance`); other modalities (a face or voice embedding later)
are stored in `features` when a sensor provides them. `resolve(kind, appearance)` is the single-modality
case and gives exactly the same results as before.
Pure Python/NumPy, O(entities) per lookup: fits embedded hardware.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from nerva.interfaces import SensorEvidence

MATCH_COSINE = 0.85
NEW_IDENTITY_COSINE = 0.5  # between this and MATCH_COSINE an observation is ambiguous: no new identity
HABITUATION_S = {"person": 30.0, "ball": 30.0}
FORGET_EXPOSURE_S = 600.0  # familiarity fades with this time constant while absent
DRIFT_S = 900.0  # threat/warmth relax toward neutral with this time constant while absent
ALPHA = 0.5
APPEARANCE_RATE = 0.1  # running-mean rate of the identity embedding
MAX_EPISODES = 20
OUTCOME_ALPHA = 0.5
EXTINCTION = 0.5  # a benign outcome also counts this much as evidence against adverse expectations
LEARNING_MODES = ("legacy", "grounded")
APPEARANCE = "vision.appearance"


@dataclass
class EntityRecord:
    eid: str
    kind: str
    appearance: np.ndarray | None
    exposure_s: float = 0.0
    last_seen_t: float = -math.inf
    threat: float = 0.0
    warmth: float = 0.0
    trust: float = 0.5
    encounters: int = 0
    episodes: list = field(default_factory=list)
    adverse: float = 0.0  # grounded mode: expected adverse outcome from this entity, [0, 1]
    benign: float = 0.0  # grounded mode: expected benign outcome, [0, 1]
    outcomes: int = 0  # grounded mode: number of attributed outcomes
    adverse_learned: float = 0.0  # grounded mode: level set by the last real outcomes (ceiling for replay)
    benign_learned: float = 0.0
    risk: float = 0.0  # grounded mode: expected risk from estimated near misses (RiskEstimate), [0, 1]
    risk_learned: float = 0.0
    features: dict = field(default_factory=dict)  # other modalities: modality -> running-mean embedding

    def novelty(self) -> float:
        return math.exp(-self.exposure_s / HABITUATION_S.get(self.kind, 30.0))


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0


class EntityMemory:
    def __init__(self, learning: str = "legacy"):
        if learning not in LEARNING_MODES:
            raise ValueError(f"learning must be one of {LEARNING_MODES}")
        self.learning = learning
        self.records: dict[str, EntityRecord] = {}
        self._counter: dict[str, int] = {}

    # ── identity ────────────────────────────────────────────────────────────
    def resolve(self, kind: str, appearance: np.ndarray | None, t: float) -> EntityRecord | None:
        """Single-modality identity (vision appearance); see resolve_evidence."""
        evidence = [] if appearance is None else [SensorEvidence(APPEARANCE, tuple(float(v) for v in appearance), 1.0, t)]
        return self.resolve_evidence(kind, evidence, t)

    @staticmethod
    def _similarity(rec: EntityRecord, obs: dict) -> float:
        """Confidence-weighted mean cosine over the modalities both have. With nothing on either side the
        kind alone identifies it (1.0); with no modality in common nothing can be compared (0.0)."""
        stored = dict(rec.features)
        if rec.appearance is not None:
            stored[APPEARANCE] = rec.appearance
        if not obs and not stored:
            return 1.0
        common = [m for m in obs if m in stored]
        wsum = sum(obs[m][1] for m in common)
        if not common or wsum <= 0:
            return 0.0
        return sum(obs[m][1] * _cosine(obs[m][0], stored[m]) for m in common) / wsum

    def resolve_evidence(self, kind: str, evidence, t: float) -> EntityRecord | None:
        """The known entity this observation belongs to, a new one if it is clearly unlike all known
        entities, or None if ambiguous (e.g. a partial view). Absence effects are applied here."""
        obs = {e.modality: (np.asarray(e.feature, float), e.confidence) for e in evidence if e.feature is not None}
        best, best_sim, closest = None, MATCH_COSINE, 0.0
        for rec in self.records.values():
            if rec.kind != kind:
                continue
            sim = self._similarity(rec, obs)
            closest = max(closest, sim)
            if sim >= best_sim:
                best, best_sim = rec, sim
        if best is None and closest >= NEW_IDENTITY_COSINE:
            return None
        if best is None:
            n = self._counter.get(kind, 0)
            self._counter[kind] = n + 1
            best = EntityRecord(f"{kind}#{n}", kind, obs[APPEARANCE][0] if APPEARANCE in obs else None, encounters=1,
                                features={m: v for m, (v, _) in obs.items() if m != APPEARANCE})
            self.records[best.eid] = best
        else:
            self._apply_absence(best, t)
            for m, (v, _) in obs.items():
                if m == APPEARANCE:
                    if best.appearance is not None:
                        best.appearance = (1 - APPEARANCE_RATE) * best.appearance + APPEARANCE_RATE * v
                elif m in best.features:
                    best.features[m] = (1 - APPEARANCE_RATE) * best.features[m] + APPEARANCE_RATE * v
                else:
                    best.features[m] = v
        return best

    def _apply_absence(self, rec: EntityRecord, t: float) -> None:
        gap = t - rec.last_seen_t
        if math.isfinite(gap) and gap > 1.0:
            rec.exposure_s *= math.exp(-gap / FORGET_EXPOSURE_S)
            drift = math.exp(-gap / DRIFT_S)
            rec.threat *= drift
            rec.warmth *= drift
            rec.adverse *= drift
            rec.benign *= drift
            rec.risk *= drift
            if self.learning == "grounded":
                self._derive(rec)
            rec.encounters += 1

    @staticmethod
    def _derive(rec: EntityRecord) -> None:
        """Grounded mode: the user-facing summaries are derived from the outcome expectations."""
        rec.threat = max(rec.adverse, rec.risk)
        rec.warmth = float(np.clip(rec.benign - rec.threat, -1.0, 1.0))

    def observe(self, rec: EntityRecord, t: float, dt: float) -> None:
        """Seen during this frame: familiarity grows."""
        rec.exposure_s += dt
        rec.last_seen_t = t

    # ── learning ────────────────────────────────────────────────────────────
    def learn(self, rec: EntityRecord, t: float, kind: str, emotions: list[tuple[str, float]],
              arousal: float, surprise_negative: bool, confidence: float = 1.0, record: bool = True) -> None:
        """Legacy mode: update the entity's associations from the emotions an event with it elicited.
        Grounded mode: only record the episode (the emotions are history, not a teaching signal)."""
        if self.learning == "grounded":
            if record:
                self._record(rec, t, kind, emotions, confidence)
            return
        fear = sum(i for lbl, i in emotions if lbl == "fear")
        valence = (sum(i for lbl, i in emotions if lbl in ("joy", "hope", "interest"))
                   - sum(i for lbl, i in emotions if lbl in ("fear", "distress")))
        a = min(1.0, ALPHA * (1.0 + max(0.0, arousal)) * confidence)
        rec.threat += a * (min(fear, 1.0) - rec.threat)
        rec.warmth += a * (float(np.clip(valence, -1, 1)) - rec.warmth)
        rec.trust += a * ((0.0 if surprise_negative else 1.0) - rec.trust)
        if not record:  # e.g. replay during consolidation: updates the association, not the history
            return
        self._record(rec, t, kind, emotions, confidence)

    @staticmethod
    def _record(rec: EntityRecord, t: float, kind: str, emotions, confidence: float) -> None:
        rec.episodes.append({"t": round(t, 2), "event": kind, "emotions": [(lbl, round(i, 2)) for lbl, i in emotions],
                             "confidence": round(confidence, 2)})
        del rec.episodes[:-MAX_EPISODES]

    def learn_outcome(self, rec: EntityRecord, outcome, confidence: float = 1.0, record: bool = True,
                      replay: bool = False) -> None:
        """Grounded mode: update the outcome expectations from a measured outcome attributed to `rec`.
        replay=True (consolidation): restore toward the learned levels only. Ignored in legacy mode."""
        if self.learning != "grounded":
            return
        a = min(1.0, OUTCOME_ALPHA * confidence)
        if replay:
            if outcome.adverse:
                rec.adverse += a * max(0.0, rec.adverse_learned - rec.adverse)
            else:
                rec.benign += a * max(0.0, rec.benign_learned - rec.benign)
            self._derive(rec)
            return
        m = outcome.magnitude
        if outcome.adverse:
            rec.adverse += a * (m - rec.adverse)
        else:
            rec.benign += a * (m - rec.benign)
            rec.adverse += a * EXTINCTION * (0.0 - rec.adverse)
            rec.risk += a * EXTINCTION * (0.0 - rec.risk)
            rec.risk_learned = rec.risk
        rec.outcomes += 1
        rec.adverse_learned, rec.benign_learned = rec.adverse, rec.benign
        self._derive(rec)
        if record:
            rec.episodes.append({"t": round(outcome.time_s, 2), "event": f"outcome:{outcome.kind}",
                                 "magnitude": round(m, 2), "confidence": round(confidence, 2)})
            del rec.episodes[:-MAX_EPISODES]

    def learn_risk(self, rec: EntityRecord, risk, confidence: float = 1.0, record: bool = True,
                   replay: bool = False) -> None:
        """Grounded mode: learn from an estimated near miss (RiskEstimate) attributed to `rec`. Same rule as
        an adverse outcome, in a separate field; replay restores toward the learned level only."""
        if self.learning != "grounded":
            return
        a = min(1.0, OUTCOME_ALPHA * confidence)
        if replay:
            rec.risk += a * max(0.0, rec.risk_learned - rec.risk)
        else:
            rec.risk += a * (risk.magnitude - rec.risk)
            rec.risk_learned = rec.risk
            if record:
                rec.episodes.append({"t": round(risk.time_s, 2), "event": f"risk:{risk.kind}",
                                     "magnitude": round(risk.magnitude, 2), "confidence": round(confidence, 2)})
                del rec.episodes[:-MAX_EPISODES]
        self._derive(rec)
