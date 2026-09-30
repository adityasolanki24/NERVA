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
All constants are NERVA design choices (docs/memory_design.md §3.3); nothing here is fitted to data.
Pure Python/NumPy, O(entities) per lookup: fits embedded hardware.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

MATCH_COSINE = 0.85
NEW_IDENTITY_COSINE = 0.5  # between this and MATCH_COSINE an observation is ambiguous: no new identity
HABITUATION_S = {"person": 30.0, "ball": 30.0}
FORGET_EXPOSURE_S = 600.0  # familiarity fades with this time constant while absent
DRIFT_S = 900.0  # threat/warmth relax toward neutral with this time constant while absent
ALPHA = 0.5
APPEARANCE_RATE = 0.1  # running-mean rate of the identity embedding
MAX_EPISODES = 20


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

    def novelty(self) -> float:
        return math.exp(-self.exposure_s / HABITUATION_S.get(self.kind, 30.0))


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0


class EntityMemory:
    def __init__(self):
        self.records: dict[str, EntityRecord] = {}
        self._counter: dict[str, int] = {}

    # ── identity ────────────────────────────────────────────────────────────
    def resolve(self, kind: str, appearance: np.ndarray | None, t: float) -> EntityRecord | None:
        """The known entity this observation belongs to, a new one if it is clearly unlike all known
        entities, or None if ambiguous (e.g. a partial view). Absence effects are applied here."""
        best, best_sim, closest = None, MATCH_COSINE, 0.0
        for rec in self.records.values():
            if rec.kind != kind:
                continue
            sim = 1.0 if (appearance is None and rec.appearance is None) else (
                _cosine(appearance, rec.appearance) if appearance is not None and rec.appearance is not None else 0.0)
            closest = max(closest, sim)
            if sim >= best_sim:
                best, best_sim = rec, sim
        if best is None and closest >= NEW_IDENTITY_COSINE:
            return None
        if best is None:
            n = self._counter.get(kind, 0)
            self._counter[kind] = n + 1
            best = EntityRecord(f"{kind}#{n}", kind, None if appearance is None else np.asarray(appearance, float),
                                encounters=1)
            self.records[best.eid] = best
        else:
            self._apply_absence(best, t)
            if appearance is not None and best.appearance is not None:
                best.appearance = (1 - APPEARANCE_RATE) * best.appearance + APPEARANCE_RATE * np.asarray(appearance)
        return best

    def _apply_absence(self, rec: EntityRecord, t: float) -> None:
        gap = t - rec.last_seen_t
        if math.isfinite(gap) and gap > 1.0:
            rec.exposure_s *= math.exp(-gap / FORGET_EXPOSURE_S)
            drift = math.exp(-gap / DRIFT_S)
            rec.threat *= drift
            rec.warmth *= drift
            rec.encounters += 1

    def observe(self, rec: EntityRecord, t: float, dt: float) -> None:
        """Seen during this frame: familiarity grows."""
        rec.exposure_s += dt
        rec.last_seen_t = t

    # ── learning ────────────────────────────────────────────────────────────
    def learn(self, rec: EntityRecord, t: float, kind: str, emotions: list[tuple[str, float]],
              arousal: float, surprise_negative: bool, confidence: float = 1.0, record: bool = True) -> None:
        """Update the entity's associations from the emotions an event with it elicited."""
        fear = sum(i for lbl, i in emotions if lbl == "fear")
        valence = (sum(i for lbl, i in emotions if lbl in ("joy", "hope", "interest"))
                   - sum(i for lbl, i in emotions if lbl in ("fear", "distress")))
        a = min(1.0, ALPHA * (1.0 + max(0.0, arousal)) * confidence)
        rec.threat += a * (min(fear, 1.0) - rec.threat)
        rec.warmth += a * (float(np.clip(valence, -1, 1)) - rec.warmth)
        rec.trust += a * ((0.0 if surprise_negative else 1.0) - rec.trust)
        if not record:  # e.g. replay during consolidation: updates the association, not the history
            return
        rec.episodes.append({"t": round(t, 2), "event": kind, "emotions": [(lbl, round(i, 2)) for lbl, i in emotions],
                             "confidence": round(confidence, 2)})
        del rec.episodes[:-MAX_EPISODES]
