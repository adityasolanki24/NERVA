"""Behaviour v2: emotion-modulated action selection (utility arbitration) instead of fixed rules.

Same perception, appraisal, affect and per-action controllers as behaviour v1
(nerva.reactive_behaviour); only the choice of WHAT to do changes. Every 0.1 s each candidate
action gets a utility from the current drives, and the highest wins:

  drives (from the active emotions):
    curiosity C = interest          social S = hope + joy          fear F = fear
    startle   Z = surprise          distress D = distress
  per target k: novelty n_k (appraisal habituation), distance d_k, proximity p_k = max(0, 1 − d_k / SAFE)

  explore        EXPLORE_BASE
  orient k       Z + NEW_BONUS · [k appeared < 2 s ago]
  approach k     (C · n_k + S · [k = person]) · [d_k > stop_k] · (1 − FEAR_BLOCK · F)⁺
  inspect k      (C · n_k + S · [k = person]) · [d_k ≤ stop_k + 0.1] · (1 − FEAR_BLOCK · F)⁺
  watch          W_WATCH · F                                  (person tracked)
  retreat        W_RETREAT · F · p_person                     (person tracked or just lost)
  freeze         W_FREEZE · Z · F · p_person
  withdraw       W_WITHDRAW · D

The current action gets PERSISTENCE extra (hysteresis); freeze and the first part of a retreat are
committed for their minimum duration. Emotions therefore compete: the same fear that wins over
curiosity up close only produces watching from far away, and curiosity about a person also depends
on how novel they still are. All weights are NERVA design choices, set so the scenario behaves
sensibly; they are not fitted to data. This is a hand-designed arbitration, not a learned policy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from nerva.reactive_behaviour import (BACKSTEP_S, FREEZE_S, SAFE_DISTANCE, ReactiveBehaviour)

EXPLORE_BASE = 0.08
NEW_BONUS = 0.3
FEAR_BLOCK = 4.0
W_WATCH, W_RETREAT, W_FREEZE, W_WITHDRAW = 1.0, 3.0, 4.0, 1.0
PERSISTENCE = 0.05


@dataclass
class UtilityBehaviour(ReactiveBehaviour):
    utilities: dict = field(default_factory=dict)

    def _candidates(self, t, pad, emo, tr) -> dict[tuple[str, str | None], float]:
        c, s = emo.get("interest", 0.0), emo.get("hope", 0.0) + emo.get("joy", 0.0)
        f, z, d = emo.get("fear", 0.0), emo.get("surprise", 0.0), emo.get("distress", 0.0)
        fear_gate = max(0.0, 1.0 - FEAR_BLOCK * f)
        u: dict[tuple[str, str | None], float] = {("explore", None): EXPLORE_BASE}
        focal = self._focal_person(tr)
        for key, track in tr.items():
            novelty = self.salience.get(key, 1.0)
            new = 1.0 if t - self.appeared_at.get(key, -1e9) < 2.0 else 0.0
            u[("orient", key)] = z + NEW_BONUS * new
            person = track.kind == "person"
            # a remembered threat is not approached, even when others are liked
            gate = fear_gate * (max(0.0, 1.0 - 4.0 * self.threats.get(key, 0.0)) if person else 1.0)
            pull = (c * novelty + (s if person else 0.0)) * gate
            stop = self._stop_distance(key, pad)
            margin = 0.3 if (self.mode == "inspect" and self.target == key) else 0.1
            u[("inspect" if track.distance <= stop + margin else "approach", key)] = pull
        if focal is not None:
            person = tr[focal]
            proximity = max(0.0, 1.0 - person.distance / SAFE_DISTANCE)
            u[("watch", focal)] = W_WATCH * f
            u[("retreat", focal)] = W_RETREAT * f * proximity
            u[("freeze", focal)] = W_FREEZE * z * f * proximity
        elif self.mode == "retreat":  # the threat just went out of view: keep moving away while afraid
            u[("retreat", self.target)] = W_RETREAT * f * 0.5
        u[("withdraw", self.target)] = W_WITHDRAW * d
        return u

    def _select(self, t, pad, emo, tr) -> None:
        age = t - self.mode_since
        if (self.mode == "freeze" and age < FREEZE_S) or (self.mode == "retreat" and age < BACKSTEP_S):
            self.utilities = {}
            return  # committed
        u = self._candidates(t, pad, emo, tr)
        current = (self.mode, self.target)
        if current in u:
            u[current] += PERSISTENCE
        self.utilities = {f"{m}:{k}" if k else m: round(v, 3) for (m, k), v in u.items()}
        (mode, target), _ = max(u.items(), key=lambda kv: kv[1])
        self._set(t, mode, target)
