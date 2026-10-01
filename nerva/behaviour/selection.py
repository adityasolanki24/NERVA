"""Behaviour v2: emotion-modulated action selection (utility arbitration) instead of fixed rules.

Same perception, appraisal, affect and per-action controllers as behaviour v1
(nerva.behaviour.modes); only the choice of WHAT to do changes. Every 0.1 s each candidate
action gets a utility from the current drives, and the highest wins:

  drives (from the affect model's ActionTendencyState; with Model A these equal the former emotion
  drives exactly, see nerva/affect/tendencies.py):
    curiosity C = explore           social S = approach            fear F = avoid
    startle   Z = orient            distress D = withdraw          freeze Q = freeze
  per target k: novelty n_k (appraisal habituation), distance d_k, proximity p_k = max(0, 1 − d_k / SAFE)

  explore        EXPLORE_BASE
  orient k       Z + NEW_BONUS · [k appeared < 2 s ago]
  approach k     (C · n_k + S · [k = person]) · [d_k > stop_k] · (1 − FEAR_BLOCK · F)⁺
  inspect k      (C · n_k + S · [k = person]) · [d_k ≤ stop_k + 0.1] · (1 − FEAR_BLOCK · F)⁺
  watch          W_WATCH · F                                  (person tracked)
  retreat        W_RETREAT · F · p_person                     (person tracked or just lost)
  freeze         W_FREEZE · Q · p_person                       (Model A: Q = Z · F)
  withdraw       W_WITHDRAW · D

The current action gets PERSISTENCE extra (hysteresis); freeze and the first part of a retreat are
committed for their minimum duration. Emotions therefore compete: the same fear that wins over
curiosity up close only produces watching from far away, and curiosity about a person also depends
on how novel they still are.

Target-conditioned arbitration (v2, when `directed` tendencies are passed): C, S and F for a target k
are the tendencies directed at k plus the undirected ones (C_k, S_k, F_k), so fear of one person gates
approaching THAT person, not everyone; watch/retreat/freeze use the focal person's F and freeze; the focal
person is the one with the highest max(remembered threat, directed avoidance). Orient (Z) and withdraw (D)
stay global. Without `directed` the selector behaves exactly as before. All weights are NERVA design choices, set so the scenario behaves
sensibly; they are not fitted to data. This is a hand-designed arbitration, not a learned policy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from nerva.behaviour.modes import (BACKSTEP_S, FREEZE_S, SAFE_DISTANCE, ReactiveBehaviour)
from nerva.interfaces import ActionTendencyState

EXPLORE_BASE = 0.08
NEW_BONUS = 0.3
FEAR_BLOCK = 4.0
W_WATCH, W_RETREAT, W_FREEZE, W_WITHDRAW = 1.0, 3.0, 4.0, 1.0
PERSISTENCE = 0.05


@dataclass
class UtilityBehaviour(ReactiveBehaviour):
    utilities: dict = field(default_factory=dict)

    def _directed(self, key) -> tuple[float, float, float, float]:
        """(explore, approach, avoid, freeze) directed at `key` plus the undirected share (v2)."""
        zero = ActionTendencyState()
        und, d = self.directed.get("", zero), self.directed.get(key, zero) if key else zero
        return (d.explore + und.explore, d.approach + und.approach, d.avoid + und.avoid, d.freeze + und.freeze)

    def _focal_person(self, tr):
        if getattr(self, "directed", None) is None:
            return super()._focal_person(tr)
        people = [k for k, x in tr.items() if x.kind == "person"]
        if not people:
            return None
        return max(people, key=lambda k: (max(self.threats.get(k, 0.0), self._directed(k)[2]), -tr[k].distance))

    def _candidates(self, t, pad, tend: ActionTendencyState, tr) -> dict[tuple[str, str | None], float]:
        c, s = tend.explore, tend.approach
        f, z, d = tend.avoid, tend.orient, tend.withdraw
        targeted = getattr(self, "directed", None) is not None
        fear_gate = max(0.0, 1.0 - FEAR_BLOCK * f)
        u: dict[tuple[str, str | None], float] = {("explore", None): EXPLORE_BASE}
        focal = self._focal_person(tr)
        for key, track in tr.items():
            novelty = self.salience.get(key, 1.0)
            new = 1.0 if t - self.appeared_at.get(key, -1e9) < 2.0 else 0.0
            u[("orient", key)] = z + NEW_BONUS * new
            person = track.kind == "person"
            if targeted:
                c_k, s_k, f_k, _ = self._directed(key)
                gate_k = max(0.0, 1.0 - FEAR_BLOCK * f_k)
            else:
                c_k, s_k, gate_k = c, s, fear_gate
            # a remembered threat is not approached, even when others are liked
            gate = gate_k * (max(0.0, 1.0 - 4.0 * self.threats.get(key, 0.0)) if person else 1.0)
            pull = (c_k * novelty + (s_k if person else 0.0)) * gate
            stop = self._stop_distance(key, pad)
            margin = 0.3 if (self.mode == "inspect" and self.target == key) else 0.1
            u[("inspect" if track.distance <= stop + margin else "approach", key)] = pull
        if focal is not None:
            person = tr[focal]
            proximity = max(0.0, 1.0 - person.distance / SAFE_DISTANCE)
            f_focal, q_focal = (self._directed(focal)[2], self._directed(focal)[3]) if targeted else (f, tend.freeze)
            u[("watch", focal)] = W_WATCH * f_focal
            u[("retreat", focal)] = W_RETREAT * f_focal * proximity
            u[("freeze", focal)] = W_FREEZE * q_focal * proximity
        elif self.mode == "retreat":  # the threat just went out of view: keep moving away while afraid
            f_gone = self._directed(self.target)[2] if targeted else f
            u[("retreat", self.target)] = W_RETREAT * f_gone * 0.5
        u[("withdraw", self.target)] = W_WITHDRAW * d
        return u

    def _select(self, t, pad, tend, tr) -> None:
        age = t - self.mode_since
        if (self.mode == "freeze" and age < FREEZE_S) or (self.mode == "retreat" and age < BACKSTEP_S):
            self.utilities = {}
            return  # committed
        u = self._candidates(t, pad, tend, tr)
        current = (self.mode, self.target)
        if current in u:
            u[current] += PERSISTENCE
        self.utilities = {f"{m}:{k}" if k else m: round(v, 3) for (m, k), v in u.items()}
        (mode, target), _ = max(u.items(), key=lambda kv: kv[1])
        self._set(t, mode, target)
