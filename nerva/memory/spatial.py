"""Spatial memory (docs/memory_design.md, phase M4): which places the robot knows and how it feels there.

A coarse grid of places (CELL_M metres), a minimal version of the place layer of a 3D scene graph
(Hydra) and of the hippocampal "cognitive map":
  familiarity   time spent in the cell → novelty = exp(-visits_s / HABITUATION_S); fades while away
  place affect  threat and valence learned from emotional events that happen there (same prediction-
                error rule as entity memory), so "the spot where I was frightened" keeps a mark
  exploration   the most attractive heading = argmax over neighbouring cells of novelty − threat,
                which lets exploration favour unfamiliar, safe places instead of wandering at random
The robot's pose comes from simulation now and from odometry/SLAM on hardware; this layer stores only
places. Dict-of-cells: memory grows with explored area only (≈ tens of bytes per cell). NERVA design
choices throughout.

Learning modes as in entity memory (refactor stage D): "legacy" learns place threat/valence from the
elicited emotions (learn); "grounded" only from measured outcomes at the place (learn_outcome), and
learn() is ignored.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

CELL_M = 0.75
HABITUATION_S = 20.0
FORGET_S = 900.0
ALPHA = 0.4
NEIGHBOUR_RING = 2  # consider cells up to this many steps away when choosing an exploration heading


@dataclass
class Place:
    visits_s: float = 0.0
    last_t: float = -math.inf
    threat: float = 0.0
    valence: float = 0.0


class PlaceMemory:
    def __init__(self, cell: float = CELL_M, learning: str = "legacy"):
        if learning not in ("legacy", "grounded"):
            raise ValueError("learning must be 'legacy' or 'grounded'")
        self.cell = cell
        self.learning = learning
        self.places: dict[tuple[int, int], Place] = {}

    def key(self, xy) -> tuple[int, int]:
        return (math.floor(xy[0] / self.cell), math.floor(xy[1] / self.cell))

    def _place(self, key, t) -> Place:
        p = self.places.setdefault(key, Place())
        if math.isfinite(p.last_t) and t - p.last_t > 1.0:
            p.visits_s *= math.exp(-(t - p.last_t) / FORGET_S)
        return p

    def observe(self, xy, t: float, dt: float) -> None:
        p = self._place(self.key(xy), t)
        p.visits_s += dt
        p.last_t = t

    def novelty(self, key) -> float:
        p = self.places.get(key)
        return 1.0 if p is None else math.exp(-p.visits_s / HABITUATION_S)

    def threat(self, key) -> float:
        p = self.places.get(key)
        return 0.0 if p is None else p.threat

    def learn(self, xy, t: float, emotions, arousal: float) -> None:
        """Legacy mode only: learn from the elicited emotions."""
        if self.learning != "legacy":
            return
        p = self._place(self.key(xy), t)
        fear = sum(i for lbl, i in emotions if lbl == "fear")
        valence = (sum(i for lbl, i in emotions if lbl in ("joy", "hope", "interest")) - fear
                   - sum(i for lbl, i in emotions if lbl == "distress"))
        a = min(1.0, ALPHA * (1.0 + max(0.0, arousal)))
        p.threat += a * (min(fear, 1.0) - p.threat)
        p.valence += a * (max(-1.0, min(1.0, valence)) - p.valence)

    def learn_outcome(self, xy, t: float, outcome) -> None:
        """Grounded mode only: an adverse outcome here raises place threat; a benign one raises valence."""
        if self.learning != "grounded":
            return
        p = self._place(self.key(xy), t)
        m = outcome.magnitude
        if outcome.adverse:
            p.threat += ALPHA * (m - p.threat)
            p.valence += ALPHA * (-m - p.valence)
        else:
            p.valence += ALPHA * (m - p.valence)

    def learn_risk(self, xy, t: float, risk) -> None:
        """Grounded mode only: an estimated near miss here raises place threat (as an adverse outcome would)."""
        if self.learning != "grounded":
            return
        p = self._place(self.key(xy), t)
        p.threat += ALPHA * (risk.magnitude - p.threat)
        p.valence += ALPHA * (-risk.magnitude - p.valence)

    def explore_heading(self, xy, yaw: float) -> tuple[float, float]:
        """(bearing relative to yaw, attractiveness) of the best nearby cell by novelty − threat."""
        cx, cy = self.key(xy)
        best = (0.0, -math.inf)
        for dx in range(-NEIGHBOUR_RING, NEIGHBOUR_RING + 1):
            for dy in range(-NEIGHBOUR_RING, NEIGHBOUR_RING + 1):
                if dx == dy == 0:
                    continue
                k = (cx + dx, cy + dy)
                centre = ((k[0] + 0.5) * self.cell, (k[1] + 0.5) * self.cell)
                bearing = math.atan2(centre[1] - xy[1], centre[0] - xy[0]) - yaw
                bearing = (bearing + math.pi) % (2 * math.pi) - math.pi
                # prefer ahead a little (turning costs time), and nearer rings
                score = self.novelty(k) - 2.0 * self.threat(k) - 0.1 * abs(bearing) - 0.1 * max(abs(dx), abs(dy))
                if score > best[1]:
                    best = (bearing, score)
        return best
