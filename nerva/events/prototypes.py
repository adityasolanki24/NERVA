"""Online, bounded prototype clustering of event embeddings, with outcome statistics (refactor stage H).

Each detected event boundary gives an embedding (here: the scaled feature vector after the boundary and
its change across it). A prototype is a running mean of the embeddings assigned to it:

  assign   nearest prototype if its distance < NEW_DISTANCE, else a new prototype (DP-means style);
           at MAX_PROTOTYPES the nearest one is used regardless (bounded memory)
  outcomes per prototype, how often each outcome kind followed within the horizon: P(kind | prototype)
           with a Beta(PRIOR_A, PRIOR_B) prior, so a rarely seen prototype stays near the prior

Prototypes are inspectable (count, centre, outcome table). They are NOT emotions and are not claimed to
match human event categories. Constants are NERVA design choices.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

NEW_DISTANCE = 2.5
MAX_PROTOTYPES = 12
PRIOR_A, PRIOR_B = 0.5, 4.5  # prior mean 0.1


@dataclass
class Prototype:
    pid: int
    centre: np.ndarray
    count: int = 1
    outcome_counts: dict = field(default_factory=dict)  # kind -> number of occurrences followed by it
    resolved: int = 0  # occurrences whose horizon has passed (denominator)

    def p(self, kind: str) -> float:
        return (self.outcome_counts.get(kind, 0) + PRIOR_A) / (self.resolved + PRIOR_A + PRIOR_B)


class PrototypeMemory:
    def __init__(self, new_distance: float = NEW_DISTANCE, max_prototypes: int = MAX_PROTOTYPES):
        self.new_distance, self.max_prototypes = new_distance, max_prototypes
        self.prototypes: list[Prototype] = []

    def assign(self, emb: np.ndarray) -> Prototype:
        emb = np.asarray(emb, dtype=float)
        if self.prototypes:
            dists = [float(np.linalg.norm(emb - p.centre)) for p in self.prototypes]
            i = int(np.argmin(dists))
            if dists[i] < self.new_distance or len(self.prototypes) >= self.max_prototypes:
                p = self.prototypes[i]
                p.count += 1
                p.centre += (emb - p.centre) / p.count
                return p
        p = Prototype(len(self.prototypes), emb.copy())
        self.prototypes.append(p)
        return p

    @staticmethod
    def resolve(p: Prototype, outcome_kinds) -> None:
        """Record which outcome kinds followed one occurrence of p within the horizon."""
        p.resolved += 1
        for k in set(outcome_kinds):
            p.outcome_counts[k] = p.outcome_counts.get(k, 0) + 1
