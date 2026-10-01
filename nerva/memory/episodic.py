"""Episodic memory and consolidation (docs/memory_design.md, phases M2-M3).

Hippocampus-like fast store of significant events, linked to the slow entity memory (nerva.memory.entity)
by offline replay:

  encode      an event is stored only if salient: strength s0 = relevance·(W_AROUSAL·arousal
              + W_SURPRISE·|prediction error| + W_NOVELTY·novelty) ≥ ENCODE_MIN      [McGaugh: arousal
              gates consolidation]. Routine moments only increment a counter.
  activation  A(t) = ln Σ_j (t − t_j)^(−DECAY) + s0 over the encoding and every retrieval time
              [ACT-R base-level learning, Anderson & Schooler 1991]: power-law forgetting, strengthened
              by use.
  retrieve    by cue (entity, event kind, place): score = activation + cue match; retrieval is itself a
              use (it raises activation) [use-dependent memory].
  consolidate "sleep" in idle time: replay the top-k episodes by priority = intensity·(1 + |PE|)·recency
              [prioritised replay, Mattar & Daw 2018] among emotionally significant ones; replay nudges the entity's association toward the
              episode's emotions (keeps important lessons from drifting away); repeated similar episodes
              are merged into one gist episode with a count; already-consolidated, low-activation
              episodes are pruned when over capacity [adaptive forgetting, Richards & Frankland 2017].
  persist     SQLite (stdlib), one file; survives restarts.
  grounded    (refactor stage D) with a grounded entity memory, measured outcomes are stored as
              "outcome:<kind>" episodes (encode_outcome) and only those are replayed into the entity's
              outcome expectations; emotion episodes stay as history.

Bounded (CAPACITY episodes), pure Python + NumPy, O(N) retrieval over N ≤ CAPACITY: fits a
Jetson-class computer. All constants are NERVA design choices.
"""

from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass, field

CAPACITY = 5000
ENCODE_MIN = 0.15
W_AROUSAL, W_SURPRISE, W_NOVELTY = 0.6, 0.8, 0.3
DECAY = 0.5  # ACT-R default base-level decay
MERGE_WINDOW_S = 30.0  # same entity + same event kind within this window → one gist episode
REPLAY_TOP_K = 10
REPLAY_RATE = 0.2  # fraction of a normal learning step applied per replay
REPLAY_MIN_INTENSITY = 0.3  # only emotionally significant episodes are replayed (trivia would dilute lessons)
PRUNE_ACTIVATION = -2.0


@dataclass
class Episode:
    t: float
    event: str
    entity: str | None
    place: tuple[float, float]
    emotions: list  # [(label, intensity)]
    arousal: float
    prediction_error: float
    strength: float
    uses: list = field(default_factory=list)  # retrieval times (encoding time is t)
    count: int = 1
    consolidated: bool = False
    magnitude: float = 1.0  # for outcome episodes ("outcome:<kind>"): the outcome's magnitude

    def intensity(self) -> float:
        return sum(i for _, i in self.emotions)

    def activation(self, now: float) -> float:
        times = [self.t, *self.uses]
        total = sum(max(now - tj, 0.05) ** (-DECAY) for tj in times)
        return math.log(total) + self.strength + math.log(self.count)


class EpisodicMemory:
    def __init__(self, capacity: int = CAPACITY):
        self.capacity = capacity
        self.episodes: list[Episode] = []
        self.routine_count = 0

    # ── encoding ────────────────────────────────────────────────────────────
    def encode(self, t: float, event: str, entity: str | None, place, emotions, relevance: float,
               arousal: float, prediction_error: float, novelty: float) -> Episode | None:
        s0 = relevance * (W_AROUSAL * max(0.0, arousal) + W_SURPRISE * abs(prediction_error) + W_NOVELTY * novelty)
        if s0 < ENCODE_MIN:
            self.routine_count += 1
            return None
        ep = Episode(t, event, entity, (float(place[0]), float(place[1])), list(emotions), arousal,
                     prediction_error, s0)
        self.episodes.append(ep)
        if len(self.episodes) > self.capacity:
            self._prune(t, force=True)
        return ep

    def encode_outcome(self, t: float, outcome, entity: str | None, place, arousal: float = 0.0) -> Episode:
        """Store a measured outcome (grounded mode). Outcomes are always significant: strength = magnitude.
        The episode is named "outcome:<kind>" and carries no emotions."""
        ep = Episode(t, f"outcome:{outcome.kind}", entity, (float(place[0]), float(place[1])), [], arousal, 0.0,
                     float(outcome.magnitude), magnitude=float(outcome.magnitude))
        self.episodes.append(ep)
        if len(self.episodes) > self.capacity:
            self._prune(t, force=True)
        return ep

    # ── retrieval ───────────────────────────────────────────────────────────
    def retrieve(self, now: float, entity: str | None = None, event: str | None = None, place=None,
                 k: int = 5, radius: float = 1.5) -> list[Episode]:
        scored = []
        for ep in self.episodes:
            if entity is not None and ep.entity != entity:
                continue  # "who" is a filter: a vivid memory of someone else must not answer the cue
            match = 0.0
            if event is not None:
                match += 0.5 if ep.event == event else 0.0
            if place is not None:
                match += 0.5 if math.dist(ep.place, place) < radius else 0.0
            scored.append((ep.activation(now) + match, ep))
        scored.sort(key=lambda x: x[0], reverse=True)
        out = [ep for _, ep in scored[:k]]
        for ep in out:
            ep.uses.append(now)
        return out

    # ── consolidation ───────────────────────────────────────────────────────
    def consolidate(self, now: float, entity_memory=None, top_k: int = REPLAY_TOP_K) -> dict:
        """One 'sleep' pass: prioritised replay → entity updates, merge repeats, prune. Returns a report."""
        grounded = entity_memory is not None and getattr(entity_memory, "learning", "legacy") == "grounded"
        if grounded:  # only measured outcomes are replayed; their significance is the outcome's magnitude
            candidates = [e for e in self.episodes if e.event.startswith("outcome:") and e.magnitude >= REPLAY_MIN_INTENSITY]
            replayed = sorted(candidates, key=lambda e: e.magnitude * math.exp(-(now - e.t) / 600.0),
                              reverse=True)[:top_k]
        else:
            candidates = [e for e in self.episodes if e.intensity() >= REPLAY_MIN_INTENSITY]
            replayed = sorted(candidates, key=lambda e: e.intensity() * (1 + abs(e.prediction_error))
                              * math.exp(-(now - e.t) / 600.0), reverse=True)[:top_k]
        for ep in self.episodes:  # everything seen in this sleep is consolidated (eligible for merge/prune)
            ep.consolidated = True
        for ep in replayed:
            ep.uses.append(now)  # replay is a use
            if entity_memory is not None and ep.entity in entity_memory.records:
                rec = entity_memory.records[ep.entity]
                if not grounded:  # legacy: replay re-learns the stored emotions
                    entity_memory.learn(rec, now, f"replay:{ep.event}", ep.emotions,
                                        ep.arousal, surprise_negative=False, confidence=REPLAY_RATE, record=False)
                elif ep.event.startswith("outcome:"):  # grounded: only replayed OUTCOMES teach
                    from nerva.interfaces import OutcomeSignal

                    entity_memory.learn_outcome(rec, OutcomeSignal(ep.event.split(":", 1)[1], ep.magnitude, now),
                                                confidence=REPLAY_RATE * ep.count, record=False, replay=True)
            ep.consolidated = True
        merged = self._merge()
        pruned = self._prune(now)
        return {"replayed": len(replayed), "merged": merged, "pruned": pruned, "size": len(self.episodes)}

    def _merge(self) -> int:
        merged, keep = 0, []
        for ep in sorted(self.episodes, key=lambda e: e.t):
            last = keep[-1] if keep else None
            if (last is not None and last.consolidated and ep.consolidated and last.entity == ep.entity
                    and last.event == ep.event and ep.t - last.t < MERGE_WINDOW_S):
                last.count += ep.count
                last.uses.extend(ep.uses)
                last.strength = max(last.strength, ep.strength)
                merged += 1
            else:
                keep.append(ep)
        self.episodes = keep
        return merged

    def _prune(self, now: float, force: bool = False) -> int:
        """Forget consolidated episodes whose activation fell below PRUNE_ACTIVATION, then enforce capacity
        (least active consolidated episodes first; unconsolidated ones only if still over capacity)."""
        before = len(self.episodes)
        self.episodes = [e for e in self.episodes if not (e.consolidated and e.activation(now) < PRUNE_ACTIVATION)]
        excess = len(self.episodes) - self.capacity
        if excess > 0:
            ranked = sorted(self.episodes, key=lambda e: (not e.consolidated, e.activation(now)))
            victims = {id(e) for e in ranked[:excess]}
            self.episodes = [e for e in self.episodes if id(e) not in victims]
        return before - len(self.episodes)

    # ── persistence ─────────────────────────────────────────────────────────
    def save(self, path: str) -> None:
        con = sqlite3.connect(path)
        with con:
            con.execute("DROP TABLE IF EXISTS episodes")
            con.execute("CREATE TABLE episodes (data TEXT)")
            con.executemany("INSERT INTO episodes VALUES (?)",
                            [(json.dumps(ep.__dict__),) for ep in self.episodes])
            con.execute("DROP TABLE IF EXISTS meta")
            con.execute("CREATE TABLE meta (routine_count INTEGER)")
            con.execute("INSERT INTO meta VALUES (?)", (self.routine_count,))
        con.close()

    @classmethod
    def load(cls, path: str, capacity: int = CAPACITY) -> "EpisodicMemory":
        mem = cls(capacity)
        con = sqlite3.connect(path)
        for (data,) in con.execute("SELECT data FROM episodes"):
            d = json.loads(data)
            d["place"] = tuple(d["place"])
            d["emotions"] = [tuple(x) for x in d["emotions"]]
            mem.episodes.append(Episode(**d))
        mem.routine_count = con.execute("SELECT routine_count FROM meta").fetchone()[0]
        con.close()
        return mem
