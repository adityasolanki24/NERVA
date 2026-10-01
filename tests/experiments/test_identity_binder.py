"""Outcome attribution through identity binding (experiments/reactive/scenario.py)."""

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments" / "reactive"))
import scenario  # noqa: E402

from nerva.interfaces import OutcomeSignal, Track  # noqa: F401  # noqa: E402
from nerva.memory.entity import EntityMemory  # noqa: E402

BLUE = np.array([0, 0, 0, 0, 0, 1.0, 0, 0])
GREEN = np.array([0, 0, 1.0, 0, 0, 0, 0, 0])


@dataclass
class _St:
    appearance: object


class _Perception:
    def __init__(self, appearances):
        self.tracks = {tid: _St(a) for tid, a in appearances.items()}


def test_an_outcome_while_identity_is_unknown_is_credited_once_the_identity_is_resolved():
    mem = EntityMemory(learning="grounded")
    binder = scenario.IdentityBinder(mem)
    a_rec = mem.resolve("person", BLUE, 0.0)
    b_rec = mem.resolve("person", GREEN, 0.0)
    tracks = (Track("person", 0.0, 0.4, tid="person-5"),)
    binder.update(1.0, tracks, _Perception({"person-5": None}))  # too close: appearance uninformative
    binder.learn_outcome("person-5", OutcomeSignal("contact_impact", 1.0, 1.0, source="person-5"))
    assert a_rec.adverse == 0.0 and b_rec.adverse == 0.0  # held back, nobody blamed yet
    binder.update(2.0, tracks, _Perception({"person-5": GREEN}))  # "oh, it was B"
    assert b_rec.adverse > 0.0 and a_rec.adverse == 0.0


def test_an_unresolved_track_that_disappears_creates_no_phantom_attribution():
    mem = EntityMemory(learning="grounded")
    binder = scenario.IdentityBinder(mem)
    a_rec = mem.resolve("person", BLUE, 0.0)
    tracks = (Track("person", 0.0, 0.4, tid="person-7"),)
    binder.update(1.0, tracks, _Perception({"person-7": None}))
    binder.learn_outcome("person-7", OutcomeSignal("contact_impact", 1.0, 1.0, source="person-7"))
    binder.update(2.0, (), _Perception({}))  # track lost before identity was known
    binder.update(3.0, (Track("person", 0.0, 2.0, tid="person-8"),), _Perception({"person-8": BLUE}))
    assert a_rec.adverse == 0.0 and not binder.pending_outcomes


def test_touch_is_attributed_from_the_world_model():
    from nerva.world.model import WorldModel

    wm = WorldModel()
    s = wm.update(0.0, (0, 0), 0.0, (Track("person", 0.0, 0.3, tid="person-1"), Track("person", 1.0, 2.0, tid="person-2")))
    assert scenario.touch_source_from_world(s) == "person-1"


def test_a_risk_estimate_on_an_unknown_track_is_credited_as_risk_not_as_an_outcome():
    from nerva.interfaces import RiskEstimate

    mem = EntityMemory(learning="grounded")
    binder = scenario.IdentityBinder(mem)
    b_rec = mem.resolve("person", GREEN, 0.0)
    tracks = (Track("person", 0.0, 0.4, tid="person-5"),)
    binder.update(1.0, tracks, _Perception({"person-5": None}))
    binder.learn_outcome("person-5", RiskEstimate("collision_risk", 1.0, 1.0, source="person-5"))
    binder.update(2.0, tracks, _Perception({"person-5": GREEN}))
    assert b_rec.risk > 0 and b_rec.adverse == 0.0
