"""Consolidation step 2: target-conditioned arbitration."""

from nerva.behaviour.selection import UtilityBehaviour
from nerva.interfaces import ActionTendencyState, PADState, Track

A = Track("person", bearing=0.6, distance=2.6, tid="person-0")
B = Track("person", bearing=-0.3, distance=2.4, tid="person-1")
TOTAL = ActionTendencyState(approach=0.4, avoid=0.2)  # liking for B and fear of A, summed


def run(beh, directed, seconds=2.0):
    d = None
    for i in range(int(seconds / 0.1)):
        d = beh.step(5.0 + 0.1 * i, 0.1, PADState(), TOTAL, (A, B), {"person-0": 0.0, "person-1": 0.0},
                     None, directed)
    return d


def test_fear_of_one_person_no_longer_blocks_approaching_another():
    directed = {"": ActionTendencyState(), "person-0": ActionTendencyState(avoid=0.2),
                "person-1": ActionTendencyState(approach=0.4)}
    d = run(UtilityBehaviour(), directed)
    assert d.mode == "approach" and d.target == "person-1"


def test_without_directed_tendencies_the_global_gate_still_blocks_everyone():
    d = run(UtilityBehaviour(), None)
    assert d.mode not in ("approach", "inspect")  # legacy behaviour: (1 - 4·0.2)·0.4 < watch utility


def test_the_feared_person_is_the_focal_one_even_if_further_away():
    directed = {"": ActionTendencyState(), "person-0": ActionTendencyState(avoid=0.5)}
    beh = UtilityBehaviour()
    run(beh, directed, 0.5)
    assert beh._focal_person({"person-0": A, "person-1": B}) == "person-0"
