"""Stage D: outcome-grounded learning (docs/architecture.md section 1.3)."""

import numpy as np

from nerva.interfaces import OutcomeSignal, RiskEstimate
from nerva.memory.entity import EntityMemory
from nerva.memory.episodic import EpisodicMemory
from nerva.memory.spatial import PlaceMemory

BLUE = np.array([0, 0, 0, 0, 0, 1.0, 0, 0])
FEAR = [("fear", 0.6), ("surprise", 0.8)]


def _feared(mode: str):
    m = EntityMemory(learning=mode)
    a = m.resolve("person", BLUE, 0.0)
    if mode == "grounded":
        m.learn_risk(a, RiskEstimate("collision_risk", 1.0, 1.0, source="person-0"))
    else:
        m.learn(a, 1.0, "person_approaching_rapidly", FEAR, 0.6, True)
    return m, a


def test_mere_reappearance_does_not_grow_the_adverse_association_when_grounded():
    """A remembered threat re-elicits fear on every sighting; grounded memory must not learn from that."""
    m, a = _feared("grounded")
    before = a.threat
    for k in range(10):  # ten sightings, each eliciting fear again, no new adverse outcome
        m.learn(a, 10.0 + k, "person_appeared", [("fear", 0.9)], 0.8, False)
    assert a.threat == before and a.risk == before and a.adverse == 0.0
    assert len(a.episodes) == 1 + 10  # the sightings are still recorded as history


def test_the_legacy_baseline_does_learn_from_re_elicited_fear():
    m, a = _feared("legacy")
    before = a.threat
    for k in range(10):
        m.learn(a, 10.0 + k, "person_appeared", [("fear", 0.9)], 0.8, False)
    assert a.threat > before  # the self-reinforcing loop the grounded path removes


def test_a_real_adverse_outcome_strengthens_and_benign_outcomes_weaken_it():
    m, a = _feared("grounded")
    m.learn_outcome(a, OutcomeSignal("contact_impact", 1.0, 5.0))  # an actual hard contact
    assert a.adverse == 0.5 and a.risk == 0.5  # outcome and near miss are kept apart
    high = a.threat
    for k in range(5):
        m.learn_outcome(a, OutcomeSignal("benign_contact", 1.0, 10.0 + k))
    assert a.threat < high and a.risk < 0.5 and a.benign > 0.5 and a.warmth > -0.5
    assert a.trust == 0.5  # trust is not defined in grounded mode: surprise is not untrustworthiness


def test_legacy_memory_ignores_outcomes():
    m, a = _feared("legacy")
    snapshot = (a.threat, a.warmth, a.trust, a.adverse, a.risk)
    m.learn_outcome(a, OutcomeSignal("contact_impact", 1.0, 5.0))
    m.learn_risk(a, RiskEstimate("collision_risk", 1.0, 5.0))
    assert (a.threat, a.warmth, a.trust, a.adverse, a.risk) == snapshot


def test_sleep_replay_restores_after_drift_but_adds_no_evidence():
    m, a = _feared("grounded")
    learned = a.risk
    epi = EpisodicMemory()
    epi.encode_risk(1.0, RiskEstimate("collision_risk", 1.0, 1.0), a.eid, (0, 0))
    for k in range(20):  # many sleeps: one lunge must stay one lunge
        epi.consolidate(50.0 + 10 * k, m)
    assert abs(a.risk - learned) < 1e-12
    a.last_seen_t = 2.0
    m.resolve("person", BLUE, 2.0 + 1800.0)  # long absence: drift
    drifted = a.risk
    assert drifted < learned
    epi.consolidate(2000.0, m)
    assert drifted < a.risk <= learned


def test_place_memory_grounded_learns_only_from_outcomes():
    pm = PlaceMemory(learning="grounded")
    pm.learn((0.1, 0.1), 1.0, FEAR, 0.8)  # emotions alone: ignored
    assert pm.threat(pm.key((0.1, 0.1))) == 0.0
    pm.learn_risk((0.1, 0.1), 2.0, RiskEstimate("collision_risk", 1.0, 2.0))
    assert pm.threat(pm.key((0.1, 0.1))) > 0.0
