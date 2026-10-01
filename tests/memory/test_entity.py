import numpy as np

from nerva.memory.entity import EntityMemory

BLUE = np.array([0, 0, 0, 0, 0, 1.0, 0, 0])
GREEN = np.array([0, 0, 1.0, 0, 0, 0, 0, 0])


def test_identity_by_appearance():
    m = EntityMemory()
    a = m.resolve("person", BLUE, 0.0)
    b = m.resolve("person", GREEN, 1.0)
    assert a.eid != b.eid
    assert m.resolve("person", BLUE * 0.9 + GREEN * 0.1, 2.0).eid == a.eid
    assert m.resolve("ball", None, 3.0).eid == "ball#0"


def test_familiarity_grows_with_exposure_and_fades_in_absence():
    m = EntityMemory()
    a = m.resolve("person", BLUE, 0.0)
    fresh = a.novelty()
    for i in range(300):
        m.observe(a, i * 0.1, 0.1)
    assert a.novelty() < 0.5 * fresh
    before = a.exposure_s
    m.resolve("person", BLUE, 30.0 + 3600.0)  # an hour later
    assert a.exposure_s < 0.01 * before and a.encounters == 2


def test_a_fright_is_learned_fast_and_extinguished_gradually():
    m = EntityMemory()
    a = m.resolve("person", BLUE, 0.0)
    m.learn(a, 1.0, "person_approaching_rapidly", [("fear", 0.6), ("surprise", 0.8)], arousal=0.4,
            surprise_negative=True)
    after_fright = a.threat
    assert after_fright > 0.3 and a.trust < 0.5 and a.warmth < 0
    for k in range(5):  # gentle encounters
        m.learn(a, 10.0 + k, "person_approaching_slowly", [("hope", 0.1)], arousal=0.0, surprise_negative=False)
    assert 0 <= a.threat < 0.1 * after_fright and a.trust > 0.5


def test_low_confidence_attribution_learns_less():
    m = EntityMemory()
    sure, unsure = m.resolve("person", BLUE, 0.0), m.resolve("person", GREEN, 0.0)
    for rec, conf in ((sure, 1.0), (unsure, 0.3)):
        m.learn(rec, 1.0, "touch_gentle", [("joy", 0.5)], arousal=0.1, surprise_negative=False, confidence=conf)
    assert sure.warmth > unsure.warmth > 0


def test_ambiguous_appearance_creates_no_new_identity():
    m = EntityMemory()
    m.resolve("person", BLUE, 0.0)
    assert m.resolve("person", 0.6 * BLUE + 0.4 * GREEN, 1.0) is None  # partial/mixed view
    assert len(m.records) == 1
