from nerva.behaviour.selection import UtilityBehaviour
from nerva.interfaces import ActionTendencyState, PADState, Track


def test_attention_biases_the_utility_selector_toward_its_visual_target():
    ball = Track("ball", 0.0, 1.4, tid="ball-0")
    person = Track("person", 0.1, 1.4, tid="person-0")
    beh = UtilityBehaviour()
    decision = None
    for i in range(15):
        decision = beh.step(5.0 + i * 0.1, 0.1, PADState(), ActionTendencyState(explore=0.5, approach=0.1),
                            (ball, person), {"ball-0": 1.0, "person-0": 1.0}, {}, None, "ball-0", "novel")
    assert decision.target == "ball-0"


def test_attention_never_overrides_a_retreat_choice():
    ball = Track("ball", 0.0, 1.4, tid="ball-0")
    person = Track("person", 0.1, 0.5, tid="person-0")
    beh = UtilityBehaviour()
    decision = beh.step(5.0, 0.1, PADState(), ActionTendencyState(explore=1.0, avoid=0.8),
                        (ball, person), {"ball-0": 1.0, "person-0": 0.1}, {}, None, "ball-0", "novel")
    assert decision.mode in ("freeze", "retreat") and decision.target == "person-0"


def test_grounded_positive_affinity_adds_one_stationary_greeting():
    person = Track("person", 0.1, 1.2, tid="person-0")
    beh = UtilityBehaviour()
    beh.notice(5.0, "person_appeared", "person-0")
    d = beh.step(5.1, 0.1, PADState(), ActionTendencyState(), (person,), {"person-0": 0.0}, {}, None,
                 "person-0", "familiar", {"person-0": 0.6})
    assert d.mode == "greet" and d.target == "person-0"
    # The acknowledgement is finite, then cannot retrigger for this appearance.
    d = beh.step(6.1, 0.1, PADState(), ActionTendencyState(), (person,), {"person-0": 0.0}, {}, None,
                 "person-0", "familiar", {"person-0": 0.6})
    assert d.mode != "greet"


def test_social_affinity_cannot_override_a_nearby_threat():
    person = Track("person", 0.1, 0.5, tid="person-0")
    d = UtilityBehaviour().step(5.0, 0.1, PADState(), ActionTendencyState(avoid=0.8), (person,),
                                {"person-0": 0.1}, {"person-0": 0.9}, None, "person-0", "unexpected",
                                {"person-0": 1.0})
    assert d.mode in ("freeze", "retreat")
