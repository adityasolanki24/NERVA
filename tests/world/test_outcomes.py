from nerva.interfaces import Track
from nerva.world.outcomes import NEAR_FALL_TILT_DEG, OutcomeMonitor


def test_a_fast_close_approach_is_a_collision_risk_once_until_rearmed():
    mon = OutcomeMonitor()
    lunge = (Track("person", 0.0, 0.55, approach_speed=1.5, tid="person-0"),)
    out = mon.proximity(1.0, lunge)
    assert [o.kind for o in out] == ["collision_risk"] and out[0].source == "person-0" and out[0].magnitude == 1.0
    assert type(out[0]).__name__ == "RiskEstimate"  # an estimate, not an outcome
    assert mon.proximity(1.1, lunge) == []  # once per encounter
    mon.proximity(2.0, (Track("person", 0.0, 2.0, tid="person-0"),))  # walks away: re-armed
    assert len(mon.proximity(3.0, lunge)) == 1


def test_slow_approaches_and_the_robots_own_closing_are_not_collisions():
    mon = OutcomeMonitor()
    slow = (Track("person", 0.0, 0.5, approach_speed=0.25, tid="person-1"),)  # time to contact 2 s
    standing = (Track("person", 0.0, 0.5, approach_speed=0.0, tid="person-2"),)  # ego-motion removed
    assert mon.proximity(1.0, slow) == [] and mon.proximity(1.0, standing) == []


def test_stability_loss_from_tilt_rearms_when_upright():
    mon = OutcomeMonitor()
    assert mon.stability(0.0, NEAR_FALL_TILT_DEG + 5)[0].kind == "stability_loss"
    assert mon.stability(0.1, NEAR_FALL_TILT_DEG + 6) == []
    mon.stability(0.2, 2.0)
    assert len(mon.stability(0.3, NEAR_FALL_TILT_DEG + 1)) == 1


def test_contact_outcomes_are_benign_and_attributed():
    o = OutcomeMonitor.contact(5.0, "person-1")
    assert o.kind == "benign_contact" and not o.adverse and o.source == "person-1"
