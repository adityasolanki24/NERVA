import numpy as np

from nerva.world.self_state import RISK_TILT_FULL, RISK_TILT_START, estimate_self_state, stability_risk


def _qpos(tilt_deg):
    a = np.radians(tilt_deg) / 2
    q = np.zeros(7)
    q[3:7] = (np.cos(a), np.sin(a), 0.0, 0.0)  # rotation about x
    return q


def test_tilt_and_risk_from_the_base_state():
    s = estimate_self_state(1.0, _qpos(5.0), np.zeros(6), "explore")
    assert abs(s.tilt_deg - 5.0) < 1e-9 and s.stability_risk == 0.0 and s.locomotion_mode == "explore"
    assert s.feet_contact is None and s.escape_room is None  # not measured / not computed: not invented
    mid = (RISK_TILT_START + RISK_TILT_FULL) / 2
    assert abs(estimate_self_state(1.0, _qpos(mid), np.zeros(6)).stability_risk - 0.5) < 1e-9


def test_fast_rotation_raises_risk_even_when_upright():
    assert stability_risk(0.0, 10.0) == 1.0 and stability_risk(0.0, 0.5) == 0.0
