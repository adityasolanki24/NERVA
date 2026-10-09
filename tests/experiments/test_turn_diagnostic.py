import numpy as np
import pytest

from experiments.locomotion_curriculum.turn_diagnostic import analyse, design


def synthetic(drift=(0., 0.), rate=.6):
    t = np.arange(1, 1001) * .02
    yaw = rate * t
    xy = (design(t, yaw, True) @ np.array([.2, -.1, -.06, .02, *drift])).reshape(-1, 2)
    return t, xy, yaw


@pytest.mark.parametrize("rate", [.6, -.6])
def test_known_orbit_and_world_drift(rate):
    orbit = analyse(*synthetic(rate=rate))
    assert orbit["classification"] == "bounded_orbit"
    np.testing.assert_allclose(orbit["orbit"]["offset_body_xy_m"], [-.06, .02], atol=1e-10)
    drift = analyse(*synthetic((.03, -.02), rate))
    assert drift["classification"] == "sustained_world_drift"
    np.testing.assert_allclose(drift["orbit_drift"]["drift_world_xy_m_s"], [.03, -.02], atol=1e-10)


def test_nonrotating_geometry_is_inconclusive():
    assert analyse(*synthetic(rate=0))["classification"] == "inconclusive"


def test_bad_trace_rejected():
    t, xy, yaw = synthetic()
    xy[0, 0] = np.nan
    with pytest.raises(ValueError):
        analyse(t, xy, yaw)
