"""Turn-translation tracking (docs/turn_translation_pilot.md): tighter width for pure-turn commands only."""
import numpy as np
import pytest

pytest.importorskip("jax")


def test_only_the_two_pure_turns_are_selected():
    import jax.numpy as jp

    from nerva.training.neutral_joystick import is_pure_turn
    from nerva.training.neutral_reference import COMMANDS
    flags = [bool(is_pure_turn(jp.asarray(list(c) + [0.] * 4))) for c in COMMANDS]
    assert flags == [False, False, False, False, False, True, True]


def test_tight_width_penalizes_threshold_drift_more():
    from nerva.motor_contract import planar_tracking
    from nerva.training.neutral_joystick import TURN_TRACKING_SIGMA
    cmd = np.r_[0., 0., .6, np.zeros(4)]
    drift = np.array([0., .03])
    assert planar_tracking(cmd, drift) == pytest.approx(.914, abs=1e-3)
    assert planar_tracking(cmd, drift, TURN_TRACKING_SIGMA) == pytest.approx(.698, abs=1e-3)
