import numpy as np
import pytest

from nerva.motor_contract import NeutralPhaseClock, at_rest, canonical_command, motor_clock_tick, planar_tracking


def command(vx=0., vy=0., yaw=0., head=.0):
    return np.array([vx, vy, yaw, 0, 0, head, 0], dtype=np.float32)


def test_rest_boundaries_use_separate_velocity_units_and_keep_head_slots():
    assert at_rest(command(.003, .004, .02))
    assert not at_rest(command(.006))
    assert not at_rest(command(yaw=.025))
    c = canonical_command(command(.005, yaw=.02, head=.4))
    np.testing.assert_array_equal(c[:3], 0)
    assert c[5] == pytest.approx(.4)


def test_rest_onset_and_wraparound_share_one_clock():
    clock = NeutralPhaseClock(3)
    assert clock.tick(command())["index"] == 0
    assert [clock.tick(command(.15))["index"] for _ in range(5)] == [0, 1, 2, 0, 1]
    np.testing.assert_array_equal(clock.tick(command(head=.4))["phase"], [1, 0])
    assert clock.tick(command(yaw=.6))["index"] == 0
    np.testing.assert_array_equal(clock.reset(), [1, 0])


def test_symmetric_tracking_penalises_lateral_drift_at_rest():
    a = planar_tracking(command(), [.05, 0, 0])
    b = planar_tracking(command(), [0, .05, 0])
    assert a == b < planar_tracking(command(), [0, 0, 0])


def test_batched_jit_matches_numpy_and_training_adapter():
    jax = pytest.importorskip("jax")
    import jax.numpy as jp
    from nerva.motor_contract import training_motor_tick
    commands = np.stack([command(), command(.15), command(yaw=.6)])
    expected = motor_clock_tick(np.array([12, 26, 0]), np.array([False, False, True]), commands, 27)
    actual = jax.jit(lambda i, r, c: motor_clock_tick(i, r, c, 27, xp=jp))(
        jp.array([12, 26, 0]), jp.array([False, False, True]), jp.array(commands))
    for a, b in zip(expected, actual):
        np.testing.assert_allclose(a, b, atol=1e-6)
    info = {"motor_phase_index": jp.array(0), "motor_previous_rest": jp.array(True)}
    output = jax.jit(lambda state, c: training_motor_tick(state, c, 27))(info, jp.array(command(.15)))
    assert int(output["motor_phase_index"]) == 0
    np.testing.assert_array_equal(output["imitation_phase"], [1, 0])


def test_malformed_runtime_commands_and_period_rejected():
    with pytest.raises(ValueError):
        NeutralPhaseClock(0)
    with pytest.raises(ValueError):
        NeutralPhaseClock().tick([0, 0, 0])
    with pytest.raises(ValueError):
        NeutralPhaseClock().tick(command(vx=float("nan")))
