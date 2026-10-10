"""Gait-averaged tracking: the reward must prefer walking the reference gait over standing still."""
from pathlib import Path

import numpy as np
import pytest

from nerva.motor_contract import planar_tracking
from nerva.training.neutral_reference import verified_references
from nerva.training.reference_kinematics import sample_reference

ROOT = Path(__file__).resolve().parents[2]


def test_window_mean_uses_only_filled_rows():
    import jax.numpy as jp

    from nerva.training.neutral_joystick import window_mean
    window = jp.zeros((27, 3)).at[0].set(jp.array([1., 2., 3.])).at[1].set(jp.array([3., 2., 1.]))
    assert np.allclose(np.asarray(window_mean(window, 2)), [2., 2., 2.])
    assert np.allclose(np.asarray(window_mean(window, 0)), 0.)


@pytest.mark.skipif(not (ROOT / "experiments/cloud_runs/neutral-reference-repair").exists(),
                    reason="needs the local reference artifacts")
def test_instantaneous_tracking_prefers_standing_and_gait_average_prefers_walking():
    records, _ = verified_references(ROOT)
    t = np.arange(27 * 40) * .02
    for record in records:
        command = np.asarray(record["command"])
        if not np.any(command[:2]):
            continue
        lin = np.asarray(sample_reference(record["reference"], t % .54)["linear_body"])[:, :2]
        averaged = np.array([lin[max(0, i - 26):i + 1].mean(0) for i in range(len(lin))])[27:]
        cmd = np.r_[command, np.zeros(4)]
        standing = planar_tracking(cmd, np.zeros(2))
        assert np.mean(planar_tracking(cmd, lin)) < standing  # the measured problem
        assert np.mean(planar_tracking(cmd, averaged)) > .99 > standing  # the fix


@pytest.mark.slow
def test_environment_steps_and_autoreset_restores_the_window():
    import os

    import jax
    import jax.numpy as jp

    from nerva.sim.open_duck import OPEN_DUCK_ROOT
    from nerva.training.b2_warm_start import balanced_environment
    from nerva.training.neutral_reference import NeutralReference
    records, _ = verified_references(ROOT)
    cwd = os.getcwd()
    os.chdir(OPEN_DUCK_ROOT / "Open_Duck_Playground")
    try:
        env = balanced_environment(NeutralReference(records), episode_length=3, persistent_command=True,
                                   gait_averaged_tracking=True)
        state = jax.jit(env.reset)(jax.random.split(jax.random.PRNGKey(0), 7))
        step = jax.jit(env.step)
        for _ in range(2):
            state = step(state, jp.zeros((7, 14)))
        assert np.all(np.asarray(state.info["velocity_count"]) == 2)
        state = step(state, jp.zeros((7, 14)))  # episode length 3: autoreset
        assert np.all(np.asarray(state.info["velocity_count"]) == 0)
        assert np.all(np.isfinite(np.asarray(state.reward)))
    finally:
        os.chdir(cwd)
