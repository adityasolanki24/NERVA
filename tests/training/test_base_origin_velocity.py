"""Base-origin reward velocity: exact rigid-body shift of the IMU velocimeter, and the reference turn no longer
pays a tracking penalty for pivoting about the base (results_turn_pivot/)."""
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
mujoco = pytest.importorskip("mujoco")


def test_shift_matches_free_joint_velocity_in_native_mujoco():
    import jax.numpy as jp

    from nerva.sim.open_duck import SCENE_BACKLASH
    from nerva.training.neutral_joystick import base_origin_velocity, imu_offset
    model = mujoco.MjModel.from_xml_path(str(SCENE_BACKLASH))
    data = mujoco.MjData(model)
    offset = imu_offset(model)
    assert np.allclose(offset, [-.08, 0., .05])
    rng = np.random.default_rng(0)
    for _ in range(20):
        mujoco.mj_resetDataKeyframe(model, data, 0)
        quat = rng.normal(size=4)
        data.qpos[3:7] = quat / np.linalg.norm(quat)
        data.qvel[:] = rng.normal(size=model.nv)
        mujoco.mj_forward(model, data)
        imu = data.sensor("local_linvel").data.copy()
        gyro = data.sensor("gyro").data.copy()
        rotation = data.xmat[model.body("base").id].reshape(3, 3)
        expected = rotation.T @ data.qvel[:3]
        assert np.allclose(np.asarray(base_origin_velocity(jp.asarray(imu), jp.asarray(gyro), offset)), expected,
                           atol=1e-5)


@pytest.mark.skipif(not (ROOT / "experiments/cloud_runs/neutral-reference-repair").exists(),
                    reason="needs the local reference artifacts")
def test_reference_turn_scores_full_tracking_only_at_the_base_origin():
    from nerva.motor_contract import planar_tracking
    from nerva.training.neutral_reference import verified_references
    from nerva.training.reference_kinematics import sample_reference
    records, _ = verified_references(ROOT)
    t = np.arange(27 * 40) * .02
    offset = np.array([-.08, 0., .05])
    for record in records:
        command = np.asarray(record["command"])
        if not command[2]:
            continue
        sample = sample_reference(record["reference"], t % .54)
        base = np.asarray(sample["linear_body"])
        imu = base + np.cross(np.asarray(sample["angular_body"]), offset)
        cmd = np.r_[command, np.zeros(4)]

        def averaged(v):
            return np.array([v[max(0, i - 26):i + 1, :2].mean(0) for i in range(len(v))])[27:]
        at_base = np.mean(planar_tracking(cmd, averaged(base)))
        at_imu = np.mean(planar_tracking(cmd, averaged(imu)))
        assert at_base > .99 and at_imu < .85  # the IMU-site reward penalizes the reference's own turn


@pytest.mark.slow
def test_environment_reward_velocity_is_the_base_origin_velocity():
    import os

    import jax
    import jax.numpy as jp

    from nerva.sim.open_duck import OPEN_DUCK_ROOT
    from nerva.training.b2_warm_start import balanced_environment
    from nerva.training.neutral_reference import NeutralReference, verified_references
    records, _ = verified_references(ROOT)
    cwd = os.getcwd()
    os.chdir(OPEN_DUCK_ROOT / "Open_Duck_Playground")
    try:
        with pytest.raises(ValueError):
            balanced_environment(NeutralReference(records), persistent_command=True, base_origin_velocity=True)
        env = balanced_environment(NeutralReference(records), episode_length=3, persistent_command=True,
                                   gait_averaged_tracking=True, base_origin_velocity=True)
        inner = env.env.env.env
        state = jax.jit(env.reset)(jax.random.split(jax.random.PRNGKey(0), 7))
        step = jax.jit(env.step)
        for _ in range(2):
            state = step(state, jp.zeros((7, 14)))
        assert np.all(np.isfinite(np.asarray(state.reward)))
        # MJX sensors are evaluated before the last integration substep, so compare with the shifted sensor
        # reading (the shift itself is checked exactly against native MuJoCo above).
        data = state.data
        imu, gyro = jax.vmap(inner.get_local_linvel)(data), jax.vmap(inner.get_gyro)(data)
        expected = imu - jp.cross(gyro, jp.asarray([-.08, 0., .05]))
        actual = jax.vmap(inner.reward_linvel)(data)
        assert np.allclose(np.asarray(actual), np.asarray(expected), atol=1e-6)
        assert not np.allclose(np.asarray(actual), np.asarray(imu), atol=1e-4)
    finally:
        os.chdir(cwd)
