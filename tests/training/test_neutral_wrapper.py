import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("mujoco_playground")
import jax.numpy as jp  # noqa: E402
from brax.envs.wrappers import training  # noqa: E402
from mujoco_playground._src import mjx_env  # noqa: E402

from nerva.training.neutral_wrapper import NeutralAutoResetWrapper  # noqa: E402


class Toy:
    def reset(self, key):
        return mjx_env.State(data=jp.array([0.]), obs={"state": jp.array([0.])},
                             reward=jp.float32(0), done=jp.float32(0), metrics={},
                             info={"rng": key, "motor_phase_index": jp.int32(0), "last_act": jp.zeros(2),
                                   "command": jp.array([.074, 0, 0, 0, 0, 0, 0]),
                                   "episode_metrics": {"length": jp.float32(0)}, "steps": jp.float32(0)})

    def step(self, state, action):
        info = {**state.info, "motor_phase_index": state.info["motor_phase_index"] + 1,
                "last_act": action, "rng": jax.random.split(state.info["rng"])[0],
                "steps": state.info["steps"] + 1, "episode_metrics": {"length": jp.float32(1)}}
        return state.replace(data=state.data + 1, obs={"state": state.obs["state"] + 1},
                             done=(action[0] > 0).astype(jp.float32), info=info)


def test_mixed_autoreset_restores_owned_info_with_obs_but_preserves_rng_and_accounting():
    env = NeutralAutoResetWrapper(training.VmapWrapper(Toy()))
    keys = jax.random.split(jax.random.PRNGKey(7), 2)
    state = jax.jit(env.reset)(keys)
    result = jax.jit(env.step)(state, jp.array([[1., .3], [0., .4]]))
    np.testing.assert_array_equal(result.info["motor_phase_index"], [0, 1])
    np.testing.assert_array_equal(result.data, [[0], [1]])
    np.testing.assert_array_equal(result.obs["state"], [[0], [1]])
    np.testing.assert_allclose(result.info["last_act"], [[0, 0], [0, .4]])
    np.testing.assert_array_equal(result.info["steps"], [1, 1])
    np.testing.assert_array_equal(result.info["episode_metrics"]["length"], [1, 1])
    assert not np.array_equal(keys[0], result.info["rng"][0])
    # The snapshot stays intact when multiple steps overwrite current motor info.
    again = jax.jit(env.step)(result, jp.array([[0., .2], [1., .4]]))
    np.testing.assert_array_equal(again.info["motor_phase_index"], [1, 0])
