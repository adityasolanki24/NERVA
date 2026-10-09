from types import SimpleNamespace

import numpy as np
import pytest

from experiments.locomotion_curriculum.learning_support import remove_zero_style


def test_zero_style_conversion_preserves_affine_input_with_nonzero_style_mean():
    rng = np.random.default_rng(2)
    kernel, bias = rng.normal(size=(8, 12)), rng.normal(size=12)
    mean, std = rng.normal(size=8), rng.uniform(.1, 2, size=8)
    weights = {"params": {"hidden_0": {"kernel": kernel.copy(), "bias": bias.copy()}}}
    result = remove_zero_style(weights, mean, std)["params"]["hidden_0"]
    raw = rng.normal(size=(10, 5))
    expected = ((np.c_[raw, np.zeros((10, 3))] - mean) / std) @ kernel + bias
    np.testing.assert_allclose((raw - mean[:-3]) / std[:-3] @ result["kernel"] + result["bias"], expected)
    np.testing.assert_array_equal(weights["params"]["hidden_0"]["kernel"], kernel)


def test_coverage_uses_canonical_collector_dtype_and_rejects_missing_command():
    from experiments.locomotion_curriculum.neutral_learning import command_coverage
    from nerva.training.neutral_reference import COMMANDS
    commands = np.repeat(np.asarray(COMMANDS, dtype=np.float32)[:, None], 32, axis=1)
    assert command_coverage(commands)
    commands[5] = commands[6]
    assert not command_coverage(commands)


def test_frozen_onnx_export_matches_silu_policy_and_rejects_overwrite(tmp_path):
    jax = pytest.importorskip("jax")
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import networks
    from experiments.locomotion_curriculum.learning_support import SHAPE, export_policy
    import jax.numpy as jp
    net = networks.make_ppo_networks(SHAPE, 14, policy_hidden_layer_sizes=(4, 3, 2),
        value_hidden_layer_sizes=(4, 3, 2), policy_obs_key="state", value_obs_key="privileged_state",
        preprocess_observations_fn=running_statistics.normalize)
    normalizer = running_statistics.init_state({name: jp.zeros(shape) for name, shape in SHAPE.items()})
    normalizer = normalizer.replace(mean={name: jp.ones(shape) * .3 for name, shape in SHAPE.items()},
                                    std={name: jp.ones(shape) * 1.7 for name, shape in SHAPE.items()})
    params = (normalizer, net.policy_network.init(jax.random.PRNGKey(1)), net.value_network.init(jax.random.PRNGKey(2)))
    path = tmp_path / "candidate.onnx"
    export_policy(path, params)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    policy = jax.jit(networks.make_inference_fn(net)(params, deterministic=True))
    for obs in np.random.default_rng(7).normal(size=(5, 101)).astype(np.float32):
        actual = session.run(None, {"obs": obs[None]})[0][0]
        expected = policy({"state": jp.asarray(obs), "privileged_state": jp.zeros(212)}, jax.random.PRNGKey(3))[0]
        np.testing.assert_allclose(actual, expected, atol=1e-6)
    with pytest.raises(FileExistsError):
        export_policy(path, params)


def test_balanced_reset_observations_and_autoreset_keep_each_command(monkeypatch):
    jax = pytest.importorskip("jax")
    pytest.importorskip("mujoco_playground")
    import jax.numpy as jp
    from mujoco_playground._src import mjx_env
    from nerva.training import neutral_joystick
    from nerva.training.neutral_reference import COMMANDS
    from experiments.locomotion_curriculum.learning_support import balanced_environment

    class Toy:
        def __init__(self, reference, task):
            assert task == "flat_terrain_backlash"
            self.SREF = reference

        def _initialize_reference(self, info):
            info["motor_phase_index"] = jp.int32(0)

        def _get_obs(self, data, info, contact):
            del data, contact
            return {"state": jp.zeros(101).at[6:13].set(info["command"])}

        def reset(self, key):
            return mjx_env.State(data=jp.zeros(1), obs={"state": jp.zeros(101)},
                reward=jp.float32(0), done=jp.float32(0), metrics={}, info={"rng": key, "command": jp.zeros(7),
                "motor_phase_index": jp.int32(4), "imitation_i": jp.int32(0), "current_reference_motion": jp.zeros(3)})

        def step(self, state, action):
            del action
            return state.replace(data=state.data + 1, reward=jp.float32(1), done=jp.float32(0))

    monkeypatch.setattr(neutral_joystick, "NeutralJoystick", Toy)
    reference = SimpleNamespace(get_reference_motion=lambda x, y, yaw, phase: jp.array([x, y, yaw]))
    env = balanced_environment(reference, episode_length=2)
    state = jax.jit(env.reset)(jax.random.split(jax.random.PRNGKey(0), 7))
    for _ in range(5):
        np.testing.assert_allclose(state.obs["state"][:, 6:9], COMMANDS, atol=1e-8)
        np.testing.assert_allclose(state.info["command"][:, :3], COMMANDS, atol=1e-8)
        np.testing.assert_allclose(state.info["current_reference_motion"], COMMANDS, atol=1e-8)
        state = jax.jit(env.step)(state, jp.zeros((7, 14)))
    from experiments.locomotion_curriculum.neutral_learning import collect_balanced_batch

    def make_policy(params):
        del params
        return lambda obs, key: (jp.zeros((7, 14)), {"raw_action": jp.zeros((7, 14))})

    _, batch, next_key = jax.jit(lambda s, key: collect_balanced_batch(env, make_policy, (), s, key))(
        state, jax.random.PRNGKey(5))
    assert batch.discount.shape == (7, 32)
    np.testing.assert_allclose(batch.extras["state_extras"]["command"][..., :3],
                               np.repeat(np.asarray(COMMANDS)[:, None], 32, axis=1), atol=1e-8)
    assert not np.array_equal(next_key, jax.random.PRNGKey(5))
