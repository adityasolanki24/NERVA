import numpy as np
import pytest

from experiments.locomotion_curriculum.normalization_timing import (
    archive, decide, fingerprint, gaussian_kl, gaussian_log_prob,
)


def test_independent_gaussian_kl_orientation_and_brax_self_offset():
    zeros, ones = np.zeros((2, 14)), np.ones((2, 14))
    np.testing.assert_allclose(gaussian_kl(zeros, ones, zeros, ones), 0, atol=1e-14)
    np.testing.assert_allclose(gaussian_kl(zeros, ones, zeros, ones, 1e-5), 14 * np.log(1.00001), atol=1e-14)
    assert gaussian_kl([0.], [1.], [2.], [1.]) == 2.
    assert np.isclose(gaussian_kl([0.], [1.], [0.], [2.]), np.log(2) + .125 - .5)
    assert not np.isclose(gaussian_kl([0.], [2.], [0.], [1.]), np.log(2) + .125 - .5)
    with pytest.raises(ValueError, match="positive"):
        gaussian_kl([0.], [0.], [0.], [1.])


def test_timing_support_requires_valid_fixed_weight_control_and_threshold():
    integrity = {"fixed_weights": True, "fixed_batch": True}
    assert decide(integrity, .00014, 10.)["hypothesis_supported"]
    assert not decide(integrity, .00014, 1.)["hypothesis_supported"]
    assert not decide({**integrity, "fixed_weights": False}, .00014, 10.)["hypothesis_supported"]
    assert not decide(integrity, .01, 10.)["integrity_pass"]
    assert not decide(integrity, .00014, float("inf"))["integrity_pass"]


def test_log_probability_includes_tanh_jacobian_and_handles_large_raw_actions():
    # Standard normal at zero; two dimensions and unit tanh derivative.
    np.testing.assert_allclose(gaussian_log_prob([0., 0.], [1., 1.], [0., 0.]), -np.log(2 * np.pi))
    assert np.isfinite(gaussian_log_prob([0.], [1.], [1000.]))
    with pytest.raises(ValueError, match="positive"):
        gaussian_log_prob([0.], [0.], [0.])


def test_offline_batch_archive_admission_checks_schema_hashes_and_signed_zero(tmp_path):
    pytest.importorskip("jax")
    from brax.training.types import Transition
    from experiments.locomotion_curriculum.normalization_timing_offline import load_batch, require_archive
    data = Transition(observation={"state": np.zeros((2, 4, 3), np.float32)},
                      next_observation={"state": np.ones((2, 4, 3), np.float32)},
                      action=np.zeros((2, 4, 2), np.float32), reward=np.zeros((2, 4), np.float32),
                      discount=np.ones((2, 4), np.float32),
                      extras={"state_extras": {"episode_metrics": {"cost/action_rate": np.zeros((2, 4), np.float32)}}})
    path = tmp_path / "batch.npz"
    metadata = archive(path, data)
    restored = load_batch(path)
    require_archive(path, metadata, restored)
    assert fingerprint(restored) == fingerprint(data)
    changed = restored._replace(reward=np.full((2, 4), -0., np.float32))
    with pytest.raises(ValueError, match="tree hash"):
        require_archive(path, metadata, changed)
    with pytest.raises(ValueError, match="file hash"):
        require_archive(path, {**metadata, "file_sha256": "invalid"}, restored)
    np.savez(tmp_path / "unsafe.npz", **{"['bad-name']": np.zeros(1)})
    with pytest.raises(ValueError, match="unsupported"):
        load_batch(tmp_path / "unsafe.npz")


def test_compiled_joint_replay_respects_dynamic_raw_actions_and_matches_behavior():
    jax = pytest.importorskip("jax")
    import jax.numpy as jp
    from brax.training.types import Transition
    from brax.training.acme import running_statistics, specs
    from brax.training.agents.ppo import losses, networks
    from experiments.locomotion_curriculum.normalization_timing import make_replay
    net = networks.make_ppo_networks({"state": (3,), "privileged_state": (4,)}, 2,
                                    preprocess_observations_fn=running_statistics.normalize,
                                    policy_hidden_layer_sizes=(8,), value_hidden_layer_sizes=(8,),
                                    policy_obs_key="state", value_obs_key="privileged_state")
    norm = running_statistics.init_state({"state": specs.Array((3,), jp.float32),
                                         "privileged_state": specs.Array((4,), jp.float32)}, std_eps=.0001)
    weights = losses.PPONetworkParams(policy=net.policy_network.init(jax.random.PRNGKey(1)),
                                     value=net.value_network.init(jax.random.PRNGKey(2)))
    obs = {"state": jp.arange(24, dtype=jp.float32).reshape(2, 4, 3) / 20,
           "privileged_state": jp.arange(32, dtype=jp.float32).reshape(2, 4, 4) / 20}
    logits = net.policy_network.apply(norm, weights.policy, obs)
    replay = make_replay(net)
    outputs = []
    for shift in (0., .3):
        raw = jp.linspace(-1, 1, 16).reshape(2, 4, 2) + shift
        stored_lp = net.parametric_action_distribution.log_prob(logits, raw)
        batch = Transition(observation=obs, next_observation=obs, action=jp.tanh(raw),
                           reward=jp.ones((2, 4)), discount=jp.ones((2, 4)),
                           extras={"policy_extras": {"raw_action": raw, "log_prob": stored_lp,
                                                     "distribution_params": logits},
                                   "state_extras": {"truncation": jp.zeros((2, 4))}})
        result = replay(norm, weights, batch, jax.random.PRNGKey(3))
        np.testing.assert_allclose(result["log_prob"], np.swapaxes(stored_lp, 0, 1), atol=1e-4, rtol=1e-5)
        np.testing.assert_allclose(result["log_prob"], gaussian_log_prob(
            result["loc"], result["scale"], np.swapaxes(raw, 0, 1)), atol=1e-4, rtol=1e-5)
        assert float(result["metrics"]["kl_mean"]) <= .001
        outputs.append(np.asarray(result["log_prob"]))
    assert not np.allclose(*outputs)


def test_fingerprints_cover_names_dtype_shape_and_literal_bytes_and_archive_refuses_overwrite(tmp_path):
    pytest.importorskip("jax")
    pytest.importorskip("flax")
    original = {"observation": np.array([0., 1.], dtype=np.float32)}
    digest = fingerprint(original)
    for other in ({"other": original["observation"]}, {"observation": original["observation"].astype(np.float64)},
                  {"observation": original["observation"].reshape(1, 2)},
                  {"observation": np.array([-0., 1.], dtype=np.float32)}):
        assert fingerprint(other) != digest
    path = tmp_path / "batch.npz"
    metadata = archive(path, original)
    assert metadata["tree_sha256"] == digest
    with np.load(path, allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["['observation']"], original["observation"])
    with pytest.raises(FileExistsError):
        archive(path, original)


def test_collector_preserves_batch_time_alignment_and_updates_exactly_eight_observations():
    jax = pytest.importorskip("jax")
    pytest.importorskip("mujoco_playground")
    import jax.numpy as jp
    from brax.training.acme import running_statistics, specs
    from mujoco_playground._src import mjx_env
    from experiments.locomotion_curriculum.normalization_timing import first_batch_keys, make_collector
    from nerva.training.neutral_wrapper import wrap_neutral_for_training
    from nerva.training.parameter_checkpoint import count_value

    class Toy:
        action_size = 2

        def reset(self, key):
            first = (key[0] % 100).astype(jp.float32)
            return mjx_env.State(data=jp.zeros(1), obs={"state": jp.array([first, 0., 0.]),
                                "privileged_state": jp.array([first, 0., 0., 0.])}, reward=jp.float32(0),
                                 done=jp.float32(0), metrics={}, info={"rng": key, "tick": jp.int32(0)})

        def step(self, state, action):
            del action
            obs = {key: value.at[1].add(1) for key, value in state.obs.items()}
            return state.replace(obs=obs, reward=jp.float32(1), info={**state.info, "tick": state.info["tick"] + 1})

    def make_policy(params):
        del params

        def policy(obs, key):
            raw = jax.random.normal(key, (obs["state"].shape[0], 2)) * .1
            return jp.tanh(raw), {"raw_action": raw, "log_prob": jp.zeros(raw.shape[0]),
                                  "distribution_params": jp.zeros(raw.shape[:-1] + (4,))}

        return policy

    keys = first_batch_keys()
    env = wrap_neutral_for_training(Toy())
    state = jax.pmap(env.reset, axis_name="i")(keys["reset"])
    normalizer = running_statistics.init_state({"state": specs.Array((3,), jp.float32),
                                               "privileged_state": specs.Array((4,), jp.float32)}, std_eps=.0001)
    data, updated = make_collector(env, make_policy, "i")(
        jax.device_put_replicated((normalizer, jp.float32(0), jp.float32(0)), jax.local_devices()), state, keys["epoch"])
    data, updated = jax.tree.map(lambda x: x[0], (data, updated))
    assert data.discount.shape == (2, 4)
    assert count_value(updated.count) == 8
    np.testing.assert_array_equal(data.observation["state"][:, :, 1], [[0, 1, 2, 3]] * 2)
    np.testing.assert_array_equal(data.observation["state"][:, :, 0],
                                  np.repeat(np.asarray(keys["reset"])[0, :, :1] % 100, 4, axis=1))
    np.testing.assert_allclose(updated.mean["state"][1], 1.5)
    np.testing.assert_allclose(updated.std["state"][1], np.sqrt(1.25 + .0001), rtol=1e-6)
