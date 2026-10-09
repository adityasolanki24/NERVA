import functools

import numpy as np
import pytest

from experiments.locomotion_curriculum.identity_preprocessing import preprocessing_invariance, screen


def test_identity_checkpoint_ignores_saved_statistics_and_contract_rejects_normalized_load(tmp_path):
    jax = pytest.importorskip("jax")
    import jax.numpy as jp
    from brax.training import types
    from brax.training.acme import running_statistics, specs
    from brax.training.agents.ppo import checkpoint, networks
    from nerva.training.parameter_checkpoint import require_contract, save_parameters, write_contract
    factory = functools.partial(networks.make_ppo_networks, policy_hidden_layer_sizes=(8,),
                                value_hidden_layer_sizes=(8,), policy_obs_key="state", value_obs_key="privileged_state")
    shape = {"state": (3,), "privileged_state": (4,)}
    net = factory(shape, 2, preprocess_observations_fn=types.identity_observation_preprocessor)
    norm = running_statistics.init_state({key: specs.Array(size, jp.float32) for key, size in shape.items()}, std_eps=.0001)
    params = (norm, net.policy_network.init(jax.random.PRNGKey(1)), net.value_network.init(jax.random.PRNGKey(2)))
    assert preprocessing_invariance(net, params)["all_pass"]
    active_net = factory(shape, 2, preprocess_observations_fn=running_statistics.normalize)
    assert not preprocessing_invariance(active_net, params)["all_pass"]
    save_parameters(tmp_path, 1, params, checkpoint.network_config(shape, 2, False, factory))
    location = tmp_path / "000000000001"
    write_contract(location, {"normalize_observations": False})
    require_contract(location, {"normalize_observations": False})
    with pytest.raises(ValueError, match="contract mismatch"):
        require_contract(location, {"normalize_observations": True})
    before = networks.make_inference_fn(net)(params, deterministic=True)
    after = checkpoint.load_policy(location, deterministic=True)
    obs = {key: np.full(size, .3, np.float32) for key, size in shape.items()}
    a, _ = before(obs, jax.random.PRNGKey(3))
    b, _ = after(obs, jax.random.PRNGKey(3))
    np.testing.assert_array_equal(a, b)


def test_identity_screen_requires_deployed_kl_and_invariance_despite_passing_epoch_metrics():
    diagnostics = {"metrics": {"kl_mean": .1, "v_loss": .1}, "manual_kl_agreement": True,
                   "manual_log_prob_agreement": True, "preprocessing_invariance": {"all_pass": True}}
    stage = {"all_pass": True, "metrics": {"training/kl_mean": .1, "training/v_loss": .1},
             "normalization_diagnostics": diagnostics}
    assert all(screen(stage).values())
    changed = {**stage, "normalization_diagnostics": {**diagnostics, "metrics": {"kl_mean": 1.1, "v_loss": .1}}}
    assert not all(screen(changed).values())
    changed = {**stage, "normalization_diagnostics": {**diagnostics, "preprocessing_invariance": {"all_pass": False}}}
    assert not all(screen(changed).values())
