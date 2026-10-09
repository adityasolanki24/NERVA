import numpy as np
import pytest

pytest.importorskip("jax")

from nerva.training.parameter_checkpoint import leaf_comparison, require_contract, save_parameters, tree_finite, write_contract  # noqa: E402


def test_leaf_comparison_handles_serialized_list_tuple_and_rejects_shape_dtype_and_nonfinite():
    tree = (np.array([1, 2], dtype=np.float32), {"count": np.int32(16)})
    assert leaf_comparison(tree, list(tree))["equal"]
    assert not leaf_comparison(tree, [tree[0], {"other": np.int32(16)}])["equal"]
    assert not leaf_comparison(np.array([0.], dtype=np.float32), np.array([-0.], dtype=np.float32))["equal"]
    assert not leaf_comparison(tree, [np.array([[1, 2]], dtype=np.float32), tree[1]])["equal"]
    assert not leaf_comparison(tree, [np.array([1, 2], dtype=np.float64), tree[1]])["equal"]
    bad = [np.array([1, np.nan], dtype=np.float32), tree[1]]
    assert not tree_finite(bad) and not leaf_comparison(tree, bad)["finite"]


def test_contract_requires_exact_reference_normalization_and_shape_match(tmp_path):
    contract = {"contract": "neutral_motor_v1", "shape": (101, 212), "normalization": True,
                "references": ["sha256-example"]}
    write_contract(tmp_path, contract)
    require_contract(tmp_path, contract)
    with pytest.raises(ValueError, match="mismatch"):
        require_contract(tmp_path, {**contract, "normalization": False})
    with pytest.raises(FileExistsError):
        write_contract(tmp_path, contract)


@pytest.mark.parametrize("variance_eps", [0., .0001])
def test_pinned_brax_checkpoint_roundtrip_and_inference_match(tmp_path, variance_eps):
    import functools
    import jax
    import jax.numpy as jp
    from brax.training.acme import running_statistics, specs
    from brax.training.agents.ppo import checkpoint, networks
    from experiments.locomotion_curriculum.neutral_ppo_smoke import NETWORK_CONFIG, action_parity

    factory = functools.partial(networks.make_ppo_networks, **NETWORK_CONFIG)
    shape = {"state": (101,), "privileged_state": (212,)}
    normalizer = running_statistics.init_state({key: specs.Array(value, jp.float32) for key, value in shape.items()},
                                             std_eps=variance_eps)
    network = factory(shape, 14, preprocess_observations_fn=running_statistics.normalize)
    keys = jax.random.split(jax.random.PRNGKey(7), 2)
    params = (normalizer, network.policy_network.init(keys[0]), network.value_network.init(keys[1]))
    # Signed zero must survive the actual pinned serialization API, not just a comparison helper.
    params[0].mean["state"] = params[0].mean["state"].at[0].set(-0.)
    save_parameters(tmp_path, 16, params, checkpoint.network_config(shape, 14, True, factory))
    location = tmp_path / "000000000016"
    restored = checkpoint.load(location)
    assert leaf_comparison(params, restored)["equal"]
    batch = {key: jp.zeros((4,) + value, dtype=jp.float32) for key, value in shape.items()}
    # Training updates run under JIT, which places Brax's restored host arrays on device.
    update = jax.jit(running_statistics.update)
    live_update = update(params[0], batch)
    restored_update = update(restored[0], batch)
    assert leaf_comparison(live_update, restored_update)["equal"]
    expected_std = max(1e-6, variance_eps**.5)
    for std in restored_update.std.values():
        np.testing.assert_allclose(std, expected_std, rtol=1e-6)
    parity = action_parity(networks.make_inference_fn(network), params,
                           lambda deterministic: checkpoint.load_policy(location, deterministic=deterministic))
    assert parity["all_pass"]
