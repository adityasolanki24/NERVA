from experiments.locomotion_curriculum.identity_multibatch import update_screen


def test_per_update_screen_rejects_bad_deployed_policy_despite_self_kl_and_separates_value_and_control_errors():
    pre = {"kl_mean": .00014}
    post = {"kl_mean": .1, "v_loss": .1}
    errors = {"logits": 1e-6, "log_prob": 1e-5}
    assert all(update_screen(pre, post, errors).values())
    assert not update_screen(pre, {**post, "kl_mean": 1.01}, errors)["deployed_kl"]
    assert not update_screen(pre, {**post, "v_loss": 101}, errors)["deployed_value_loss"]
    assert not update_screen(pre, post, {**errors, "logits": 1e-4})["control_logits"]
    assert not update_screen({"kl_mean": .01}, post, errors)["control_kl"]
