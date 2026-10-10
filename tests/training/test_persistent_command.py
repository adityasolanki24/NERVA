"""Upstream resamples commands once step > 500; balanced long-episode curricula must keep theirs."""

from nerva.training.neutral_joystick import PersistentCommand


class _Info(dict):
    pass


class _State:
    def __init__(self, info):
        self.info = info


class _Upstream:
    """Mimics StyleJoystick.step: after step 500 a different command is sampled."""

    def step(self, state, action):
        state.info["step"] += 1
        if state.info["step"] > 500:
            state.info["command"] = ("resampled",)
            state.info["step"] = 0
        return state


class _Persistent(PersistentCommand, _Upstream):
    pass


def test_command_survives_the_upstream_resampling_boundary():
    state = _State(_Info(step=0, command=("forward",)))
    env = _Persistent()
    for _ in range(1000):
        state = env.step(state, None)
        assert state.info["command"] == ("forward",)


def test_without_the_mixin_the_boundary_changes_the_command():
    state = _State(_Info(step=0, command=("forward",)))
    for _ in range(501):
        state = _Upstream().step(state, None)
    assert state.info["command"] == ("resampled",)


def test_replicated_balanced_order_is_index_mod_seven():
    import numpy as np

    from nerva.training.neutral_reference import COMMANDS
    tiled = np.tile(np.asarray(COMMANDS), (3, 1))
    assert all(tuple(tiled[i]) == COMMANDS[i % 7] for i in range(21))


def test_per_command_means_follow_the_balanced_order():
    import numpy as np

    from experiments.locomotion_curriculum.gpu_pilot import FULL, per_command
    values = np.tile(np.arange(7, dtype=float), 3)[:, None] * np.ones((1, 4))  # env i has value i mod 7
    assert per_command(values, 3) == list(range(7))
    assert 7 * FULL["replicas"] % FULL["num_minibatches"] == 0
