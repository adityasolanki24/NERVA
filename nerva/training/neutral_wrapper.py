"""Candidate-only autoreset keeping motor info consistent with restored observations."""
from brax.envs.wrappers import training
import jax
import jax.numpy as jp
from mujoco_playground import wrapper

WRAPPER_INFO = {"first_state", "first_obs", "steps", "truncation", "episode_done", "episode_metrics", "rng"}


class NeutralAutoResetWrapper(wrapper.BraxAutoResetWrapper):
    def reset(self, rng):
        state = super().reset(rng)
        state.info["first_motor_info"] = {key: value for key, value in state.info.items() if key not in WRAPPER_INFO}
        return state

    def step(self, state, action):
        state = super().step(state, action)

        def restore(initial, current):
            initial = jp.asarray(initial)
            done = state.done
            if done.ndim:
                done = done.reshape(done.shape + (1,) * (initial.ndim - done.ndim))
            return jp.where(done, initial, current)

        for key, initial in state.info["first_motor_info"].items():
            state.info[key] = jax.tree.map(restore, initial, state.info[key])
        return state


def wrap_neutral_for_training(env, episode_length=8, action_repeat=1, randomization_fn=None):
    if randomization_fn is not None:
        raise ValueError("this neutral wrapper has no validated domain randomization path")
    env = training.VmapWrapper(env)
    env = training.EpisodeWrapper(env, episode_length, action_repeat)
    return NeutralAutoResetWrapper(env)
