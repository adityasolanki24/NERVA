"""Experimental neutral motor semantics shared by future training and inference.

Opt-in infrastructure only: historical policies, rewards and safety do not use it.
"""
from __future__ import annotations

import numpy as np

PLANAR_REST_LIMIT = .005
YAW_REST_LIMIT = .02


def at_rest(command, xp=np):
    command = xp.asarray(command)
    planar = xp.sqrt(xp.sum(command[..., :2] ** 2, axis=-1))
    return (planar <= PLANAR_REST_LIMIT) & (xp.abs(command[..., 2]) <= YAW_REST_LIMIT)


def canonical_command(command, xp=np):
    command = xp.asarray(command)
    motion = xp.where(at_rest(command, xp)[..., None], xp.zeros_like(command[..., :3]), command[..., :3])
    return xp.concatenate((motion, command[..., 3:]), axis=-1)


def motor_clock_tick(index, previous_rest, command, period_steps, xp=np):
    """Rest freezes at zero; movement onset starts at zero; subsequent ticks advance."""
    if period_steps <= 0:
        raise ValueError("period_steps must be positive")
    command = xp.asarray(command)
    rest = at_rest(command, xp)
    next_index = xp.where(rest, 0, xp.where(previous_rest, 0, (index + 1) % period_steps))
    angle = next_index * (2 * xp.pi / period_steps)
    phase = xp.stack((xp.cos(angle), xp.sin(angle)), axis=-1)
    return next_index, rest, phase, canonical_command(command, xp)


def planar_tracking(command, local_velocity, sigma=.01, xp=np):
    target = canonical_command(command, xp)[..., :2]
    return xp.exp(-xp.sum((target - xp.asarray(local_velocity)[..., :2]) ** 2, axis=-1) / sigma)


class NeutralPhaseClock:
    """NumPy inference adapter; does not attach itself to any policy or simulator."""
    def __init__(self, period_steps=27):
        if period_steps <= 0:
            raise ValueError("period_steps must be positive")
        self.period_steps = period_steps
        self.reset()

    def reset(self):
        self.index, self.previous_rest = 0, True
        return np.array([1., 0.])

    def tick(self, command):
        if np.shape(command) != (7,) or not np.isfinite(command).all():
            raise ValueError("expected finite seven-slot motor command")
        index, rest, phase, canonical = motor_clock_tick(self.index, self.previous_rest, command, self.period_steps)
        self.index, self.previous_rest = int(index), bool(rest)
        return {"index": self.index, "rest": self.previous_rest, "phase": phase, "command": canonical}


def training_motor_tick(info, command, period_steps):
    """JIT-friendly future training adapter; callers own state, observations and reward dispatch."""
    import jax.numpy as jp
    index, rest, phase, canonical = motor_clock_tick(info["motor_phase_index"], info["motor_previous_rest"],
                                                    command, period_steps, xp=jp)
    return {**info, "motor_phase_index": index, "motor_previous_rest": rest,
            "imitation_phase": phase, "command": canonical}
