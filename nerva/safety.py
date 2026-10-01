"""Deterministic safety supervisor: an independent override path below behaviour (docs/architecture.md §3).

Applied to every behaviour request before it reaches the locomotion policy. Its inputs are the request
and the robot's SelfState only. It never reads affect, action tendencies, memory or appraisal, so none of
them can delay, suppress or gate it (enforced by the signature and tested).

Rule (simulation; NERVA design values):
  STOP   when tilt ≥ STOP_TILT_DEG or stability_risk ≥ STOP_RISK: zero velocity command (the policy's
         trained stand-still command) and neutral head offsets
  HOLD   the stop until tilt ≤ RESUME_TILT_DEG continuously for RESUME_HOLD_S
Normal walking in the recorded scenarios tilts at most ~16° (B2), so the rule does not act there.

What is enforced where (as of 2026-10-02):
  this module            stop-on-instability override of the behaviour command (simulation)
  MuJoCo model           actuator force and control ranges (upstream Open Duck model)
  upstream control loop  joint-target rate limit (5.24 rad/s) in the policy wrapper (nerva/sim/open_duck.py)
  NOT YET                emergency stop, motor torque/temperature limits and fall-triggered motor disable on
                         hardware: required before anything runs on the physical robot
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from nerva.interfaces import BehaviourCommand, SelfState

STOP_TILT_DEG = 25.0
STOP_RISK = 0.75
RESUME_TILT_DEG = 12.0
RESUME_HOLD_S = 0.5


@dataclass
class SafetySupervisor:
    stopped: bool = False
    _calm_since: float | None = None
    interventions: int = 0

    def filter(self, command: BehaviourCommand, head: tuple[float, float, float, float],
               self_state: SelfState) -> tuple[BehaviourCommand, tuple[float, float, float, float], str]:
        """Returns (command, head offsets, reason); reason is "" when the request passes unchanged."""
        t = self_state.time_s
        if self_state.tilt_deg >= STOP_TILT_DEG or self_state.stability_risk >= STOP_RISK:
            if not self.stopped:
                self.interventions += 1
            self.stopped, self._calm_since = True, None
        elif self.stopped:
            if self_state.tilt_deg <= RESUME_TILT_DEG:
                self._calm_since = t if self._calm_since is None else self._calm_since
                if t - self._calm_since >= RESUME_HOLD_S:
                    self.stopped, self._calm_since = False, None
            else:
                self._calm_since = None
        if not self.stopped:
            return command, head, ""
        stop = dataclasses.replace(command, vx=0.0, vy=0.0, yaw_rate=0.0)
        return stop, (0.0, 0.0, 0.0, 0.0), f"safety stop (tilt {self_state.tilt_deg:.0f} deg)"
