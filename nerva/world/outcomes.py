"""Outcome monitor: measurements → `OutcomeSignal`s (what happened) and `RiskEstimate`s (what nearly did).

Memory should learn "what happens with this person / at this place" from consequences the robot can
measure, not from the emotions its own appraisal produced (docs/architecture.md §1.3, refactor stage D).
Only kinds the system can measure or estimate are produced:

  stability_loss  torso tilt crosses NEAR_FALL_TILT_DEG (IMU orientation). Re-arms below half of it.
                  magnitude = (tilt − threshold) / TILT_SPAN, clipped to [0.2, 1].
  collision_risk  (RiskEstimate, not an outcome) a tracked agent within NEAR_M whose time to contact
                  (distance / ego-motion-corrected closing speed) is below TTC_S. Once per track until it
                  is beyond REARM_M again. magnitude = closing speed / CLOSING_FULL, clipped to [0, 1].
                  Estimated from tracking (depth camera on hardware). The MuJoCo people have no collision
                  geometry, so no contact_impact outcome can occur in the current scenes.
  benign_contact  a gentle touch measured by the touch sensor (simulated in MuJoCo as "touch_gentle"
                  events), attributed to the touching track.

All thresholds are NERVA design choices, fixed before the first grounded-memory evaluation (2026-10-02).
"""

from __future__ import annotations

from nerva.interfaces import OutcomeSignal, RiskEstimate, Track

NEAR_FALL_TILT_DEG = 20.0  # same threshold as the scenario's near_fall event
TILT_SPAN = 25.0
NEAR_M = 0.8
TTC_S = 1.0
REARM_M = 1.2
CLOSING_FULL = 1.5  # m/s


class OutcomeMonitor:
    def __init__(self):
        self._tilt_armed = True
        self._collision_armed: dict[str, bool] = {}

    def stability(self, t: float, tilt_deg: float) -> list[OutcomeSignal]:
        if self._tilt_armed and tilt_deg > NEAR_FALL_TILT_DEG:
            self._tilt_armed = False
            mag = min(1.0, max(0.2, (tilt_deg - NEAR_FALL_TILT_DEG) / TILT_SPAN))
            return [OutcomeSignal("stability_loss", mag, t, provenance=f"imu tilt {tilt_deg:.1f} deg")]
        if tilt_deg < NEAR_FALL_TILT_DEG / 2:
            self._tilt_armed = True
        return []

    def proximity(self, t: float, tracks: tuple[Track, ...]) -> list[RiskEstimate]:
        out = []
        for tr in tracks:
            if tr.kind != "person" or not tr.tid:
                continue
            armed = self._collision_armed.get(tr.tid, True)
            closing = tr.approach_speed
            if armed and tr.visible and tr.distance < NEAR_M and closing > 0 and tr.distance / closing < TTC_S:
                self._collision_armed[tr.tid] = False
                out.append(RiskEstimate("collision_risk", min(1.0, closing / CLOSING_FULL), t, source=tr.tid,
                                        provenance=f"track d={tr.distance:.2f} m closing={closing:.2f} m/s"))
            elif tr.distance > REARM_M:
                self._collision_armed[tr.tid] = True
        return out

    @staticmethod
    def contact(t: float, source: str, magnitude: float = 1.0) -> OutcomeSignal:
        return OutcomeSignal("benign_contact", magnitude, t, source=source, provenance="touch sensor (simulated)")
