"""Simulated perception: a stand-in for a face/object detector on the robot's head camera.

We cannot meaningfully run a real face detector on MuJoCo renders, so detections are made from
ground-truth entity positions, restricted to what the head camera could see
(docs/reactive_behaviour_design.md §3):
  - field of view of the "robot_eye" camera (it moves with head yaw/pitch) and a range per kind
  - detection probability falling with distance; bearing and distance noise (seeded)
  - a filter per kind: smoothed distance; approach speed = least-squares slope of the raw distances
    over the last FAST_WINDOW / SLOW_WINDOW detections; kept for MEMORY_S after the last detection
Events are derived from the tracks. All numbers are NERVA engineering choices, not a sensor model.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace

import numpy as np

from nerva.interfaces import Event, PerceptionState, Track

HALF_FOV_H = np.radians(55.0)  # horizontal half-angle
HALF_FOV_V = np.radians(45.0)  # vertical half-angle (camera fovy 70° plus some margin)
RANGE_M = {"person": 5.0, "ball": 3.0}
BEARING_NOISE = np.radians(2.0)
DISTANCE_NOISE = 0.02  # relative
MEMORY_S = 1.5  # keep an unseen track this long
LOST_EVENT_S = 3.0  # report "<kind>_lost" after this long unseen (track already dropped)
ALPHA_POS = 0.4  # per-frame smoothing of distance/bearing/elevation, for FRAME_HZ
FRAME_HZ = 10  # detector rate: call detect() every 0.1 s
FAST_WINDOW, SLOW_WINDOW = 6, 12  # detections used for the fast / slow approach-speed slopes

FAST_APPROACH = 0.5  # m/s
SLOW_APPROACH = 0.1  # m/s
APPROACH_RANGE = 3.0  # m: approach events only within this distance
CLOSE = 0.6  # m
REARM_SPEED = 0.05  # m/s: an approach event can fire again once the speed has dropped below this


def _wrap(a: float) -> float:
    return float((a + np.pi) % (2 * np.pi) - np.pi)


@dataclass
class _TrackState:
    track: Track
    last_seen_t: float
    approach_armed: bool = True
    close_armed: bool = True
    history: deque = field(default_factory=lambda: deque(maxlen=SLOW_WINDOW))  # (t, raw distance)


def _slope_towards(history, n: int) -> float:
    """Approach speed: minus the least-squares slope of distance over the last n detections."""
    pts = list(history)[-n:]
    if len(pts) < n:
        return 0.0
    t, d = np.array(pts).T
    t = t - t.mean()
    return float(-(t @ (d - d.mean())) / (t @ t))


class SimulatedPerception:
    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.tracks: dict[str, _TrackState] = {}
        self._lost_pending: dict[str, float] = {}

    def detect(self, t: float, cam_pos: np.ndarray, cam_xmat: np.ndarray, body_yaw: float,
               entities: dict[str, np.ndarray], dt: float) -> PerceptionState:
        """One camera frame. cam_xmat: 3x3 camera orientation (MuJoCo: looks along -z, +y up)."""
        events: list[Event] = []
        forward, up = -cam_xmat[:, 2], cam_xmat[:, 1]
        right = np.cross(forward, up)
        for kind, pos in entities.items():
            rel = pos - cam_pos
            x, y, z = rel @ forward, rel @ right, rel @ up
            h_dist = float(np.hypot(rel[0], rel[1]))
            # A person is detectable if any part of the body (floor to face) is in the vertical field
            # of view; up close a small robot sees legs, not the face. Elevation still refers to `pos`.
            z_low = z - (pos[2] - 0.1) * up[2] if kind == "person" else z
            in_view = (x > 0 and abs(np.arctan2(y, x)) < HALF_FOV_H and h_dist < RANGE_M[kind]
                       and np.arctan2(z_low, x) < HALF_FOV_V and np.arctan2(z, x) > -HALF_FOV_V)
            p_detect = 1.0 - 0.6 * (h_dist / RANGE_M[kind]) ** 2
            if in_view and self.rng.random() < p_detect:
                bearing = _wrap(np.arctan2(rel[1], rel[0]) - body_yaw) + self.rng.normal(0, BEARING_NOISE)
                dist = h_dist * (1 + self.rng.normal(0, DISTANCE_NOISE))
                elevation = float(np.arctan2(z, np.hypot(x, y)))
                events += self._update(kind, t, bearing, dist, elevation, dt)
        for kind, st in list(self.tracks.items()):
            unseen = t - st.last_seen_t
            if unseen > 0:
                st.track = replace(st.track, visible=False, unseen_for_s=unseen)
            if unseen > MEMORY_S:
                del self.tracks[kind]
                self._lost_pending[kind] = st.last_seen_t
        for kind, last in list(self._lost_pending.items()):
            if kind in self.tracks:
                del self._lost_pending[kind]
            elif t - last > LOST_EVENT_S:
                events.append(Event(f"{kind}_lost"))
                del self._lost_pending[kind]
        return PerceptionState(t, tuple(events), tuple(st.track for st in self.tracks.values()))

    def _update(self, kind, t, bearing, dist, elevation, dt) -> list[Event]:
        events = []
        st = self.tracks.get(kind)
        if st is None:
            st = _TrackState(Track(kind, bearing, dist, elevation), t)
            st.history.append((t, dist))
            self.tracks[kind] = st
            events.append(Event(f"{kind}_appeared"))
            return events
        old = st.track
        gap = max(t - st.last_seen_t, dt)
        d = (1 - ALPHA_POS) * old.distance + ALPHA_POS * dist
        st.history.append((t, dist))
        fast, speed = _slope_towards(st.history, FAST_WINDOW), _slope_towards(st.history, SLOW_WINDOW)
        st.track = Track(kind, _wrap(old.bearing + ALPHA_POS * _wrap(bearing - old.bearing)), d,
                         (1 - ALPHA_POS) * old.elevation + ALPHA_POS * elevation, speed, True,
                         old.seen_for_s + gap, 0.0)
        st.last_seen_t = t
        if kind == "person" and d < APPROACH_RANGE:
            if st.approach_armed and fast > FAST_APPROACH:
                events.append(Event("person_approaching_rapidly", min(1.0, fast / (2 * FAST_APPROACH))))
                st.approach_armed = False
            elif st.approach_armed and speed > SLOW_APPROACH and speed < FAST_APPROACH / 2 and st.track.seen_for_s > 1:
                events.append(Event("person_approaching_slowly"))
                st.approach_armed = False
            elif max(speed, fast) < REARM_SPEED:
                st.approach_armed = True
        if st.close_armed and d < CLOSE:
            events.append(Event(f"{kind}_close"))
            st.close_armed = False
        elif d > CLOSE + 0.3:
            st.close_armed = True
        return events
