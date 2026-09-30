"""Perception front end and multi-object tracker for the robot's head camera.

Detections come either from ground truth restricted to what the camera could see (SimulatedPerception,
a stand-in for a detector) or from the camera images (nerva.vision.VisionPerception). Both feed the same
tracker (docs/reactive_behaviour_design.md §3):
  - several objects of the same kind are tracked at once, each with an ID ("person-0", "person-1", …)
  - association: each detection is matched to the nearest existing track of its kind in the world
    frame (gate ASSOC_GATE_M), penalised by appearance mismatch when both have an appearance; unmatched
    detections start new tracks
  - per track: smoothed distance/bearing/elevation; approach speed = least-squares slope of the
    ego-motion-corrected distances over FAST_WINDOW / SLOW_WINDOW detections; kept MEMORY_S unseen
  - events carry the track ID (Event.source), so appraisal and memory know WHO an event is about
SimulatedPerception adds field-of-view/range limits, distance-dependent misses and noise (seeded).
All numbers are NERVA engineering choices, not a sensor model.
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
ASSOC_GATE_M = 0.8  # max world-frame jump between frames for the same track (a lunge moves 0.18 m/frame)
APPEARANCE_MISMATCH = 0.5  # cosine below this between appearances forbids an association

FAST_APPROACH = 0.7  # m/s (a lunge; the friendly scenario walk is 0.25)
SLOW_APPROACH = 0.1  # m/s
APPROACH_RANGE = 3.0  # m: approach events only within this distance
CLOSE = 0.6  # m
REARM_SPEED = 0.05  # m/s: an approach event can fire again once the speed has dropped below this


def _wrap(a: float) -> float:
    return float((a + np.pi) % (2 * np.pi) - np.pi)


def entity_class(name: str) -> str:
    """Simulated entity name → detector class ("person_b" is a person)."""
    return "person" if name.startswith("person") else name


@dataclass(frozen=True)
class Observation:
    kind: str
    bearing: float  # rad, relative to the body heading
    distance: float  # m, horizontal
    elevation: float  # rad
    world_xy: tuple[float, float]
    ego_closing: float = 0.0  # m/s, the robot's own speed toward the observed point
    appearance: np.ndarray | None = None


@dataclass
class _TrackState:
    track: Track
    last_seen_t: float
    world_xy: tuple[float, float]
    appearance: np.ndarray | None = None
    approach_armed: bool = True  # slow approach
    rapid_armed: bool = True  # independent, so a slow approach cannot mask a lunge
    close_armed: bool = True
    history: deque = field(default_factory=lambda: deque(maxlen=SLOW_WINDOW))  # (t, ego-corrected distance)
    ego_travel: float = 0.0


def _slope_towards(history, n: int) -> float:
    """Approach speed: minus the least-squares slope of distance over the last n detections."""
    pts = list(history)[-n:]
    if len(pts) < n:
        return 0.0
    t, d = np.array(pts).T
    t = t - t.mean()
    return float(-(t @ (d - d.mean())) / (t @ t))


def _cosine(a, b) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0


class SimulatedPerception:
    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.tracks: dict[str, _TrackState] = {}  # by track ID
        self._lost_pending: dict[str, tuple[str, float]] = {}  # tid -> (kind, last seen)
        self._next_id: dict[str, int] = {}

    # ── detector (ground truth + camera limits + noise) ────────────────────
    def detect(self, t: float, cam_pos: np.ndarray, cam_xmat: np.ndarray, body_yaw: float,
               entities: dict[str, np.ndarray], dt: float,
               ego_velocity: np.ndarray | None = None) -> PerceptionState:
        """One camera frame. cam_xmat: 3x3 camera orientation (MuJoCo: looks along -z, +y up).

        entities: name → salient world point (a person's face, the ball's centre); names starting with
        "person" are persons. ego_velocity: the robot's horizontal world velocity (odometry), or None.
        """
        ego = np.zeros(2) if ego_velocity is None else np.asarray(ego_velocity, dtype=float)[:2]
        forward, up = -cam_xmat[:, 2], cam_xmat[:, 1]
        right = np.cross(forward, up)
        observations = []
        for name, pos in entities.items():
            kind = entity_class(name)
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
                heading = body_yaw + bearing
                world = (float(cam_pos[0] + dist * np.cos(heading)), float(cam_pos[1] + dist * np.sin(heading)))
                toward = rel[:2] / max(h_dist, 1e-6)
                observations.append(Observation(kind, bearing, dist, float(np.arctan2(z, np.hypot(x, y))), world,
                                                float(ego @ toward)))
        return self.ingest(t, observations, dt)

    # ── tracker ─────────────────────────────────────────────────────────────
    def ingest(self, t: float, observations: list[Observation], dt: float) -> PerceptionState:
        """Associate this frame's observations with tracks, update them, and age the rest."""
        events: list[Event] = []
        pairs = []
        for oi, ob in enumerate(observations):
            for tid, st in self.tracks.items():
                if st.track.kind != ob.kind:
                    continue
                d = float(np.hypot(ob.world_xy[0] - st.world_xy[0], ob.world_xy[1] - st.world_xy[1]))
                cost = d
                if ob.appearance is not None and st.appearance is not None:
                    c = _cosine(ob.appearance, st.appearance)
                    if c < APPEARANCE_MISMATCH:
                        continue
                    cost += 0.5 * (1.0 - c)
                if d < ASSOC_GATE_M:
                    pairs.append((cost, oi, tid))
        used_obs, used_tracks = set(), set()
        for cost, oi, tid in sorted(pairs):
            if oi in used_obs or tid in used_tracks:
                continue
            used_obs.add(oi)
            used_tracks.add(tid)
            events += self._update(tid, observations[oi], t, dt)
        for oi, ob in enumerate(observations):
            if oi not in used_obs:
                n = self._next_id.get(ob.kind, 0)
                self._next_id[ob.kind] = n + 1
                events += self._update(f"{ob.kind}-{n}", ob, t, dt)
        return self._age_tracks(t, events)

    def _age_tracks(self, t: float, events: list[Event]) -> PerceptionState:
        """Mark unseen tracks, forget old ones, report losses; returns this frame's PerceptionState."""
        for tid, st in list(self.tracks.items()):
            unseen = t - st.last_seen_t
            if unseen > 0:
                st.track = replace(st.track, visible=False, unseen_for_s=unseen)
            if unseen > MEMORY_S:
                del self.tracks[tid]
                self._lost_pending[tid] = (st.track.kind, st.last_seen_t)
        for tid, (kind, last) in list(self._lost_pending.items()):
            if tid in self.tracks:
                del self._lost_pending[tid]
            elif t - last > LOST_EVENT_S:
                events.append(Event(f"{kind}_lost", source=tid))
                del self._lost_pending[tid]
        return PerceptionState(t, tuple(events), tuple(st.track for st in self.tracks.values()))

    def _update(self, tid: str, ob: Observation, t: float, dt: float) -> list[Event]:
        events = []
        st = self.tracks.get(tid)
        if st is None:
            st = _TrackState(Track(ob.kind, ob.bearing, ob.distance, ob.elevation, tid=tid), t, ob.world_xy,
                             ob.appearance)
            st.history.append((t, ob.distance))  # distances corrected for the robot's own motion
            self.tracks[tid] = st
            events.append(Event(f"{ob.kind}_appeared", source=tid))
            return events
        old = st.track
        gap = max(t - st.last_seen_t, dt)
        d = (1 - ALPHA_POS) * old.distance + ALPHA_POS * ob.distance
        st.ego_travel += ob.ego_closing * gap  # how much closer the robot itself has moved
        st.history.append((t, ob.distance + st.ego_travel))
        fast, speed = _slope_towards(st.history, FAST_WINDOW), _slope_towards(st.history, SLOW_WINDOW)
        st.track = Track(ob.kind, _wrap(old.bearing + ALPHA_POS * _wrap(ob.bearing - old.bearing)), d,
                         (1 - ALPHA_POS) * old.elevation + ALPHA_POS * ob.elevation, speed, True,
                         old.seen_for_s + gap, 0.0, tid=tid)
        st.last_seen_t = t
        st.world_xy = ob.world_xy
        if ob.appearance is not None:
            st.appearance = ob.appearance
        if ob.kind == "person" and d < APPROACH_RANGE:
            if st.rapid_armed and fast > FAST_APPROACH:
                events.append(Event("person_approaching_rapidly", min(1.0, fast / (2 * FAST_APPROACH)), source=tid))
                st.rapid_armed = st.approach_armed = False
            elif (st.approach_armed and SLOW_APPROACH < speed < FAST_APPROACH / 2 and fast < FAST_APPROACH / 2
                  and st.track.seen_for_s > 1):
                events.append(Event("person_approaching_slowly", source=tid))
                st.approach_armed = False
            if max(speed, fast) < REARM_SPEED:
                st.approach_armed = st.rapid_armed = True
        if st.close_armed and d < CLOSE:
            events.append(Event(f"{ob.kind}_close", source=tid))
            st.close_armed = False
        elif d > CLOSE + 0.3:
            st.close_armed = True
        return events
