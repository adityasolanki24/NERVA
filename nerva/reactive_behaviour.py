"""Behaviour v1: reactive modes driven by emotion, PAD and perception (docs/reactive_behaviour_design.md §3).

WHAT: a mode chosen from the active emotions, PAD and the perceived tracks, with a minimum dwell
time (hysteresis); urgent modes (freeze, retreat) pre-empt. Each mode is a small continuous
controller over (vx, yaw_rate) and a head gaze:
  explore   wander and scan with the head
  orient    stop, turn head and body toward something that just appeared or surprised
  approach  walk toward the most salient visible thing, stop at a distance (larger when dominance is low)
  inspect   stay there, gaze at it with a curious head tilt; ends as interest habituates
  freeze    startle: stop, head drops, for FREEZE_S
  retreat   step back facing the threat, then turn away and walk off, glancing back; until safe
  watch     wary (mild fear): stand facing the person, gaze at them, back off if closer than WARY_DISTANCE
  withdraw  distress: stop, head down, look away
HOW: S1 style e from PAD (nerva.behaviour.pad_to_style_vector), tempo bounded per mode.
Only commands the S1 policy already takes are produced (velocities, style, head offsets); PAD never
reaches joints. All gains, thresholds and durations are NERVA design choices, tuned for visibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from nerva.behaviour import pad_to_style_vector
from nerva.interfaces import BehaviourCommand, PADState, StyleVector, Track

MODES = ("explore", "orient", "approach", "inspect", "freeze", "retreat", "watch", "withdraw")
WARY_FEAR = 0.05
WARY_DISTANCE = 1.5  # m
MIN_DWELL_S = 1.0
FREEZE_S = 0.8
FREEZE_DISTANCE = 2.0  # m: a startle closer than this freezes before retreating
BACKSTEP_S = 1.5
SAFE_DISTANCE = 2.2  # m: retreat until at least this far (and fear has dropped)
MAX_VX, MAX_YAW = 0.15, 1.0  # trained command range
# rad. POSITIVE head_pitch tilts the face UP: measured on the robot_eye camera, +0.4 → 35° up,
# -0.4 → 14° down (2026-10-01). (experiments/expressive_locomotion/head_posture.py assumes the
# opposite sign; see docs/development_log.md.)
HEAD_YAW_MAX, HEAD_PITCH_DOWN, HEAD_PITCH_UP = 1.3, -0.6, 0.6
HEAD_TAU_S = 0.25
GAZE_GAIN = 0.3  # per perception frame, on the camera elevation error
# S1 was trained without head motion applied (upstream), and stops walking when the head moves
# (measured 2026-09-30: 0.2 rad yaw → 0.012 m/s; ≤0.1 yaw / ≤0.2 pitch, roll tolerated). While
# walking, head offsets are therefore limited to this (pitch, yaw, roll); None = no limit (S3).
S1_WALKING_HEAD_LIMIT = (0.15, 0.08, 0.15)
FEAR_ENTER, FEAR_KEEP = 0.15, 0.06
POSITIVE_ENTER, POSITIVE_KEEP = 0.12, 0.05  # hysteresis for approach/inspect


def _wrap(a: float) -> float:
    return float((a + np.pi) % (2 * np.pi) - np.pi)


@dataclass
class ReactiveDecision:
    command: BehaviourCommand
    head: tuple[float, float, float, float]  # neck_pitch, head_pitch, head_yaw, head_roll [rad]
    mode: str
    target: str | None
    reason: str


@dataclass
class ReactiveBehaviour:
    mode: str = "explore"
    mode_since: float = 0.0
    target: str | None = None
    threat_bearing: float = 0.0
    head: np.ndarray = field(default_factory=lambda: np.zeros(4))
    appeared_at: dict = field(default_factory=dict)
    walking_head_limit: tuple[float, float, float] | None = S1_WALKING_HEAD_LIMIT

    def notice(self, t: float, event_kind: str) -> None:
        """Perception events the behaviour itself cares about (something new to orient to)."""
        if event_kind.endswith("_appeared"):
            self.appeared_at[event_kind.split("_")[0]] = t

    def step(self, t: float, dt: float, pad: PADState, emotions: dict[str, float],
             tracks: tuple[Track, ...], salience: dict[str, float] | None = None) -> ReactiveDecision:
        """salience: per-kind novelty from appraisal (0..1); used by selectors that weigh targets."""
        self.salience = salience or {}
        tr = {x.kind: x for x in tracks}
        if "person" in tr:
            self.threat_bearing = tr["person"].bearing
        self._select(t, pad, emotions, tr)
        vx, yaw, head_target, reason = self._control(t, pad, tr)

        e = pad_to_style_vector(pad)
        tempo = e.tempo
        if self.mode in ("retreat", "freeze"):
            tempo = max(tempo, 0.6)
        elif self.mode == "withdraw":
            tempo = min(tempo, -0.5)
        style = StyleVector(tempo=float(np.clip(tempo, -1, 1)), step_height=0.0, torso_pitch=e.torso_pitch)
        if self.walking_head_limit is not None and (abs(vx) > 0.02 or abs(yaw) > 0.3):
            lp, ly, lr = self.walking_head_limit
            head_target = (0.0, float(np.clip(head_target[1], -lp, lp)), float(np.clip(head_target[2], -ly, ly)),
                           float(np.clip(head_target[3], -lr, lr)))
        a = 1.0 - np.exp(-dt / HEAD_TAU_S)
        self.head = self.head + a * (np.asarray(head_target, dtype=float) - self.head)
        cmd = BehaviourCommand(vx=float(np.clip(vx, -MAX_VX, MAX_VX)),
                               yaw_rate=float(np.clip(yaw, -MAX_YAW, MAX_YAW)), style_vector=style)
        return ReactiveDecision(cmd, tuple(float(h) for h in self.head), self.mode, self.target, reason)

    # ── mode selection ──────────────────────────────────────────────────────

    def _set(self, t: float, mode: str, target: str | None = None) -> None:
        if mode != self.mode or target != self.target:
            self.mode, self.mode_since, self.target = mode, t, target

    def _select(self, t, pad, emo, tr) -> None:
        fear, surprise = emo.get("fear", 0.0), emo.get("surprise", 0.0)
        distress = emo.get("distress", 0.0)
        positive = max(emo.get("interest", 0.0), emo.get("hope", 0.0), emo.get("joy", 0.0))
        age = t - self.mode_since
        person = tr.get("person")
        near_person = person is not None and person.distance < SAFE_DISTANCE

        if self.mode == "freeze" and age < FREEZE_S:
            return
        if surprise > 0.4 and fear > FEAR_ENTER and person is not None and person.distance < FREEZE_DISTANCE \
                and self.mode not in ("freeze", "retreat"):
            return self._set(t, "freeze", "person")
        if fear > FEAR_ENTER and (person is None or person.distance < SAFE_DISTANCE + 0.5):
            return self._set(t, "retreat", "person")
        if self.mode == "retreat" and fear > FEAR_KEEP and (person is None or near_person):
            return  # keep retreating (hysteresis)
        if age < MIN_DWELL_S and self.mode != "explore":  # exploring can be interrupted any time
            return
        if distress > 0.2 and fear < 0.1:
            return self._set(t, "withdraw", self.target)
        if fear > WARY_FEAR and person is not None:
            return self._set(t, "watch", "person")
        target = self._approach_target(tr)
        engaged = self.mode in ("approach", "inspect") and self.target == target
        if (positive > POSITIVE_ENTER or (engaged and positive > POSITIVE_KEEP)) and fear < 0.05 and target is not None:
            stop = self._stop_distance(target, pad)
            margin = 0.3 if (self.mode == "inspect" and self.target == target) else 0.1
            return self._set(t, "inspect" if tr[target].distance < stop + margin else "approach", target)
        recent = self._recent_appearance(t, tr)
        if recent is not None or (surprise > 0.25 and tr):
            return self._set(t, "orient", recent or next(iter(tr)))
        self._set(t, "explore")

    def _recent_appearance(self, t, tr):
        recent = [(t0, k) for k, t0 in self.appeared_at.items() if t - t0 < 2.0 and k in tr]
        return max(recent)[1] if recent else None

    @staticmethod
    def _approach_target(tr):
        for kind in ("person", "ball"):  # a person is more salient than a ball
            if kind in tr:  # visible, or seen within perception's memory
                return kind
        return None

    @staticmethod
    def _stop_distance(kind, pad):
        if kind == "ball":
            return 0.45
        return 0.7 + 0.5 * max(0.0, -pad.dominance)  # keep more distance when feeling less in control

    # ── controllers ─────────────────────────────────────────────────────────

    def _gaze(self, track, tilt=0.0):
        """Head offsets pointing at a track (head pitch closed-loop on the camera elevation)."""
        pitch = float(np.clip(self.head[1] + GAZE_GAIN * track.elevation, HEAD_PITCH_DOWN, HEAD_PITCH_UP))
        return (0.0, pitch, float(np.clip(track.bearing, -HEAD_YAW_MAX, HEAD_YAW_MAX)), tilt)

    def _control(self, t, pad, tr):
        m = self.mode
        target = tr.get(self.target) if self.target else None
        age = t - self.mode_since
        if m == "freeze":
            return 0.0, 0.0, (0.0, -0.3, self.head[2], 0.0), "freeze: startled"
        if m == "retreat":
            look = (0.0, self.head[1], float(np.clip(self.threat_bearing, -HEAD_YAW_MAX, HEAD_YAW_MAX)), 0.0)
            if age < BACKSTEP_S:  # step back while facing the threat
                return -MAX_VX, 1.5 * self.threat_bearing, look, "retreat: step back"
            away = _wrap(self.threat_bearing + np.pi)  # heading error to "directly away"
            vx = MAX_VX if abs(away) < 0.6 else 0.02
            return vx, 2.0 * away, look, "retreat: turn away and leave"
        if m == "watch" and target is not None:
            vx = -MAX_VX if target.distance < WARY_DISTANCE else 0.0
            return vx, 1.5 * target.bearing, self._gaze(target), "watch: wary, keep distance"
        if m == "withdraw":
            yaw = float(np.clip(-np.sign(self.threat_bearing) * 0.8, -HEAD_YAW_MAX, HEAD_YAW_MAX))
            return 0.0, 0.0, (0.0, -0.5, yaw, 0.0), "withdraw: head down, look away"
        if m == "inspect" and target is not None:
            tilt = 0.3 * np.sin(2 * np.pi * 0.25 * age)  # curious head tilt
            return 0.0, 1.2 * target.bearing, self._gaze(target, tilt), f"inspect {self.target}"
        if m == "approach" and target is not None:
            stop = self._stop_distance(self.target, pad)
            # full speed or nothing: the policy barely moves for small speed commands (measured)
            vx = MAX_VX if (target.distance > stop and abs(target.bearing) < 0.8) else 0.0
            return vx, 1.5 * target.bearing, self._gaze(target), f"approach {self.target}"
        if m == "orient" and target is not None:
            return 0.0, 1.5 * target.bearing, self._gaze(target), f"orient to {self.target}"
        scan = 0.6 * np.sin(2 * np.pi * 0.15 * t)
        return 0.10, 0.3 * np.sin(2 * np.pi * 0.05 * t), (0.0, 0.0, scan, 0.0), "explore"
