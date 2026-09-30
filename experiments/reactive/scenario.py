"""Reactive-behaviour scenario: scripted agents in the scene, the robot's full loop reacting to them.

Loop per 20 ms control step (docs/reactive_behaviour_design.md §2):
  agents move (their scripts may refer to where the robot is NOW, so the scene stays in view)
  every 0.1 s: head camera → SimulatedPerception → events + tracks
               → ContextualAppraiser → CategoricalAffectModel (v0.2, with interest) → PAD, emotions
               → ReactiveBehaviour → velocity command + S1 style vector + head gaze
  OpenDuckSim with the S1 policy (unchanged), 500 Hz physics.
Nothing about the robot's reactions is scripted: only the person and the ball follow scripts.

With use_memory=True (docs/memory_design.md, M1) appraisal reads an entity memory: identities come
from the vision's appearance feature, stay bound to a track while appearance is uninformative (up
close), and events with an unknown identity are held back and learned once the identity is resolved.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from nerva import gait_metrics as gm
from nerva.affect import CategoricalAffectModel
from nerva.appraisal import ContextualAppraiser, MemoryAppraiser
from nerva.episodic import EpisodicMemory
from nerva.memory import EntityMemory
from nerva.spatial import PlaceMemory
from nerva.interfaces import BehaviourCommand, Event
from nerva.open_duck_sim import OpenDuckSim
from nerva.perception import FRAME_HZ, SimulatedPerception
from nerva.vision import VisionPerception
from nerva.action_selection import UtilityBehaviour
from nerva.reactive_behaviour import ReactiveBehaviour
from nerva.world import World, extend_scene

CTRL_DT = 0.02
PERCEIVE_EVERY = int(round(1 / (FRAME_HZ * CTRL_DT)))  # control steps per perception frame
HIDDEN = (30.0, 30.0)
NEAR_FALL_TILT_DEG = 20.0
EYE_W, EYE_H = 160, 120  # vision resolution
EMOTIONS = ("joy", "hope", "fear", "distress", "surprise", "interest")


@dataclass
class Agent:
    """A scripted mover. Steps: (start_t, action, params).

    Actions:
      hide                          out of the world
      appear   dist, bearing        appear at this distance/bearing relative to the robot's current pose
      approach speed, stop_dist     walk toward the robot's current position, stop at stop_dist
      wait                          stand still, facing the robot
      leave    speed                walk directly away from the robot
      pet      stop_dist            walk up to the robot and stay there, touching it (touch events)
    """

    steps: list[tuple[float, str, tuple]]
    xy: np.ndarray = field(default_factory=lambda: np.array(HIDDEN))
    yaw: float = 0.0
    _i: int = -1

    def update(self, t: float, dt: float, robot_xy: np.ndarray, robot_yaw: float) -> None:
        while self._i + 1 < len(self.steps) and self.steps[self._i + 1][0] <= t:
            self._i += 1
            _, action, p = self.steps[self._i]
            if action == "hide":
                self.xy = np.array(HIDDEN)
            elif action == "appear":
                dist, bearing = p
                a = robot_yaw + bearing
                self.xy = robot_xy + dist * np.array([np.cos(a), np.sin(a)])
        if self._i < 0:
            return
        _, action, p = self.steps[self._i]
        to_robot = robot_xy - self.xy
        d = float(np.linalg.norm(to_robot))
        if action in ("approach", "wait", "appear") and d > 1e-6:
            self.yaw = float(np.arctan2(to_robot[1], to_robot[0]))
        if action == "pet":
            step = min(0.25 * dt, max(0.0, d - p[0]))
            self.xy = self.xy + step * to_robot / max(d, 1e-6)
        if action == "approach":
            speed, stop = p
            step = min(speed * dt, max(0.0, d - stop))
            self.xy = self.xy + step * to_robot / max(d, 1e-6)
        elif action == "leave":
            (speed,) = p
            self.yaw = float(np.arctan2(-to_robot[1], -to_robot[0]))
            self.xy = self.xy - speed * dt * to_robot / max(d, 1e-6)

    @property
    def action(self) -> str | None:
        return self.steps[self._i][1] if self._i >= 0 else None


def default_scenario() -> dict[str, Agent]:
    """~100 s: a ball to be curious about, a friendly person, a lunge, the person leaving and returning."""
    ball = Agent([(0.0, "hide", ()), (4.0, "appear", (1.3, 0.5)), (4.0, "wait", ())])
    person = Agent([
        (0.0, "hide", ()),
        (30.0, "appear", (3.5, 0.2)), (30.0, "approach", (0.25, 1.4)),  # slow, friendly approach
        (43.0, "leave", (0.8,)), (44.8, "wait", ()),  # steps back ...
        (46.5, "approach", (1.8, 0.45)),  # ... then lunges at the robot
        (48.5, "wait", ()),
        (56.0, "leave", (0.5,)),
        (68.0, "hide", ()),
        (72.0, "appear", (3.5, -0.2)), (72.0, "approach", (0.25, 1.2)),  # returns while the lunge is recent
        (90.0, "wait", ()),
    ])
    return {"ball": ball, "person": person}


def memory_scenario() -> dict[str, Agent]:
    """~135 s, two people never in view together: A lunges early; B (green) is new, then pets the robot;
    A returns 70 s after the lunge (beyond v1's 45 s threat timer); then B returns."""
    a = Agent([(0.0, "hide", ()), (5.0, "appear", (3.5, 0.1)), (5.0, "approach", (0.25, 1.4)),
               (18.0, "leave", (0.8,)), (19.8, "wait", ()), (21.0, "approach", (1.8, 0.45)), (23.0, "wait", ()),
               (28.0, "leave", (0.5,)), (40.0, "hide", ()),
               (92.0, "appear", (3.5, -0.1)), (92.0, "approach", (0.25, 1.2)), (106.0, "leave", (0.5,)),
               (116.0, "hide", ())])
    b = Agent([(0.0, "hide", ()), (46.0, "appear", (3.5, 0.15)), (46.0, "approach", (0.25, 0.9)),
               (58.0, "pet", (0.3,)), (68.0, "leave", (0.5,)), (80.0, "hide", ()),
               (118.0, "appear", (3.5, 0.0)), (118.0, "approach", (0.25, 1.0))])
    return {"person": a, "person_b": b}


IDLE_BEFORE_SLEEP_S = 5.0  # no one in view for this long → consolidate ("sleep")
SLEEP_EVERY_S = 20.0
DEFERRED_CONFIDENCE = 0.8  # learning applied later, once an unknown identity is resolved
TOUCH_REACH = 0.5  # m
TOUCH_PERIOD_S = 1.0


class IdentityBinder:
    """Which remembered entity each perceived track is (docs/memory_design.md section 3.7)."""

    def __init__(self, memory: EntityMemory):
        self.memory = memory
        self.current: dict = {}
        self.pending: dict[str, list] = {}

    def update(self, t: float, tracks, perception) -> dict:
        present = {tr.kind for tr in tracks}
        for kind in list(self.current):
            if kind not in present:  # track lost: continuity broken, identity must be re-established
                self.current.pop(kind)
        for kind in list(self.pending):
            if kind not in present:
                self.pending.pop(kind)  # never resolved: dropped (M2 will generalise to "unknown person")
        for tr in tracks:
            st = perception.tracks.get(tr.kind)
            appearance = getattr(st, "appearance", None) if tr.visible else None
            if appearance is None:
                continue  # up close / occluded: keep the track's identity (continuity)
            rec = self.memory.resolve(tr.kind, appearance, t)
            if rec is None:
                continue  # ambiguous appearance: keep whatever identity the track already has
            if self.current.get(tr.kind) is not rec:
                self.current[tr.kind] = rec
                for args in self.pending.pop(tr.kind, []):  # "oh, it was you"
                    self.memory.learn(rec, *args, confidence=DEFERRED_CONFIDENCE)
        return self.current

    def learn(self, kind: str, t: float, event: str, emotions, arousal: float, surprise_negative: bool) -> None:
        rec = self.current.get(kind)
        if rec is None:
            self.pending.setdefault(kind, []).append((t, event, emotions, arousal, surprise_negative))
        else:
            self.memory.learn(rec, t, event, emotions, arousal, surprise_negative)


def run(policy: str, seed: int = 0, duration: float = 100.0, agents: dict[str, Agent] | None = None,
        record_every: int | None = None, head_moves_while_walking: bool = False, selector: str = "utility",
        walking_head_limit=None, perception_mode: str = "simulated", use_memory: bool = False,
        use_spatial: bool | None = None):
    """selector: "utility" (behaviour v2, emotion-modulated action selection) or "rules" (v1).
    perception_mode: "simulated" (ground-truth positions + noise) or "vision" (colour + depth images
    from the robot's head camera, nerva.vision)."""
    """Simulate the closed loop. Returns (sim, rows, frames, fired); frames = qpos + mocap snapshots."""
    agents = agents or default_scenario()
    sim = OpenDuckSim(raw_accel=True, obs_noise=True, init_joint_noise=0.02, seed=seed, policy_path=policy,
                      scene_extender=extend_scene)
    world = World(sim.model)
    vision = perception_mode == "vision"
    perception = VisionPerception(seed=seed) if vision else SimulatedPerception(seed=seed)
    eye = mujoco.Renderer(sim.model, EYE_H, EYE_W) if vision else None
    memory = EntityMemory() if use_memory else None
    binder = IdentityBinder(memory) if use_memory else None
    appraiser = MemoryAppraiser(memory) if use_memory else ContextualAppraiser()
    episodic = EpisodicMemory() if use_memory else None
    places = PlaceMemory() if (use_memory if use_spatial is None else use_spatial) else None
    last_seen_anything, last_sleep, sleep_log = 0.0, -1e9, []
    last_touch = -1e9
    affect = CategoricalAffectModel()
    behaviour_cls = {"utility": UtilityBehaviour, "rules": ReactiveBehaviour}[selector]
    if head_moves_while_walking:
        behaviour = behaviour_cls(walking_head_limit=None)
    elif walking_head_limit is not None:
        behaviour = behaviour_cls(walking_head_limit=walking_head_limit)
    else:
        behaviour = behaviour_cls()
    cam = sim.model.camera("robot_eye").id
    decision = behaviour.step(0.0, 0.1, affect.pad, {}, ())
    sim.set_behaviour(decision.command)
    rows, frames, fired = [], [], []
    tracks = ()
    near_fall_armed = True
    d = sim.data
    for k in range(int(round(duration / CTRL_DT))):
        t = k * CTRL_DT
        w, x, y, z = d.qpos[3:7]
        robot_yaw = float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
        robot_xy = d.qpos[0:2].copy()
        for name, agent in agents.items():
            agent.update(t, CTRL_DT, robot_xy, robot_yaw)
            world.place(d, name, agent.xy, agent.yaw)
        if k % PERCEIVE_EVERY == 0:
            mujoco.mj_kinematics(sim.model, d)
            entities = {n: world.entity_position(d, n) for n in agents if n in ("person", "ball")}
            if vision:
                mujoco.mj_forward(sim.model, d)
                eye.update_scene(d, camera=cam)
                rgb = eye.render()
                eye.enable_depth_rendering()
                eye.update_scene(d, camera=cam)
                depth = eye.render()
                eye.disable_depth_rendering()
                pstate = perception.detect_frame(t, rgb, depth, d.cam_xpos[cam].copy(),
                                                 d.cam_xmat[cam].reshape(3, 3).copy(), float(sim.model.cam_fovy[cam]),
                                                 robot_yaw, 1.0 / FRAME_HZ, ego_velocity=d.qvel[0:2].copy())
            else:
                pstate = perception.detect(t, d.cam_xpos[cam].copy(), d.cam_xmat[cam].reshape(3, 3).copy(),
                                           robot_yaw, entities, 1.0 / FRAME_HZ, ego_velocity=d.qvel[0:2].copy())
            tracks = pstate.tracks
            events = list(pstate.events)
            if binder is not None:
                appraiser.identity = binder.update(t, tracks, perception)
            for name, agent in agents.items():  # simulated touch sensing: gentle contact while petted
                if (agent.action == "pet" and np.linalg.norm(agent.xy - robot_xy) < TOUCH_REACH
                        and t - last_touch >= TOUCH_PERIOD_S):
                    events.append(Event("touch_gentle"))
                    last_touch = t
            tilt = float(gm.tilt_deg(d.qpos[3:7][None])[0])
            if near_fall_armed and tilt > NEAR_FALL_TILT_DEG:
                events.append(Event("near_fall"))
                near_fall_armed = False
            elif tilt < NEAR_FALL_TILT_DEG / 2:
                near_fall_armed = True
            for ev in events:
                behaviour.notice(t, ev.kind)
                a = appraiser.appraise(ev, t, tracks)
                if a is not None:
                    new = affect.add(a)
                    elicited = [(e.label, e.intensity) for e in new]
                    fired.append((t, ev.kind, a, elicited))
                    if ev.kind.startswith(("person", "touch")):
                        about = "person"
                    elif ev.kind.startswith("ball"):
                        about = "ball"
                    else:
                        about = None
                    if binder is not None and about and not ev.kind.endswith("_lost"):
                        binder.learn(about, t, ev.kind, elicited, affect.pad.arousal,
                                     a.desirability < 0 and any(lbl == "surprise" for lbl, _ in elicited))
                    if places is not None:
                        places.learn(robot_xy, t, elicited, affect.pad.arousal)
                    if episodic is not None:
                        who_ev = appraiser.identity.get(about) if about else None
                        episodic.encode(t, ev.kind, who_ev.eid if who_ev else None, robot_xy, elicited,
                                        a.relevance, affect.pad.arousal, 1.0 - a.expectedness,
                                        appraiser.novelty(about) if about else 0.0)
            for kind, a in appraiser.observe(t, tracks, 1.0 / FRAME_HZ):
                affect.add(a)
            if episodic is not None:  # consolidation ("sleep") when nothing has been in view for a while
                if any(tr.visible for tr in tracks):
                    last_seen_anything = t
                elif t - last_seen_anything > IDLE_BEFORE_SLEEP_S and t - last_sleep > SLEEP_EVERY_S:
                    sleep_log.append((round(t, 1), episodic.consolidate(t, memory)))
                    last_sleep = t
            if places is not None:
                places.observe(robot_xy, t, 1.0 / FRAME_HZ)
                behaviour.explore_bearing = places.explore_heading(robot_xy, robot_yaw)[0]
            pad = affect.step(1.0 / FRAME_HZ)
            emotions = {lbl: sum(e.intensity for e in affect.emotions if e.label == lbl) for lbl in EMOTIONS}
            salience = {tr.kind: appraiser.novelty(tr.kind) for tr in tracks}
            decision = behaviour.step(t, 1.0 / FRAME_HZ, pad, emotions, tracks, salience)
            sim.set_behaviour(decision.command)
            sim.set_head_offset(*decision.head)
        sim.step_physics(10)
        pad = affect.pad
        tilt = float(gm.tilt_deg(d.qpos[3:7][None])[0])
        person_xy = agents["person"].xy if "person" in agents else np.array(HIDDEN)
        person_b_xy = agents["person_b"].xy if "person_b" in agents else np.array(HIDDEN)
        who = appraiser.identity.get("person") if use_memory else None
        rows.append({
            "t": round(t + CTRL_DT, 3), "valence": pad.valence, "arousal": pad.arousal, "dominance": pad.dominance,
            **{lbl: sum(e.intensity for e in affect.emotions if e.label == lbl) for lbl in EMOTIONS},
            "mode": decision.mode, "reason": decision.reason, "target": decision.target or "",
            "cmd_vx": decision.command.vx, "cmd_yaw": decision.command.yaw_rate,
            "e_tempo": decision.command.style_vector.tempo, "e_torso_pitch": decision.command.style_vector.torso_pitch,
            "head_pitch": decision.head[1], "head_yaw": decision.head[2], "head_roll": decision.head[3],
            "robot_x": float(d.qpos[0]), "robot_y": float(d.qpos[1]),
            "person_x": float(person_xy[0]), "person_y": float(person_xy[1]),
            "dist_person": float(np.linalg.norm(person_xy - d.qpos[0:2])),
            "dist_person_b": float(np.linalg.norm(person_b_xy - d.qpos[0:2])),
            "identity": who.eid if who is not None else "",
            "mem_threat": who.threat if who is not None else float("nan"),
            "mem_warmth": who.warmth if who is not None else float("nan"),
            "dist_ball": float(np.linalg.norm(agents["ball"].xy - d.qpos[0:2])) if "ball" in agents else np.nan,
            "sees_person": any(tr.kind == "person" and tr.visible for tr in tracks),
            "sees_ball": any(tr.kind == "ball" and tr.visible for tr in tracks),
            "tilt_deg": tilt,
        })
        if record_every and k % record_every == 0:
            boxes = None
            if vision:  # what the vision actually found: (kind, row0, col0, row1, col1) in EYE_H x EYE_W pixels
                boxes = [(bl.kind, *bl.pixels.min(0), *bl.pixels.max(0)) for bl in perception.last_blobs]
            frames.append((d.qpos.copy(), d.mocap_pos.copy(), d.mocap_quat.copy(), tracks, boxes))
    if use_memory:
        sim.memory, sim.episodic, sim.sleep_log = memory, episodic, sleep_log  # for inspection by callers
    sim.places = places
    return sim, rows, frames, fired
