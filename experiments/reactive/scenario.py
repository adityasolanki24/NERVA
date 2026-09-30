"""Reactive-behaviour scenario: scripted agents in the scene, the robot's full loop reacting to them.

Loop per 20 ms control step (docs/reactive_behaviour_design.md §2):
  agents move (their scripts may refer to where the robot is NOW, so the scene stays in view)
  every 0.1 s: head camera → SimulatedPerception → events + tracks
               → ContextualAppraiser → CategoricalAffectModel (v0.2, with interest) → PAD, emotions
               → ReactiveBehaviour → velocity command + S1 style vector + head gaze
  OpenDuckSim with the S1 policy (unchanged), 500 Hz physics.
Nothing about the robot's reactions is scripted: only the person and the ball follow scripts.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from nerva import gait_metrics as gm
from nerva.affect import CategoricalAffectModel
from nerva.appraisal import ContextualAppraiser
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
        if action == "approach":
            speed, stop = p
            step = min(speed * dt, max(0.0, d - stop))
            self.xy = self.xy + step * to_robot / max(d, 1e-6)
        elif action == "leave":
            (speed,) = p
            self.yaw = float(np.arctan2(-to_robot[1], -to_robot[0]))
            self.xy = self.xy - speed * dt * to_robot / max(d, 1e-6)


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


def run(policy: str, seed: int = 0, duration: float = 100.0, agents: dict[str, Agent] | None = None,
        record_every: int | None = None, head_moves_while_walking: bool = False, selector: str = "utility",
        walking_head_limit=None, perception_mode: str = "simulated"):
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
    appraiser = ContextualAppraiser()
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
            entities = {n: world.entity_position(d, n) for n in agents}
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
                    fired.append((t, ev.kind, a, [(e.label, e.intensity) for e in new]))
            for kind, a in appraiser.observe(t, tracks, 1.0 / FRAME_HZ):
                affect.add(a)
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
    return sim, rows, frames, fired
