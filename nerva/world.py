"""A scene for reactive behaviour: the upstream Open Duck scene plus things to react to.

Adds, via mujoco.MjSpec, without touching upstream files (docs/reactive_behaviour_design.md §3):
  - visual-only mocap entities (no collision with anything), moved by scripted Movers:
      "person"  1.7 m figure with a face on the +x side of its head
      "ball"    0.12 m sphere
  - a camera "robot_eye" on the robot's head, looking forward along the head, for a robot's-eye view.
Entities add no degrees of freedom and are appended after all existing bodies/geoms, so the robot's
indices and dynamics are unchanged (tests/test_world.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

PERSON_HEIGHT = 1.7  # m
FACE_HEIGHT = 1.58  # m, centre of the person's face
BALL_RADIUS = 0.06  # m
ENTITY_KINDS = ("person", "ball")


@dataclass
class Mover:
    """Piecewise-linear path: waypoints [(t_s, x_m, y_m), ...]; holds the first/last point outside them.

    Heading follows the direction of travel; while stationary it keeps the last heading, or
    `face_yaw` if given (e.g. a person turned toward the robot).
    """

    waypoints: list[tuple[float, float, float]]
    face_yaw: float | None = None
    _last_yaw: float = field(default=0.0, init=False)

    def position(self, t: float) -> np.ndarray:
        ts = np.array([w[0] for w in self.waypoints])
        xs = np.array([w[1] for w in self.waypoints])
        ys = np.array([w[2] for w in self.waypoints])
        return np.array([np.interp(t, ts, xs), np.interp(t, ts, ys)])

    def yaw(self, t: float, dt: float = 0.05) -> float:
        d = self.position(t + dt) - self.position(t)
        if np.linalg.norm(d) > 1e-4:
            self._last_yaw = float(np.arctan2(d[1], d[0]))
        elif self.face_yaw is not None:
            self._last_yaw = self.face_yaw
        return self._last_yaw


def _add_person(world: mujoco.MjsBody) -> None:
    body = world.add_body(name="person", mocap=True, pos=[5.0, 0.0, 0.0])
    skin, cloth, dark = [0.87, 0.72, 0.60, 1], [0.25, 0.40, 0.65, 1], [0.1, 0.1, 0.1, 1]

    def geom(**kw):
        g = body.add_geom(contype=0, conaffinity=0, **kw)
        return g

    cap = mujoco.mjtGeom.mjGEOM_CAPSULE
    for y in (-0.1, 0.1):  # legs
        geom(type=cap, size=[0.06, 0.0, 0.0], fromto=[0, y, 0.06, 0, y, 0.85], rgba=[0.2, 0.2, 0.25, 1])
    geom(type=cap, size=[0.17, 0.0, 0.0], fromto=[0, 0, 0.95, 0, 0, 1.35], rgba=cloth)  # torso
    for y in (-0.24, 0.24):  # arms
        geom(type=cap, size=[0.045, 0.0, 0.0], fromto=[0, y, 1.40, 0.02, y, 0.95], rgba=cloth)
    geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.12, 0, 0], pos=[0, 0, FACE_HEIGHT], rgba=skin)  # head
    for y in (-0.04, 0.04):  # eyes: the face is on the +x side
        geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.018, 0, 0], pos=[0.11, y, FACE_HEIGHT + 0.02], rgba=dark)


def _add_ball(world: mujoco.MjsBody) -> None:
    body = world.add_body(name="ball", mocap=True, pos=[5.0, 2.0, BALL_RADIUS])
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[BALL_RADIUS, 0, 0], rgba=[0.95, 0.45, 0.10, 1],
                  contype=0, conaffinity=0)


def _add_robot_eye(spec: mujoco.MjSpec) -> None:
    """Camera on the head body, looking along world +x (robot forward) in the home pose."""
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    head = model.body("head_assembly").id
    site = model.site("head").id
    r_body = data.xmat[head].reshape(3, 3)
    # MuJoCo cameras look along their -z axis with +y up: -z = world +x, +y = world +z.
    r_cam = np.column_stack([[0, -1, 0], [0, 0, 1], [-1, 0, 0]])  # columns: cam x, y, z in world
    r_rel = r_body.T @ r_cam
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, r_rel.flatten())
    pos = r_body.T @ (data.site_xpos[site] - data.xpos[head])
    spec.body("head_assembly").add_camera(name="robot_eye", pos=pos, quat=quat, fovy=70.0)


def extend_scene(spec: mujoco.MjSpec) -> None:
    """OpenDuckSim scene_extender: person, ball and the robot_eye camera."""
    _add_robot_eye(spec)  # compiles a copy first, so do it before adding bodies
    _add_person(spec.worldbody)
    _add_ball(spec.worldbody)


class World:
    """Moves the mocap entities of an extended scene along their Movers."""

    def __init__(self, model: mujoco.MjModel, movers: dict[str, Mover]):
        unknown = set(movers) - set(ENTITY_KINDS)
        if unknown:
            raise ValueError(f"unknown entities {unknown}")
        self.movers = movers
        self._mocap = {n: model.body_mocapid[model.body(n).id] for n in ENTITY_KINDS}

    def update(self, data: mujoco.MjData, t: float) -> None:
        for name, mover in self.movers.items():
            i = self._mocap[name]
            xy = mover.position(t)
            z = BALL_RADIUS if name == "ball" else 0.0
            data.mocap_pos[i] = [xy[0], xy[1], z]
            yaw = mover.yaw(t)
            data.mocap_quat[i] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]

    def entity_position(self, data: mujoco.MjData, name: str) -> np.ndarray:
        """World position of the entity's salient point: the person's face, the ball's centre."""
        p = data.mocap_pos[self._mocap[name]].copy()
        if name == "person":
            p[2] = FACE_HEIGHT
        return p
