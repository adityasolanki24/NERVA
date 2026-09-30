"""The reactive-behaviour scene must not change the robot (docs/reactive_behaviour_design.md §3)."""

import mujoco
import numpy as np
import pytest

pytest.importorskip("playground.open_duck_mini_v2.mujoco_infer")

from nerva.interfaces import BehaviourCommand  # noqa: E402
from nerva.open_duck_sim import OpenDuckSim  # noqa: E402
from nerva.world import FACE_HEIGHT, Mover, World, extend_scene  # noqa: E402


def test_extended_scene_leaves_robot_dynamics_unchanged():
    plain, extended = OpenDuckSim(), OpenDuckSim(scene_extender=extend_scene)
    world = World(extended.model, {"person": Mover([(0, 0.5, 0.0), (1, 0.4, 0.0)]),  # walks INTO the robot
                                   "ball": Mover([(0, 0.2, 0.0)])})
    for sim in (plain, extended):
        sim.set_behaviour(BehaviourCommand(vx=0.15))
    for k in range(100):
        world.update(extended.data, k * 0.02)
        plain.step_physics(10)
        extended.step_physics(10)
    np.testing.assert_array_equal(plain.data.qpos, extended.data.qpos)
    assert extended.model.opt.timestep == plain.model.opt.timestep


def test_entities_move_and_face_their_direction():
    sim = OpenDuckSim(scene_extender=extend_scene)
    world = World(sim.model, {"person": Mover([(0, 2.0, 0.0), (2, 0.0, 0.0)])})
    world.update(sim.data, 1.0)
    np.testing.assert_allclose(world.entity_position(sim.data, "person"), [1.0, 0.0, FACE_HEIGHT])
    quat = sim.data.mocap_quat[sim.model.body_mocapid[sim.model.body("person").id]]
    assert abs(quat[3]) == pytest.approx(1.0, abs=1e-6)  # yaw = pi: moving in -x, facing the robot


def test_robot_eye_camera_looks_forward():
    sim = OpenDuckSim(scene_extender=extend_scene)
    mujoco.mj_forward(sim.model, sim.data)
    cam = sim.model.camera("robot_eye").id
    forward = -sim.data.cam_xmat[cam].reshape(3, 3)[:, 2]
    assert forward[0] > 0.9  # robot starts facing world +x
