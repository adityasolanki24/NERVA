"""Isolated upstream recorder adaptation; never writes upstream files."""
import argparse
from pathlib import Path
import sys
import types

import numpy as np


def replace_once(source, replacements):
    for old, new in replacements.items():
        if source.count(old) != 1:
            raise ValueError("upstream source mismatch")
        source = source.replace(old, new)
    return source


def pivot_step(offset, theta):
    offset = np.asarray(offset, dtype=float)
    if offset.shape != (2,) or not np.isfinite(offset).all() or not np.isfinite(theta):
        raise ValueError("invalid planar pivot")
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return (np.eye(2) - rotation) @ offset


def verify_planner(placo):
    parameters = placo.HumanoidParameters()
    parameters.feet_spacing = .16
    parameters.walk_max_dtheta = 1.
    parameters.walk_max_dx_forward = .08
    parameters.walk_max_dx_backward = .03
    parameters.walk_max_dy = .1
    parameters.walk_dtheta_spacing = 0.
    left, right = np.eye(4), np.eye(4)
    left[1, 3], right[1, 3] = .08, -.08
    for theta in (-.162, .162):
        planner = placo.FootstepsPlannerRepetitive(parameters)
        planner.configure(.001, -.004, theta, 5)
        foot = planner.plan(placo.HumanoidRobot_Side.left, left, right)[2]
        # 0.6.3 cannot convert Footstep.frame's Eigen::Affine3d; its polygon is exposed.
        corners = np.array([[-1, 1], [1, 1], [1, -1], [-1, -1]]) * [foot.foot_length / 2, foot.foot_width / 2]
        rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        expected = corners @ rotation.T + [.001, .08 - .004]
        actual = np.asarray(foot.support_polygon())
        if not np.allclose(actual, expected, atol=1e-8, rtol=0):
            raise ValueError("installed planner geometry differs")


def adapt_engine(source, pivot=False):
    replacements = {
        'knee_limits = knee_limits or [0.2, 0.01]': 'knee_limits = knee_limits or [0.01, np.pi / 2]',
        'self.solver.enable_joint_limits(False)': 'self.solver.enable_joint_limits(True)',
        '# Placing the robot in the initial position':
            'self.robot.set_joint("left_knee", 1.2)\n        self.robot.set_joint("right_knee", 1.2)\n'
            '        self.robot.update_kinematics()\n        # Placing the robot in the initial position',
        'def tick(self, dt, left_contact=True, right_contact=True):':
            'def tick(self, dt, left_contact=True, right_contact=True):\n        self.solver.dt = dt / REFINE',
    }
    if pivot:
        replacements['print(self.get_angles())'] = (
            'print(self.get_angles())\n'
            '        base = self.robot.get_T_world_fbase()\n'
            '        center = (self.robot.get_T_world_left()[:3, 3] + self.robot.get_T_world_right()[:3, 3]) / 2\n'
            '        yaw = np.arctan2(base[1, 0], base[0, 0])\n'
            '        rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])\n'
            '        self.neutral_pivot = rotation.T @ (base[:2, 3] - center[:2])')
        replacements['def set_traj(self, d_x, d_y, d_theta):'] = (
            'def set_traj(self, d_x, d_y, d_theta):\n'
            '        if d_x == 0 and d_y == 0 and d_theta != 0:\n'
            '            d_x, d_y = pivot_step(self.neutral_pivot, d_theta)')
    return replace_once(source, replacements)


def static_support(left, right):
    feet = np.asarray([left, right])
    if feet.shape != (2, 4, 4) or not np.isfinite(feet).all() or np.max(np.abs(feet[:, 2, 3])) > .001:
        raise ValueError("static feet are not on the floor")
    return [1, 1]


def adapt(source, repair=False, pivot=False):
    replacements = {
        "pwe.set_traj(args.dx, args.dy, args.dtheta + 0.00955)":
            "pwe.set_traj(args.dx, args.dy, args.dtheta)",
        'if args.debug:\n    episode["Debug_info"] = []':
            'episode["FrameTimes"] = []\nif args.debug:\n    episode["Debug_info"] = []',
        'if prev_initialized:\n            if args.hardware:':
            'if prev_initialized:\n            episode["FrameTimes"].append(pwe.t)\n            if args.hardware:',
    }
    if repair:
        replacements['foot_contacts = pwe.get_current_support_phase()'] = (
            'foot_contacts = static_support(T_world_leftFoot, T_world_rightFoot) if args.stand '
            'else pwe.get_current_support_phase()')
    if pivot:
        replacements['pwe.set_traj(args.dx, args.dy, args.dtheta + 0.00955)'] += (
            '\nepisode["NeutralPivot"] = {"offset_body_xy": pwe.neutral_pivot.tolist(), '
            '"planner_step": [float(pwe.d_x), float(pwe.d_y), float(pwe.d_theta)], '
            '"installed_planner_geometry_verified": True}')
    return replace_once(source, replacements)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--repair", action="store_true")
    parser.add_argument("--pivot", action="store_true")
    args, remaining = parser.parse_known_args()
    if args.pivot and not args.repair:
        raise ValueError("pivot requires repaired recorder")
    source = args.source.resolve()
    sys.path.insert(0, str(source.parent))
    if args.repair:
        engine_path = source.with_name("placo_walk_engine.py")
        engine = types.ModuleType("placo_walk_engine")
        engine.__file__ = str(engine_path)
        engine.pivot_step = pivot_step
        exec(compile(adapt_engine(engine_path.read_text(encoding="utf-8"), args.pivot), str(engine_path), "exec"), engine.__dict__)
        if args.pivot:
            verify_planner(engine.placo)
        sys.modules["placo_walk_engine"] = engine
    sys.argv = [str(source), *remaining]
    exec(compile(adapt(source.read_text(encoding="utf-8"), args.repair, args.pivot), str(source), "exec"),
         {"__name__": "__main__", "__file__": str(source), "static_support": static_support})


if __name__ == "__main__":
    main()
