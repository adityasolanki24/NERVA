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


def adapt_engine(source):
    return replace_once(source, {
        'knee_limits = knee_limits or [0.2, 0.01]': 'knee_limits = knee_limits or [0.01, np.pi / 2]',
        'self.solver.enable_joint_limits(False)': 'self.solver.enable_joint_limits(True)',
        '# Placing the robot in the initial position':
            'self.robot.set_joint("left_knee", 1.2)\n        self.robot.set_joint("right_knee", 1.2)\n'
            '        self.robot.update_kinematics()\n        # Placing the robot in the initial position',
        'def tick(self, dt, left_contact=True, right_contact=True):':
            'def tick(self, dt, left_contact=True, right_contact=True):\n        self.solver.dt = dt / REFINE',
    })


def static_support(left, right):
    feet = np.asarray([left, right])
    if feet.shape != (2, 4, 4) or not np.isfinite(feet).all() or np.max(np.abs(feet[:, 2, 3])) > .001:
        raise ValueError("static feet are not on the floor")
    return [1, 1]


def adapt(source, repair=False):
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
    return replace_once(source, replacements)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--repair", action="store_true")
    args, remaining = parser.parse_known_args()
    source = args.source.resolve()
    sys.path.insert(0, str(source.parent))
    if args.repair:
        engine_path = source.with_name("placo_walk_engine.py")
        engine = types.ModuleType("placo_walk_engine")
        engine.__file__ = str(engine_path)
        exec(compile(adapt_engine(engine_path.read_text(encoding="utf-8")), str(engine_path), "exec"), engine.__dict__)
        sys.modules["placo_walk_engine"] = engine
    sys.argv = [str(source), *remaining]
    exec(compile(adapt(source.read_text(encoding="utf-8"), args.repair), str(source), "exec"),
         {"__name__": "__main__", "__file__": str(source), "static_support": static_support})


if __name__ == "__main__":
    main()
