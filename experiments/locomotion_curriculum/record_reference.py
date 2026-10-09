"""Isolated upstream recorder adaptation; never writes upstream files."""
import argparse
from pathlib import Path
import sys


def adapt(source):
    replacements = {
        "pwe.set_traj(args.dx, args.dy, args.dtheta + 0.00955)":
            "pwe.set_traj(args.dx, args.dy, args.dtheta)",
        'if args.debug:\n    episode["Debug_info"] = []':
            'episode["FrameTimes"] = []\nif args.debug:\n    episode["Debug_info"] = []',
        'if prev_initialized:\n            if args.hardware:':
            'if prev_initialized:\n            episode["FrameTimes"].append(pwe.t)\n            if args.hardware:',
    }
    for old, new in replacements.items():
        if source.count(old) != 1:
            raise ValueError("upstream recorder source mismatch")
        source = source.replace(old, new)
    return source


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    source = args.source.resolve()
    sys.path.insert(0, str(source.parent))
    sys.argv = [str(source), *remaining]
    exec(compile(adapt(source.read_text(encoding="utf-8")), str(source), "exec"),
         {"__name__": "__main__", "__file__": str(source)})


if __name__ == "__main__":
    main()
