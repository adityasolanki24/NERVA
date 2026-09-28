"""Command-confound check for RQ1b.

Speed matching gives each style a DIFFERENT command (Style −1 needs a much higher
vx than Style +1). The command is itself a policy input, so a difference at
matched speed could come from the command rather than the gait clock. The
calibration sweep contains every style at IDENTICAL commands; if a difference
also appears there, it is not a command effect.

Usage:  <venv>/Scripts/python experiments/expressive_locomotion/same_command_check.py <results_dir>
"""

import csv
import sys
from pathlib import Path

import numpy as np

METRICS = ["v_fwd", "pitch_mean_deg", "lift_left_mm", "roll_std_deg", "joint_range_rms_rad", "power_w"]
COMMANDS = ("0.105", "0.12", "0.135", "0.15")  # where all styles walk (above the dead zone)


def main(results_dir: Path) -> None:
    rows = list(csv.DictReader((results_dir / "calibration_trials.csv").open()))
    print("Same command, calibration seeds, mean over seeds")
    print(f"{'command':>8} {'style':>6} " + " ".join(f"{m:>20}" for m in METRICS))
    for c in COMMANDS:
        for s in ("-1.0", "0.0", "1.0"):
            rs = [r for r in rows if r["command_vx"] == c and r["style"] == s]
            vals = [np.mean([float(r[m]) for r in rs]) for m in METRICS]
            print(f"{c:>8} {s:>6} " + " ".join(f"{v:20.4g}" for v in vals))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
