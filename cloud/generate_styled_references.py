"""Generate the seven-reference R1 pilot with the pinned upstream generator.

The upstream ``auto_waddle.py`` only reads its in-tree ``medium.json`` preset.
R1 therefore changes that preset in the disposable pinned checkout, runs the
upstream sweep unmodified, and restores the original bytes in a ``finally``
block. Every effective preset and generator log is retained with the output.
"""

from __future__ import annotations

import argparse
import ast
import json
import pickle
import shutil
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path

# The generator's own uv environment (Python 3.10, numpy) runs this script; the NERVA
# package is not installed there, so import the pure-numpy validator from the repo.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nerva.reference_validation import substitute_invalid, validate_reference  # noqa: E402

MAX_REPAIR_ATTEMPTS = 2
# Upper bound on gaits replaced by a neighbour; more than this means something is broken.
MAX_SUBSTITUTIONS = 4


PILOT_STYLES: tuple[tuple[str, tuple[float, float, float]], ...] = (
    ("neutral", (0.0, 0.0, 0.0)),
    ("e1_neg", (-1.0, 0.0, 0.0)),
    ("e1_pos", (1.0, 0.0, 0.0)),
    ("e2_neg", (0.0, -1.0, 0.0)),
    ("e2_pos", (0.0, 1.0, 0.0)),
    ("e3_neg", (0.0, 0.0, -1.0)),
    ("e3_pos", (0.0, 0.0, 1.0)),
)


def apply_style(base: dict, style: tuple[float, float, float]) -> dict:
    """Apply the preregistered e1/e2/e3 mapping to an upstream preset copy."""
    if len(style) != 3 or any(not -1.0 <= value <= 1.0 for value in style):
        raise ValueError("style must contain three values in [-1, 1]")
    for key in ("single_support_duration", "walk_foot_height", "walk_trunk_pitch"):
        if key not in base:
            raise KeyError(f"upstream preset is missing {key}")
    e1, e2, e3 = style
    result = deepcopy(base)
    result["single_support_duration"] = round(float(base["single_support_duration"]) * (1 - 0.25 * e1), 6)
    result["walk_foot_height"] = round(float(base["walk_foot_height"]) * (1 + 0.5 * e2), 6)
    result["walk_trunk_pitch"] = round(float(base["walk_trunk_pitch"]) + 6.0 * e3, 6)
    return result


def run_logged(command: list[str], cwd: Path, log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as stream:
        subprocess.run(command, cwd=cwd, check=True, text=True, stdout=stream,
                       stderr=subprocess.STDOUT)


def gait_parameters_from_log(log_path: Path) -> dict:
    """The exact preset upstream gait_generator.py used, from its 'gait_parameters {...}' log line."""
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("gait_parameters "):
            return ast.literal_eval(line[len("gait_parameters "):])
    raise ValueError(f"no gait_parameters line in {log_path}")


def regenerate_gait(generator_root: Path, recordings: Path, key: str, attempt: int) -> None:
    """Re-run one gait of a sweep with its original parameters (overwrites its recording)."""
    matches = sorted(recordings.glob(f"*_{key}.json"))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one recording for gait {key}, found {len(matches)}")
    index = matches[0].name.split("_", 1)[0]
    params = gait_parameters_from_log(recordings / "log" / f"{index}.log")
    preset = recordings / "log" / f"{index}_repair_preset.json"
    preset.write_text(json.dumps(params, indent=2) + "\n", encoding="utf-8")
    run_logged([
        "uv", "run", "python", str(generator_root / "open_duck_reference_motion_generator/gait_generator.py"),
        "--duck", "open_duck_mini_v2", "--preset", str(preset), "--name", index,
        "--output_dir", str(recordings),
    ], generator_root, recordings / "log" / f"{index}_repair{attempt}.log")


def fit_validated(generator_root: Path, recordings: Path, destination: Path,
                  max_attempts: int = MAX_REPAIR_ATTEMPTS) -> dict:
    """Fit polynomials; regenerate gaits with backward knees; substitute the rest.

    2026-09-29 R0: the generator's initial inverse-kinematics placement intermittently
    lands on the mirror-image knee solution; re-running the gait usually fixes it.
    2026-09-29 R1: a few extreme gaits pass the knee through full extension mid-walk
    on every regeneration (upstream's shipped file has the same two); after
    max_attempts they are replaced by their nearest valid grid neighbour. Raises if
    more than MAX_SUBSTITUTIONS gaits would need that.
    """
    fit_poly = generator_root / "scripts/fit_poly.py"
    repaired: dict[str, int] = {}
    for attempt in range(max_attempts + 1):
        generated = generator_root / "polynomial_coefficients.pkl"
        generated.unlink(missing_ok=True)
        run_logged(["uv", "run", "python", str(fit_poly), "--ref_motion", str(recordings)],
                   generator_root, recordings / "log" / f"fit{attempt}.log")
        with generated.open("rb") as stream:
            fitted = pickle.load(stream)  # trusted, locally generated output
        report = validate_reference(fitted)
        substituted: dict[str, str] = {}
        if not report["valid"] and attempt == max_attempts:
            invalid = report["backward_knee_gaits"] + report["nonfinite_gaits"]
            if len(invalid) > MAX_SUBSTITUTIONS:
                raise RuntimeError(f"{len(invalid)} gaits still invalid after {max_attempts} repairs: {invalid}")
            fitted, substituted = substitute_invalid(fitted, invalid)
            unsubstituted = report
            report = validate_reference(fitted)
            report["before_substitution"] = unsubstituted
            with generated.open("wb") as stream:
                pickle.dump(fitted, stream)
        if report["valid"]:
            shutil.move(str(generated), destination)
            report.update(repair_attempts=attempt, repaired_gaits=repaired,
                          substituted_gaits=substituted, fitted_keys=len(fitted))
            return report
        for key in report["backward_knee_gaits"]:
            repaired[key] = repaired.get(key, 0) + 1
            regenerate_gait(generator_root, recordings, key, attempt)
    raise AssertionError("unreachable")


def generate_pilot(generator_root: Path, output_root: Path, jobs: int) -> dict:
    if jobs <= 0:
        raise ValueError("jobs must be positive")
    generator_root = generator_root.resolve()
    output_root = output_root.resolve()
    preset_path = generator_root / (
        "open_duck_reference_motion_generator/robots/open_duck_mini_v2/placo_presets/medium.json"
    )
    auto_waddle = generator_root / "scripts/auto_waddle.py"
    fit_poly = generator_root / "scripts/fit_poly.py"
    for required in (preset_path, auto_waddle, fit_poly):
        if not required.is_file():
            raise FileNotFoundError(required)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to mix R1 with existing output: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    original_bytes = preset_path.read_bytes()
    base = json.loads(original_bytes)
    results: list[dict] = []
    started = time.time()
    try:
        for name, style in PILOT_STYLES:
            style_root = output_root / name
            recordings = style_root / "recordings"
            style_root.mkdir(parents=True)
            effective = apply_style(base, style)
            rendered = json.dumps(effective, indent=2) + "\n"
            (style_root / "preset.json").write_text(rendered, encoding="utf-8")
            preset_path.write_text(rendered, encoding="utf-8")

            style_started = time.time()
            run_logged([
                "uv", "run", "python", str(auto_waddle), "-j", str(jobs),
                "--duck", "open_duck_mini_v2", "--sweep", "--output_dir", str(recordings),
            ], generator_root, style_root / "generate.log")

            pickle_name = f"polynomial_coefficients_{name}.pkl"
            destination = output_root / pickle_name
            validation = fit_validated(generator_root, recordings, destination)

            gait_files = list(recordings.glob("*.json"))
            child_logs = list((recordings / "log").glob("*.log"))
            failed_logs = [
                path.name for path in child_logs
                if "Traceback (most recent call last)" in path.read_text(encoding="utf-8", errors="replace")
            ]
            result = {
                "name": name,
                "style": list(style),
                "pickle": pickle_name,
                "parameters": {
                    "single_support_duration": effective["single_support_duration"],
                    "walk_foot_height": effective["walk_foot_height"],
                    "walk_trunk_pitch": effective["walk_trunk_pitch"],
                },
                "candidate_logs": len(child_logs),
                "recordings": len(gait_files),
                "fitted_keys": validation["fitted_keys"],
                "validation": validation,
                "failed_logs": failed_logs,
                "seconds": time.time() - style_started,
            }
            if not gait_files or not validation["fitted_keys"]:
                raise RuntimeError(f"style {name} produced no usable references")
            results.append(result)
            (output_root / "progress.json").write_text(
                json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
    finally:
        preset_path.write_bytes(original_bytes)

    styles = [{"style": item["style"], "pickle": item["pickle"]} for item in results]
    (output_root / "styles.json").write_text(
        json.dumps(styles, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary = {
        "styles": results,
        "style_count": len(results),
        "total_seconds": time.time() - started,
        "jobs": jobs,
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--repair-recordings", type=Path,
                        help="validate/repair an existing sweep (e.g. R0) instead of generating R1")
    parser.add_argument("--repair-output", type=Path, help="validated pickle path for --repair-recordings")
    args = parser.parse_args()
    if args.repair_recordings:
        report = fit_validated(args.generator_root.resolve(), args.repair_recordings.resolve(),
                               args.repair_output.resolve())
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    summary = generate_pilot(args.generator_root, args.output_root, args.jobs)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
