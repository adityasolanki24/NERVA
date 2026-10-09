"""Preregistered saved-trace geometry analysis; no simulation or cloud work."""
import argparse
import contextlib
import hashlib
import io
import subprocess
from pathlib import Path

import numpy as np

from experiments.locomotion_curriculum.gate import velocities, write_json
from nerva.analysis.gait_metrics import quat_to_rpy
from nerva.sim.open_duck import OPEN_DUCK_ROOT, REFERENCE


def design(t, yaw, drift):
    n = len(t)
    out = np.zeros((2 * n, 6 if drift else 4))
    c, s = np.cos(yaw), np.sin(yaw)
    out[0::2, 0], out[1::2, 1] = 1, 1
    out[0::2, 2], out[0::2, 3] = c, -s
    out[1::2, 2], out[1::2, 3] = s, c
    if drift:
        out[0::2, 4], out[1::2, 5] = t - 5, t - 5
    return out


def analyse(t, xy, yaw):
    if (xy.shape != (len(t), 2) or yaw.shape != t.shape
            or not all(np.isfinite(x).all() for x in (t, xy, yaw))
            or len(t) != 1000 or not np.allclose(t, np.arange(1, 1001) * .02)):
        raise ValueError("expected complete finite 20-second trace at 50 Hz")
    train, test = (t >= 5) & (t < 12.5), (t >= 12.5) & (t <= 20)
    models = {}
    valid = True
    for drift, name in ((False, "orbit"), (True, "orbit_drift")):
        a = design(t[train], yaw[train], drift)
        coef, _, rank, _ = np.linalg.lstsq(a, xy[train].ravel(), rcond=None)
        cond = float(np.linalg.cond(a))
        valid &= rank == a.shape[1] and cond <= 10000
        residual = (design(t[test], yaw[test], drift) @ coef).reshape(-1, 2) - xy[test]
        models[name] = {"rank": int(rank), "condition_number": cond,
                        "centre_xy_m": coef[:2].tolist(), "offset_body_xy_m": coef[2:4].tolist(),
                        "heldout_rmse_m": float(np.sqrt(np.mean(np.sum(residual ** 2, axis=1))))}
        if drift:
            models[name].update(drift_world_xy_m_s=coef[4:].tolist(), drift_speed_m_s=float(np.linalg.norm(coef[4:])))
    orbit, drift = models["orbit"], models["orbit_drift"]
    bounded = orbit["heldout_rmse_m"] <= .01 and drift["drift_speed_m_s"] <= .01
    sustained = drift["drift_speed_m_s"] >= .02 and drift["heldout_rmse_m"] <= .5 * orbit["heldout_rmse_m"]
    models["classification"] = ("bounded_orbit" if bounded else "sustained_world_drift" if sustained
                                else "inconclusive") if valid else "inconclusive"
    return models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", type=Path, default=Path("experiments/cloud_runs/b2-gate-local"))
    ap.add_argument("--out", type=Path, default=Path("experiments/locomotion_curriculum/results_turn"))
    args = ap.parse_args()
    if args.out.exists():
        raise FileExistsError("never overwrite diagnostic results")
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise RuntimeError("commit implementation before analysis")
    from playground.common.poly_reference_motion_numpy import PolyReferenceMotion
    with contextlib.redirect_stdout(io.StringIO()):
        ref = PolyReferenceMotion(str(REFERENCE))
    rows, hashes, audit = [], {}, {}
    for direction, rate in (("left", .6), ("right", -.6)):
        ix, iy, iz = ref.vel_to_index(0, 0, rate)
        frames = np.array([ref.sample_polynomial(p, ref.data_array[ix][iy][iz]) for p in np.arange(50) / 50])
        if frames.shape != (50, 40) or not np.isfinite(frames).all():
            raise ValueError("malformed reference")
        audit[direction] = {"selected_key": [ref.dxs[ix], ref.dys[iy], ref.dthetas[iz]],
                            "mean_linear_velocity": frames[:, 34:37].mean(axis=0).tolist(),
                            "mean_angular_velocity": frames[:, 37:40].mean(axis=0).tolist()}
        for seed in range(5):
            name = f"steady_turn_{direction}_{seed}_primary.npz"
            path = args.raw_dir / name
            hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
            with np.load(path) as arrays:
                result = analyse(arrays["t"], arrays["base_pos"][:, :2], quat_to_rpy(arrays["base_quat"])[:, 2])
                mask = arrays["t"] >= 5
                lateral = float(velocities(arrays)[mask, 1].mean())
            rows.append({"direction": direction, "seed": seed, **result,
                         "mean_heading_lateral_m_s": lateral,
                         "reference_lateral_sign_agrees": bool(np.sign(lateral) == np.sign(frames[:, 35].mean()))})
    counts = {d: {c: sum(r["direction"] == d and r["classification"] == c for r in rows)
                   for c in ("bounded_orbit", "sustained_world_drift", "inconclusive")} for d in ("left", "right")}
    supported = [c for c in ("bounded_orbit", "sustained_world_drift") if all(counts[d][c] >= 4 for d in counts)]
    report = {"counts": counts, "aggregate": supported[0] if supported else "inconclusive", "trials": rows,
              "reference_audit": audit, "original_gate": "failed; unchanged"}
    provenance = {"preregistration_commit": "69dc1e0",
                  "implementation_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  "upstream_commit": subprocess.check_output(["git", "-C", str(OPEN_DUCK_ROOT / "Open_Duck_Playground"),
                                                              "rev-parse", "HEAD"], text=True).strip(),
                  "input_sha256": hashes, "reference_sha256": hashlib.sha256(REFERENCE.read_bytes()).hexdigest(),
                  "cloud_work": False}
    args.out.mkdir(parents=True)
    write_json(args.out / "report.json", report)
    write_json(args.out / "provenance.json", provenance)
    print(report["aggregate"], counts)


if __name__ == "__main__":
    main()
