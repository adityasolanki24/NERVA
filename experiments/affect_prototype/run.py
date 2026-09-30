"""Affect prototype demo: a scripted timeline of synthetic events → appraisal → emotions → PAD.

Simulation only; the robot is not involved. Shows how the persistent PAD state
evolves, with every step traceable to docs/affect_model.md.

Usage:
    <venv>/Scripts/python experiments/affect_prototype/run.py
Writes experiments/affect_prototype/results/<MODEL_VERSION>/{timeline.csv, pad_timeline.png}.
Outputs are written per model version (results/v0.1 is committed).
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from nerva.affect import CategoricalAffectModel  # noqa: E402
from nerva.appraisal import appraise  # noqa: E402
from nerva.interfaces import Event  # noqa: E402

MODEL_VERSION = "v0.1"  # affect model version this demo documents (see docs/affect_model.md)
DT = 0.05  # s
DURATION = 70.0  # s
TIMELINE = [  # (time s, event)
    (2.0, "successful_walking"),
    (8.0, "successful_walking"),
    (15.0, "person_approaching_slowly"),
    (25.0, "person_approaching_rapidly"),
    (27.0, "near_fall"),
    (40.0, "obstacle_blocking_goal"),
    (52.0, "successful_walking"),
]
LABELS = ["joy", "hope", "fear", "distress", "surprise"]


def simulate():
    model = CategoricalAffectModel()
    pending = sorted(TIMELINE)
    rows, fired = [], []
    for k in range(int(round(DURATION / DT)) + 1):
        t = k * DT
        while pending and pending[0][0] <= t + 1e-9:
            te, kind = pending.pop(0)
            appraisal = appraise(Event(kind))
            new = model.add(appraisal)
            # record intensities now: EmotionInstance objects decay in place afterwards
            fired.append((te, kind, appraisal, [(e.label, e.intensity) for e in new]))
        pad = model.step(DT)
        intensity = {lbl: sum(e.intensity for e in model.emotions if e.label == lbl) for lbl in LABELS}
        rows.append({"t": round(t + DT, 4), "valence": pad.valence, "arousal": pad.arousal,
                     "dominance": pad.dominance, **intensity})
    return rows, fired


def main() -> None:
    out = Path(__file__).resolve().parent / "results" / MODEL_VERSION
    out.mkdir(parents=True, exist_ok=True)
    rows, fired = simulate()

    with (out / "timeline.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"{'t':>5}  {'event':28s} {'appraisal (rel, des, lik, exp, ctl)':38s} emotions elicited")
    for te, kind, a, new in fired:
        emos = ", ".join(f"{label} {i:.2f}" for label, i in new) or "none"
        print(f"{te:5.1f}  {kind:28s} ({a.relevance:.1f}, {a.desirability:+.1f}, {a.likelihood:.1f}, "
              f"{a.expectedness:.1f}, {a.controllability:.1f})          {emos}")

    print(f"\n{'t':>5}  {'valence':>8} {'arousal':>8} {'dominance':>9}")
    for t in (1, 3, 10, 16, 26, 28, 31, 38, 42, 50, 54, 70):
        r = min(rows, key=lambda r: abs(r["t"] - t))
        print(f"{r['t']:5.1f}  {r['valence']:+8.3f} {r['arousal']:+8.3f} {r['dominance']:+9.3f}")

    t = [r["t"] for r in rows]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True, height_ratios=[1, 1.3])
    for lbl in LABELS:
        ax1.plot(t, [r[lbl] for r in rows], label=lbl, lw=1.6)
    ax1.set_ylabel("emotion intensity")
    ax1.legend(loc="upper right", ncol=5, fontsize=8, frameon=False)
    ax1.set_title(f"Affect model {MODEL_VERSION}: synthetic events → appraisal → emotions (top) → PAD (bottom)")
    for key, colour in (("valence", "#2a7ab9"), ("arousal", "#d1492e"), ("dominance", "#3b9a57")):
        ax2.plot(t, [r[key] for r in rows], label=key, lw=2, color=colour)
    ax2.axhline(0, color="0.6", lw=0.8)
    ax2.set_ylim(-0.6, 0.6)
    ax2.set_ylabel("PAD (−1 … +1)")
    ax2.set_xlabel("time (s)")
    ax2.legend(loc="upper right", ncol=3, fontsize=8, frameon=False)
    for te, kind, _, _ in fired:
        for ax in (ax1, ax2):
            ax.axvline(te, color="0.75", lw=0.8, ls="--")
        ax1.text(te, ax1.get_ylim()[1], kind.replace("_", " "), rotation=90, va="top", ha="right", fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "pad_timeline.png", dpi=130)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
