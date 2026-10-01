"""Attribute Model B's PAD drive to appraisal features and to event vs in-view appraisals (diagnosis only).

Default scenario, vision, seed 0, profile v2 with the old Model B. Results: development log 2026-10-02.
Usage: python experiments/affect_models/diagnose_model_b.py --policy B2.onnx
"""
# ruff: noqa: E302, E305, E402
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reactive"))
import scenario
from nerva.affect import model_b as mb

_ap = argparse.ArgumentParser()
_ap.add_argument("--policy", required=True)
B2 = _ap.parse_args().policy
log = []  # (kind, source, features, appraisal, origin)
orig_add = mb.DimensionalAffectModel.add
current = {"kind": "?"}
def add(self, appraisal, source=""):
    from nerva.interfaces import AppraisalFrame
    a = appraisal.as_appraisal_state() if isinstance(appraisal, AppraisalFrame) else appraisal
    kind = appraisal.hypothesis.kind if isinstance(appraisal, AppraisalFrame) else "?"
    log.append((kind, source, mb.appraisal_features(a), a))
    return orig_add(self, appraisal, source)
mb.DimensionalAffectModel.add = add
# split event vs in-view: FrameAppraiser.observe frames come through scenario's observed loop; tag by monkeypatching observe
from nerva.affect import frames as fr
orig_obs = fr.FrameAppraiser.observe
inview_ids = set()
def obs(self, *a, **k):
    out = orig_obs(self, *a, **k)
    for _, f in out:
        inview_ids.add(id(f))
    return out
fr.FrameAppraiser.observe = obs
orig_add2 = mb.DimensionalAffectModel.add
def add2(self, appraisal, source=""):
    r = orig_add2(self, appraisal, source)
    log[-1] = log[-1] + ("in_view" if id(appraisal) in inview_ids else "event",)
    return r
mb.DimensionalAffectModel.add = add2

_, rows, _, fired = scenario.run(B2, seed=0, perception_mode="vision", affect_model="B")
W = np.array(mb.DEFAULT_B.w)
tot = {"in_view": np.zeros(5), "event": np.zeros(5)}
cnt = {"in_view": 0, "event": 0}
for kind, src, u, a, origin in log:
    tot[origin] += u; cnt[origin] += 1
print("appraisals added:", cnt)
names = mb.FEATURES
for origin in tot:
    print(f"{origin:8s} summed features:", dict(zip(names, tot[origin].round(2))))
    print(f"{origin:8s} summed drive W·u (V, A, D):", (W @ tot[origin]).round(2))
iv = [(a.relevance, a.desirability, a.likelihood, a.expectedness, a.controllability) for *_, a, o in log if o == "in_view"]
iv = np.array(iv)
print("in-view appraisal means (r, d, l, e, c):", iv.mean(0).round(2), " min e", iv[:, 3].min().round(2))
print("in-view per-feature mean u:", dict(zip(names, np.array([u for k, s, u, a, o in log if o == "in_view"]).mean(0).round(3))))
# steady state of a repeated impulse u every 2 s with trace tau 4 s and PAD leak tau
tau_z, period = 4.0, 2.0
gain = 1 / (1 - np.exp(-period / tau_z))
print(f"repeat gain of the trace (impulse every {period}s, tau {tau_z}s): {gain:.2f}x one impulse")
lam = 1 / np.array(mb.DEFAULT_B.tau_pad_s)
u = np.array([u for k, s, u, a, o in log if o == "in_view"]).mean(0)
print("predicted steady PAD from mean in-view input alone (V, A, D):", (W @ (u * gain) / lam).round(2))
