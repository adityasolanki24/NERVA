# Neutral motor-target audit

Preregistered after source inspection but before numerical probes. This audit
checks current formulas and reference targets; it cannot establish why B2
learned a particular behaviour. No policy, reward, reference or gate changes.

## Fixed questions, probes and decision rules

1. **Zero-command reference:** inventory shipped grid and exact nearest keys
   for stop, forward/backward vx ±0.15, lateral vy ±0.10 and yaw ±0.60, other
   components zero. Sample each selected reference at the 27 actual training
   phases `i/nb_steps`, where nb_steps is the recorded period × FPS truncated
   as upstream does. Require the expected 40 dimensions and finite values;
   report velocity means/ranges and joint/contact slices. Also report missing
   exact zero lateral/yaw grid entries. A selected moving target is an issue
   for nonzero-command imitation, but not proof of a zero-command reward.
2. **Stationary reward dispatch:** call the actual B2 `_get_reward` methods on
   synthetic data via a sensor adapter (no dynamics). Use motion-norm commands
   vx = 0, 0.009, 0.010, 0.011, with other slots zero, plus zero motion with
   head yaw 0.4. Identical qpos, qvel, contacts and reference across cases:
   base `(0,0,0.15,1,0,0,0)`, 14 actuator positions/velocities 0.1,
   zero free-joint velocities, two contacts, zero actions/forces. Report every
   raw/scaled reward term (B2 height scale -30, other defaults), including exact
   boundary handling. Float32 results; tolerance 1e-6 for zero and equality.
   Imitation is disabled if all zero-motion cases yield zero; stand-still is
   active if its raw cost is positive at zero; head-only command must not
   activate imitation. No total-reward or behavioural causal ranking.
3. **Tracking tolerances:** actual functions, sigma 0.01, command stop and yaw
   ±0.60. Full Cartesian probe velocities: vx = 0, 0.013, 0.10;
   vy = 0, 0.05, 0.10, 0.11; wz = 0, 0.033, 0.60. Report raw and scaled
   tracking rewards. Confirm lateral tolerance if vy 0/0.05/0.10 yield equal
   reward at fixed vx; verify 0.11 lowers it. Also report isolated stop loss
   at vx 0.013 and yaw 0.033 versus exact rest. These are controlled formula
   sensitivities, not reconstructed rewards from B2 rollouts.
4. **Touchdown height:** actual helper and B2 dispatch at stop/forward 0.15,
   both foot peaks stance_z + lift, lift = 0, 0.02, 0.04 m;
   first-contact = (0,0), (1,0), (1,1). Confirm no-contact cost zero and target
   touchdown zero; otherwise under-lift costs positive independent of command.
   This must not be described as a reward for moving or a penalty for never
   stepping. B2 has no S6 air-time term enabled.
5. **Reference velocity provenance:** inspect fitter ordering, reward slices,
   world/local frame choices and generator quaternion handling. If extracting
   an angular-velocity helper is safe, test pure world-yaw rotation +0.60 rad/s
   over dt 0.02 at yaw 0 and 1 rad, supplying the generator's actual quaternion
   convention. Expected `(0,0,0.60)`, tolerance 1e-6. Otherwise explicitly mark
   unverified. Source evidence can establish a formula bug; it cannot prove
   the historical shipped pickle was produced by this exact source.

Complete all fixed probes once, preserve contradictions and negative results;
do not tune thresholds or rerun physics. Stop on malformed/nonfinite inputs or
source mismatch. Record preregistration/implementation revisions, source hashes,
reference hash and upstream revisions. Small JSON reports only; no personal
paths or identifiers, raw artifacts remain ignored. Run meaningful regression
tests, full suite including slow tests, Ruff, public-content checks and final
read-only cloud audit. No cloud run, spending, deletion, hardware or training.

Deliver a motor-only candidate **design** grounded in the audit: explicit rest
and movement semantics, valid reference velocity frames, stop/turn validation
and neutral-first reward priorities. Do not implement/train the candidate or
replace historical defaults in this phase. Existing B2 gate remains failed;
expressive training still waits for a separately preregistered motor baseline.
