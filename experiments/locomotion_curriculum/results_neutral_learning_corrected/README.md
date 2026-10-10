# Balanced learned neutral candidate: negative overall pilot, visible motor result

Preregistered `8fd4fd3`; corrected training implementation `2de7528`.
Evaluation repair `4747b99`, video renderer `21c8a8d`. Protocol:
`docs/neutral_learning_pilot.md`. No criterion was changed after collection.

B2's retained 300,482,560-step walking weights initialize a motor-only 101/212
observation network, retaining 512/256/128 SiLU layers and frozen B2 preprocessing.
Constant-zero style inputs are folded into the first-layer biases. Against B2
ONNX, 100 admitted probes give max action error 7.45e-7. The final exported
candidate matches its checkpoint on 100 probes within 6.85e-7.

**Actual learning completed:** 28,672 new transitions, 128 accepted Adam updates,
all seven commands, 300.11 s under the 1,200 s hard cap. Fresh optimizer/RNG/
environment at seed 27, parameter warm-start only; states carry within the run.
Maximum post-update Gaussian KL 2.485e-5 (limit 0.05). Frozen statistics never
change; all 25 checkpoint leaves roundtrip in literal bytes. Training has 15
true episode terminations (one forward, fourteen left-turn) and 108 timeouts;
these are distinct from the deterministic native motor evaluation's fall metric.

**Motor evaluation completed:** 63 independent 20 s native MuJoCo trials, three
seeds, seven commands, three policy arms. Same backlash scene/noise, score 5–20 s.
No falls in any arm; no safety override. Evaluation takes 112.94 s on resume.
The preceding one-trial metadata-interface abort and its original rollout/log
remain retained; that completed B2 trial is reused rather than repeated.

Passing trials per command (a command requires 3/3):

| Command | Historical B2 | Untrained neutral clock | Learned candidate | Candidate mean requested-axis velocity |
|---|---:|---:|---:|---:|
| Rest | 0/3 | 3/3 | 3/3 | horizontal RMS 0.00132 m/s |
| Forward +0.074 m/s | 0/3 | 0/3 | 0/3 | +0.03206 m/s |
| Backward −0.074 m/s | 0/3 | 0/3 | 0/3 | −0.01648 m/s |
| Left +0.074 m/s | 1/3 | 1/3 | 1/3 | +0.03637 m/s |
| Right −0.074 m/s | 1/3 | 1/3 | 3/3 | −0.03845 m/s |
| Left turn +0.60 rad/s | 0/3 | 0/3 | 0/3 | +0.55808 rad/s |
| Right turn −0.60 rad/s | 0/3 | 0/3 | 0/3 | −0.59874 rad/s |

Forward/backward/left fail the half-requested-speed requirement; backward also
fails tracking RMSE. Both turn directions fail horizontal translation RMS:
candidate mean 0.03376/0.04428 m/s against <=0.03 m/s. Yaw tracking passes.
Rest displacement: historical B2 0.2565–0.2611 m, untrained neutral clock
0.0124–0.0202 m, learned candidate 0.0158–0.0225 m. The rest improvement exists
before SGD and is attributed to the clock contract, not learned weights.

Normalized moving-command axis RMSE: B2 0.430156, untrained 0.425003, candidate
0.413030. Candidate/control ratio 0.971827: **2.82% improvement, below the fixed
10% requirement**. Candidate/B2 improvement is 3.98%; both comparisons are
reported, with no checkpoint selection. Only rest and rightward motion pass all
three seeds. Overall pilot fails; original B2 gate and long motor gate remain
unmet. No new controller/default promotion or expressive training.

Seven 20 s side-by-side clips plus a combined 140 s video are under ignored
`experiments/cloud_runs/neutral-learning-pilot-corrected/`. Each shows historical
B2 and the final candidate, requested command, measured velocity and PASS/FAIL
label. `videos.json` and `combined_video.json` retain artifact hashes. Raw native
rollouts, initial/final/intermediate checkpoints, optimizer/batch/key snapshots
and logs are retained locally. They are not a full physical-environment resume
checkpoint. The earlier collector-interface abort and float-representation
integrity stop remain in the original and retry report directories.

The experiment initially exported with ONNX 1.17.0. The public optional exporter
pin is aligned to the repository's existing ONNX 1.22.0 pin; compatibility
verification is recorded separately. Existing evaluated artifact bytes remain
preserved; ONNX 1.22.0 produces a literally byte-identical final policy and zero
action difference on 100 probes. Full suite: 312 tests pass in 289.93 s, Ruff
passes; eight focused tests pass again under the patched exporter in 7.27 s.
No simulator, physical-safety, affect, upstream or cloud changes.

Next: preregister a longer capped neutral continuation targeting low-speed
tracking and pure-turn translation, with balanced commands and held-out motor
rollouts. A 300k-step local parameter continuation is a plausible next budget,
but this pilot does not demonstrate that more steps will solve those failures.
Declare a hard wall limit before running it, preserve this negative control,
and complete the long robustness gate before any expressive objectives.
