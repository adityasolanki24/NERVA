# Cloud GPU smoke report — 2026-09-29

## Outcome

The end-to-end cloud path works: a private L4 VM booted, installed the locked environment, exposed the GPU to JAX, compiled MJX/PPO, trained one batch, wrote Orbax checkpoints, exported ONNX, synced results, and deleted itself. The successful run used NERVA commit `a042c5d` and pinned Open Duck Playground commit `b9be205`.

This was a pipeline test, not a locomotion result. The final policy stays upright but has not learned to walk after one batch.

## Successful run

| item | measured result |
|---|---|
| machine | `g2-standard-8`, one NVIDIA L4 |
| JAX | 0.5.3, `CudaDevice(id=0)` |
| requested smoke workload | 200,000 environment steps, 2 evaluations |
| actual Brax workload | 327,680 steps (rounded to whole PPO batches) |
| job wall time | 14 min 15 s |
| VM lifetime including bootstrap/sync/delete | about 18 min |
| initial evaluation reward | 13.961 ± 8.708 |
| final evaluation reward | 17.348 ± 12.064 |
| exit | 0; VM self-deleted |
| output | initial and final Orbax checkpoints, two ONNX models, TensorBoard events, frozen package list and logs |

Both ONNX files load and execute locally as `[1,101] → [1,14]` models with finite outputs.

## Visual inspection

The final checkpoint was replayed deterministically for 10 s at a 0.10 m/s forward command:

| metric | result |
|---|---:|
| mean forward speed after 2 s | −0.0001 m/s |
| maximum tilt | 5.53° |
| minimum base height | 0.154 m |
| fall | no |

The MP4 and JSON are local, git-ignored run artifacts under `experiments/cloud_runs/smoke-20260929-002136/`. The policy's failure to walk is expected after one PPO batch and must not be presented as evidence about the proposed style conditioning.

## What the first attempt found

The original 2-million-step smoke used upstream's 15 evaluations. It produced a valid step-0 checkpoint, but the measured evaluation interval showed it would exceed its one-hour cap. It was deliberately stopped after about 17 minutes. The smoke wrapper now changes only the disposable smoke workload to 200,000 requested steps and two evaluations; full scientific jobs keep the upstream settings.

Other operational findings:

- the organisation policy forbids public VM IPs, so the runner now uses `--no-address` and temporary Cloud NAT;
- L4 capacity moved between zones during the session, so a future launcher should retry zones A/B/C automatically;
- tf2onnx 1.16 logged a caught NumPy 2 `np.cast` error even though its files were valid;
- the public lockfile's ONNX 1.17 and protobuf 3.20 pins triggered 10 Dependabot alerts.

The export-only dependencies were updated to TensorFlow CPU 2.20, tf2onnx 1.17, ONNX 1.22 and protobuf 5.29.6. A clean Python 3.12 environment successfully converted and checked a Keras model with that exact stack, with no `np.cast` error. The next cloud smoke should still confirm the complete upstream export before a scientific run.

## Throughput and implication for B0

The interval between initial and final checkpoints was about 6 min 14 s for 327,680 steps, roughly 876 environment steps/s. At that observed rate, 300 million steps would take about 95 hours before allowing for startup and evaluations. At the observed public on-demand price for this VM, compute alone would be roughly $81.

**Correction (2026-09-29, Claude review):** this rate cannot support a B0 projection.
- The run had **one** training chunk.
- Brax's own metrics in the downloaded TensorBoard file give `training/sps` = 1,154 over `training/walltime` = 284 s for that chunk.
- Brax's timer wraps the first jitted training call, so those 284 s **include compiling the training step**.
- The checkpoint interval (about 876 steps/s) additionally includes the final evaluation and ONNX export.
- **Steady-state L4 throughput is therefore unknown,** and the 95 h / $81 figure is an upper bound dominated by one-time costs, not an estimate.
- `cloud/smoke_train.py` now reports `steady_training_steps_per_second` from Brax's `training/walltime` between the last two evaluations, and the benchmark job uses 3 evaluations, so this is measured properly.

Therefore **do not launch B0 on this L4 configuration yet**. First run a short throughput comparison on a faster accelerator or reduce the baseline workload with a clearly justified pilot protocol. A full B0 must remain unchanged if it is used as the scientific reproduction control.

## Cost and shutdown

The two billable VM lifetimes totalled roughly 35 minutes. At the listed on-demand VM price, compute is expected to be about $0.50, plus short-lived disk, NAT gateway/IP and NAT data-processing charges. Cloud billing reports are delayed, so this is an estimate rather than an invoice.

After downloading both runs, the following were deleted and independently listed as zero: instances, disks, addresses, forwarding rules, routers/NAT, snapshots, buckets and the dedicated runner service account. There are no NERVA cloud resources left that can continue accruing compute, network or storage charges.
