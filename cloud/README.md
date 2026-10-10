# Cloud training (Google Cloud)

Cloud project, bucket and account identifiers are deliberately not committed.
Before using the launcher, configure them in your shell. For PowerShell:

```powershell
$env:NERVA_GCP_PROJECT = "<your-project-id>"
# Optional overrides:
# $env:NERVA_GCP_REGION = "us-central1"
# $env:NERVA_GCP_ZONE = "us-central1-a"
# $env:NERVA_GCP_ZONES = "us-central1-a,us-central1-b,us-central1-c"
# $env:NERVA_GCP_BUCKET = "<your-results-bucket>"
# $env:NERVA_GCP_SERVICE_ACCOUNT = "nerva-runner"
# $env:NERVA_GCP_ROUTER = "nerva-router"
# $env:NERVA_GCP_NAT = "nerva-nat"
```

**Every job runs on its own VM,** which:
- runs **committed** code only (a `git archive` of HEAD is uploaded to Cloud Storage)
- builds the environment from `requirements-train.lock.txt`
- runs `jobs/<job>.sh` and syncs `/work/out` to the configured private results bucket every 10 min
- **deletes itself** when the job ends

**Cost protection:**
- **Hard cap:** `--max-run-duration` with `--instance-termination-action=DELETE`, enforced by Google even if the job hangs.
- **Self-deletion:** the VM deletes itself at the end of the job (`vm_startup.sh`).
- **Capacity fallback:** launches try the configured zone and then region zones a/b/c; `kill` discovers the selected zone automatically.
- **Private networking:** VMs have no public IP. Temporary Cloud NAT provides outbound package downloads and should be removed after the jobs finish.
- **Safe attached mode:** `--wait --cleanup-network` waits for self-deletion, fetches the result and then removes Cloud NAT/router.
- **Zero-resource teardown:** add `--teardown` to attached mode to also delete the dedicated results bucket and runner account after fetching, or run `teardown --yes` later.
- **Backstop:** the budget alert "nerva" ($150). It's an alert only; Google's spend caps don't cover Compute Engine.

## Commands (run from the NERVA repo root)

```bash
python cloud/launch.py setup            # one-time: bucket + VM service account (shows plan; add --yes)
python cloud/launch.py network-up --yes # temporary egress for private VMs (small hourly/data charge)
python cloud/launch.py launch --job smoke --max-minutes 60       # shows plan; add --yes to start
python cloud/launch.py launch --job throughput_benchmark --hw l4 --max-minutes 45 --wait --cleanup-network
python cloud/launch.py launch --job s1_smoke --input-dir experiments/cloud_runs/R1/r1/references --max-minutes 45
python cloud/launch.py launch --job neutral_gpu_pilot --hw l4 --max-minutes 45 --input-bundle cloud/inputs/neutral_gpu_pilot.txt --wait --cleanup-network
#   --input-bundle LIST: uploads only the listed repo-relative files as inputs.tar.gz + MANIFEST.sha256; the job verifies them
python cloud/launch.py status [RUN]     # running VMs; with RUN, tail of that run's log
python cloud/launch.py fetch RUN        # results → experiments/cloud_runs/RUN/ (git-ignored)
python cloud/launch.py kill RUN --yes   # delete a VM immediately
python cloud/launch.py network-down --yes # after all NERVA VMs are gone
python cloud/launch.py audit            # read-only inventory; should print "none" after cleanup
python cloud/launch.py teardown --yes   # destructive: remove dedicated bucket/account/network
```

Check nothing is left running:

```powershell
gcloud compute instances list --project "$env:NERVA_GCP_PROJECT"
```

## Jobs

| job | what | hardware | used cap |
|---|---|---|---|
| `smoke` | upstream PPO path, one short batch: checks GPU, JAX, training, ONNX, sync, self-delete (not a result) | L4 | 1 h |
| `throughput_benchmark` | steady-state steps/s for hardware selection (not a result) | L4 | 45 min |
| `r0_references` / `r1_references` | regenerate neutral / seven-style reference gaits with validation and repair | CPU | 1–3 h |
| `b0_baseline` | upstream baseline, unchanged, 300 M steps (backlash scene) | L4 | 3 h |
| `b1_neutral` / `b2_neutral` | NERVA env at neutral; B2 adds the feet-height cost (the policy that walks backward) | L4 | 3 h |
| `s1_smoke` | tiny seven-style compile/train check (not a result) | L4 | 1 h |
| `s1_pilot` … `s5_pilot` | seven-style policies: S1 base, S2 feet-height −30, S3 head commands applied, S4 S3 + backward emphasis, S5 feet-height −10 | L4 | 3 h |
| `s1_eval` / `s2_eval` / `s3_eval` | style evaluation of a trained policy (MuJoCo rollouts) | CPU | 1 h |
| `demo_s1` / `reactive_demo` | render demo videos offscreen | CPU | 1–1.5 h |
| `neutral_gpu_pilot` | capped neutral motor continuation (`docs/gpu_neutral_pilot.md`); needs `--input-bundle cloud/inputs/neutral_gpu_pilot.txt` | L4 | 45 min |
| `gait_averaged_tracking` | same trainer with gait-averaged tracking (`docs/gait_averaged_tracking_pilot.md`); needs `--input-bundle cloud/inputs/gait_averaged_tracking.txt` | L4 | 45 min |

## Environment

- **`requirements-train.txt`** pins the science-relevant packages to the local inference versions: mujoco 3.3.0, jax 0.5.3, playground 0.0.4, brax 0.14.2, flax 0.10.6, onnxruntime 1.21.0. It adds `jax[cuda12]`, tensorflow-cpu and tf2onnx for ONNX export, and tensorboardX. ONNX and protobuf are pinned to patched versions because the generated lockfile is public and GitHub audits it.
- **`requirements-train.lock.txt`** is compiled by `uv pip compile --python-platform x86_64-manylinux_2_28 --python-version 3.12`. NumPy 2.x requires tf2onnx 1.17 or newer (`np.cast` was removed). The export-only stack uses TensorFlow 2.20 so it can share patched ONNX 1.22, protobuf 5.29 and ml-dtypes 0.5.x.
- **Image:** Deep Learning VM `common-cu129-ubuntu-2204-nvidia-580` (CUDA 12.9, driver 580).
- **Verified 2026-09-29:** JAX 0.5.3 used the L4 successfully with the CUDA wheels and driver 580. The post-run security update to the export-only TensorFlow/tf2onnx/ONNX/protobuf stack passed a clean local Python 3.12 conversion test and should be reconfirmed by the next capped cloud smoke.
- **Benchmark interpretation:** compare `benchmark.json` → `training_steps_per_second`, which is measured between batch-rounded checkpoints and excludes bootstrap time. Use `wall_seconds` only to budget total VM lifetime.
