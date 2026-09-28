# Cloud training (Google Cloud, project `nerva-adityapersonal`)

**Every job runs on its own VM,** which:
- runs **committed** code only (a `git archive` of HEAD is uploaded to Cloud Storage)
- builds the environment from `requirements-train.lock.txt`
- runs `jobs/<job>.sh` and syncs `/work/out` to `gs://nerva-adityapersonal-runs/runs/<run>/out` every 10 min
- **deletes itself** when the job ends

**Cost protection:**
- **Hard cap:** `--max-run-duration` with `--instance-termination-action=DELETE`, enforced by Google even if the job hangs.
- **Self-deletion:** the VM deletes itself at the end of the job (`vm_startup.sh`).
- **Backstop:** the budget alert "nerva" ($150). It's an alert only; Google's spend caps don't cover Compute Engine.

## Commands (run from the NERVA repo root)

```bash
python cloud/launch.py setup            # one-time: bucket + VM service account (shows plan; add --yes)
python cloud/launch.py launch --job smoke --max-hours 1          # shows plan; add --yes to start
python cloud/launch.py status [RUN]     # running VMs; with RUN, tail of that run's log
python cloud/launch.py fetch RUN        # results → experiments/cloud_runs/RUN/ (git-ignored)
python cloud/launch.py kill RUN --yes   # delete a VM immediately
```

Check nothing is left running:

```bash
gcloud compute instances list --project nerva-adityapersonal
```

## Jobs

| job | what | hardware | typical cap |
|---|---|---|---|
| `smoke` | upstream runner, 2 M steps: checks GPU, JAX, training, ONNX export, sync and self-delete | L4 | 1 h |
| `b0_baseline` | upstream baseline, unchanged, 300 M steps on `flat_terrain_backlash` (README's "current win") | L4 | to be set from smoke timing |
| `r0_references` | regenerate the neutral reference set with the upstream generator; time the generation | CPU or L4 | to be set |
| `session1` | R0 in the background plus B0 | L4 | to be set |

## Environment

- **`requirements-train.txt`** pins the science-relevant packages to the local inference versions: mujoco 3.3.0, jax 0.5.3, playground 0.0.4, brax 0.14.2, flax 0.10.6, onnxruntime 1.21.0. It adds `jax[cuda12]`, tensorflow-cpu and tf2onnx for ONNX export, and tensorboardX.
- **`requirements-train.lock.txt`** is compiled by `uv pip compile --python-platform x86_64-manylinux_2_28 --python-version 3.12`. It resolves NumPy to 2.0.2 and protobuf to 3.20.3, which TensorFlow 2.18 and tf2onnx need; locally NumPy is 2.5.3.
- **Image:** Deep Learning VM `common-cu129-ubuntu-2204-nvidia-580` (CUDA 12.9, driver 580).
- **Unverified until the first smoke run:** whether JAX 0.5.3 with the CUDA 12.9 wheels in the lockfile runs on the L4 with that driver.
