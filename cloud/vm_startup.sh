#!/bin/bash
# NERVA cloud job runner — executed ONCE as root by GCE at first boot (metadata startup-script).
#
# Reads its job from instance metadata (set by cloud/launch.py), then:
#   1. waits for the NVIDIA driver (if a GPU is attached)
#   2. downloads the exact NERVA code snapshot (git archive of a commit) from GCS
#   3. clones Open_Duck_Playground at the pinned upstream commit
#   4. builds the Python env from cloud/requirements-train.lock.txt with uv
#   5. runs cloud/jobs/<job>.sh, which writes to /work/out
#   6. syncs /work/out to GCS every 10 min and at the end
#   7. deletes this VM (fallback: power off). The VM ALSO has --max-run-duration as a hard cap.
set -uo pipefail

MD=http://metadata.google.internal/computeMetadata/v1
meta() { curl -sf -H "Metadata-Flavor: Google" "$MD/instance/attributes/$1"; }

[ -f /var/nerva_started ] && exit 0   # never re-run on reboot
touch /var/nerva_started

JOB=$(meta nerva-job)
BUCKET=$(meta nerva-bucket)
RUN_ID=$(meta nerva-run-id)
CODE_SHA=$(meta nerva-code-sha)
UPSTREAM_SHA=$(meta nerva-upstream-sha)
INPUT_URI=$(meta nerva-input-uri || true)
NAME=$(curl -sf -H "Metadata-Flavor: Google" "$MD/instance/name")
ZONE=$(curl -sf -H "Metadata-Flavor: Google" "$MD/instance/zone" | awk -F/ '{print $NF}')
OUT_GS="gs://$BUCKET/runs/$RUN_ID"
export NERVA_BUCKET="$BUCKET"
export NERVA_RUN_ID="$RUN_ID"
export NERVA_INPUT_URI="$INPUT_URI"
if [ "$(meta nerva-isaac-eula || true)" = "Y" ]; then export ISAAC_ACCEPT_EULA=Y; fi

mkdir -p /work/out && cd /work
exec > >(tee -a /work/out/vm.log) 2>&1
echo "=== NERVA job $JOB run $RUN_ID on $NAME ($ZONE) code $CODE_SHA upstream $UPSTREAM_SHA ==="
date -u +"start %Y-%m-%dT%H:%M:%SZ"

sync_out() { gsutil -q -m rsync -r /work/out "$OUT_GS/out" || echo "WARN: sync failed"; }
finish() {
  code=$?
  echo "$code" > /work/out/EXIT_CODE
  date -u +"end %Y-%m-%dT%H:%M:%SZ exit=$code"
  sync_out
  echo "deleting VM $NAME"
  gcloud compute instances delete "$NAME" --zone "$ZONE" --quiet || shutdown -h now
}
trap finish EXIT

( while true; do sleep 600; sync_out; done ) &

# 1. GPU driver (Deep Learning VM images install it at first boot; wait up to 15 min)
if lspci | grep -qi nvidia; then
  for i in $(seq 1 90); do nvidia-smi >/dev/null 2>&1 && break; sleep 10; done
  nvidia-smi || { echo "ERROR: NVIDIA driver not available"; exit 10; }
fi

# 2. NERVA code
gsutil -q cp "gs://$BUCKET/code/$CODE_SHA.tar.gz" /work/nerva.tar.gz || exit 11
mkdir -p /work/NERVA && tar -xzf /work/nerva.tar.gz -C /work/NERVA

# 3. uv is needed by every job. R0 manages its generator-specific Python 3.10
# environment itself; avoid installing the multi-GB GPU stack on its CPU VM.
export HOME=/root
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
if [ "$JOB" != "r0_references" ] && [ "$JOB" != "r1_references" ] && [[ "$JOB" != isaac_* ]]; then
  git clone -q https://github.com/apirrone/Open_Duck_Playground.git /work/Open_Duck_Playground || exit 12
  git -C /work/Open_Duck_Playground checkout -q "$UPSTREAM_SHA" || exit 12
  uv venv --python 3.12 /work/venv || exit 13
  source /work/venv/bin/activate
  uv pip install -r /work/NERVA/cloud/requirements-train.lock.txt || exit 13
  uv pip install --no-deps -e /work/Open_Duck_Playground -e /work/NERVA || exit 13
  uv pip freeze > /work/out/pip-freeze.txt
  python -c "import jax; print('jax', jax.__version__, jax.devices())" | tee /work/out/jax-devices.txt
else
  uv --version | tee /work/out/uv-version.txt
fi

# 4. job
date -u +"job-start %Y-%m-%dT%H:%M:%SZ"
bash "/work/NERVA/cloud/jobs/$JOB.sh"
