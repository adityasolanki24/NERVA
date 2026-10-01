#!/bin/bash
# Isaac Sim feasibility spike on an L4 (experiments/isaac/isaac_spike.py). Requires the user's explicit
# acceptance of the NVIDIA Omniverse License Agreement: the launcher passes ISAAC_ACCEPT_EULA only then.
# Attempt 1 (2026-10-01) hung silently until the VM cap: output was buffered by `tail` and the mounted
# directories were root-owned while the container runs as uid 1234. Now: streamed logs, owned mounts,
# a minimal startup test first, and timeouts that leave time to sync results before the VM cap.
set -uo pipefail
: "${ISAAC_ACCEPT_EULA:?the user has not accepted the NVIDIA Omniverse License Agreement}"
IMAGE=${ISAAC_IMAGE:-nvcr.io/nvidia/isaac-sim:5.0.0}
OUT=/work/out/isaac
mkdir -p "$OUT" /work/isaac_cache/{kit,ov,pip,glcache,computecache} /work/isaac_logs
if ! command -v docker >/dev/null; then
  apt-get update -qq && apt-get install -y -qq docker.io >/dev/null
fi
if ! docker info 2>/dev/null | grep -qi nvidia; then
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-ct.gpg
  curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-ct.gpg] https://#' \
    > /etc/apt/sources.list.d/nvidia-container-toolkit.list
  apt-get update -qq && apt-get install -y -qq nvidia-container-toolkit >/dev/null
  nvidia-ctk runtime configure --runtime=docker && systemctl restart docker
fi
git clone -q -b v2 --depth 1 https://github.com/apirrone/Open_Duck_Mini.git /work/odm
ROBOT=/work/odm/mini_bdx/robots/open_duck_mini_v2
sed 's#package:///##g' "$ROBOT/robot.urdf" > "$ROBOT/robot_local.urdf"
nvidia-smi > "$OUT/nvidia-smi.txt"
( time docker pull -q "$IMAGE" ) > "$OUT/pull.txt" 2>&1
chown -R 1234:1234 "$OUT" /work/isaac_cache /work/isaac_logs   # the container's user

RUN=(docker run --rm --gpus all --network host -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=N
     -v /work/NERVA/experiments/isaac:/nerva_isaac:ro -v "$ROBOT":/robot:ro -v "$OUT":/out
     -v /work/isaac_cache/kit:/isaac-sim/kit/cache -v /work/isaac_cache/ov:/root/.cache/ov
     -v /work/isaac_cache/glcache:/root/.cache/nvidia/GLCache
     -v /work/isaac_cache/computecache:/root/.nv/ComputeCache -v /work/isaac_logs:/root/.nvidia-omniverse/logs
     --entrypoint /isaac-sim/python.sh "$IMAGE")

echo "== step 0: minimal headless start (20 min limit)" | tee "$OUT/startup.log"
timeout 20m "${RUN[@]}" -c "from isaacsim import SimulationApp; a = SimulationApp({'headless': True}); print('ISAAC_STARTED', flush=True); a.close()" \
  >> "$OUT/startup.log" 2>&1
echo "startup exit=$?" | tee -a "$OUT/startup.log"
gsutil -q -m rsync -r "$OUT" "gs://$NERVA_BUCKET/runs/$NERVA_RUN_ID/out/isaac" || true

if grep -q ISAAC_STARTED "$OUT/startup.log"; then
  echo "== step 1: spike (35 min limit)" | tee "$OUT/spike.log"
  timeout 35m "${RUN[@]}" /nerva_isaac/isaac_spike.py --urdf /robot/robot_local.urdf --out /out \
    --frames 120 --replay /nerva_isaac/replays/memory_s1.npz >> "$OUT/spike.log" 2>&1
  echo "spike exit=$?" | tee -a "$OUT/spike.log"
fi
cp -r /work/isaac_logs "$OUT/kit_logs" 2>/dev/null || true
ls -R "$OUT" | head -60
