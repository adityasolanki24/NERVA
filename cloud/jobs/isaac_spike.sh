#!/bin/bash
# Isaac Sim feasibility spike on an L4 (experiments/isaac/isaac_spike.py). Requires the user's explicit
# acceptance of the NVIDIA Omniverse License Agreement: the launcher passes ISAAC_ACCEPT_EULA only then.
#
# Attempt 3 found the cause: the Deep Learning image's driver has no Vulkan at all on the host. The
# launcher now starts isaac_* jobs from plain Ubuntu 22.04 and this script installs nvidia-driver-570.
# Attempt 1 (2026-10-01) hung silently until the VM cap (output buffered by `tail`; root-owned mounts while
# the container runs as uid 1234). Attempt 2 started Isaac but rendering failed: "vkCreateInstance failed
# ... ERROR_INCOMPATIBLE_DRIVER" (no Vulkan inside the container). This version: streamed logs, owned
# mounts, NVIDIA_DRIVER_CAPABILITIES=all (graphics, not only compute), host Vulkan diagnostics, and image
# versions tried in order (the image's driver 580 may be newer than 5.0 supports). Each step has a timeout
# that leaves time to sync results before the VM's hard cap.
set -uo pipefail
: "${ISAAC_ACCEPT_EULA:?the user has not accepted the NVIDIA Omniverse License Agreement}"
IMAGES=${ISAAC_IMAGES:-"nvcr.io/nvidia/isaac-sim:5.1.0"}  # 5.1 renders with driver 570 (attempt 4)
OUT=/work/out/isaac
sync_isaac() { gsutil -q -m rsync -r "$OUT" "gs://$NERVA_BUCKET/runs/$NERVA_RUN_ID/out/isaac" || true; }
mkdir -p "$OUT" /work/isaac_cache/{kit,ov,glcache,computecache} /work/isaac_logs

# Full NVIDIA driver with the graphics (Vulkan/OpenGL) components: the VM is plain Ubuntu 22.04.
if ! command -v nvidia-smi >/dev/null || ! ls /usr/share/vulkan/icd.d/nvidia_icd.json >/dev/null 2>&1; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq && apt-get install -y -qq "linux-headers-$(uname -r)" nvidia-driver-570 >/dev/null
  modprobe nvidia || true
fi
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
git clone -q https://github.com/apirrone/Open_Duck_Playground.git /work/odp
UPSTREAM=$(curl -sf -H "Metadata-Flavor: Google" \
  http://metadata.google.internal/computeMetadata/v1/instance/attributes/nerva-upstream-sha)
git -C /work/odp checkout -q "$UPSTREAM"
MJCF_DIR=/work/odp/playground/open_duck_mini_v2/xmls
ROBOT=/work/odm/mini_bdx/robots/open_duck_mini_v2
sed 's#package:///##g' "$ROBOT/robot.urdf" > "$ROBOT/robot_local.urdf"

nvidia-smi > "$OUT/nvidia-smi.txt"
{
  echo "== host Vulkan ICDs"; ls -la /usr/share/vulkan/icd.d /etc/vulkan/icd.d 2>&1
  echo "== host NVIDIA graphics libraries"; ls /usr/lib/x86_64-linux-gnu | grep -iE "GLX_nvidia|EGL_nvidia|vulkan|glvk" 2>&1
  echo "== docker runtimes"; docker info 2>/dev/null | grep -iE "runtime|nvidia"
} > "$OUT/host_graphics.txt" 2>&1
chown -R 1234:1234 "$OUT" /work/isaac_cache /work/isaac_logs  # the container's user
sync_isaac

RUN=(docker run --rm --gpus all --network host
     -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=N -e NVIDIA_DRIVER_CAPABILITIES=all
     -v /work/NERVA/experiments/isaac:/nerva_isaac:ro -v "$ROBOT":/robot:ro -v "$MJCF_DIR":/mjcf:ro -v "$OUT":/out
     -v /work/isaac_cache/kit:/isaac-sim/kit/cache -v /work/isaac_cache/ov:/root/.cache/ov
     -v /work/isaac_cache/glcache:/root/.cache/nvidia/GLCache
     -v /work/isaac_cache/computecache:/root/.nv/ComputeCache
     -v /work/isaac_logs:/root/.nvidia-omniverse/logs
     --entrypoint /isaac-sim/python.sh)
STARTUP="from isaacsim import SimulationApp; a = SimulationApp({'headless': True}); print('ISAAC_STARTED', flush=True); a.close()"

IMAGE=""
for candidate in $IMAGES; do
  tag=$(basename "$candidate" | tr ':' '_')
  echo "== pull $candidate" >> "$OUT/pull.txt"
  ( time docker pull -q "$candidate" ) >> "$OUT/pull.txt" 2>&1 || continue
  log="$OUT/startup_$tag.log"
  echo "== minimal headless start with $candidate (15 min limit)" > "$log"
  timeout 15m "${RUN[@]}" "$candidate" -c "$STARTUP" >> "$log" 2>&1
  echo "startup exit=$?" >> "$log"
  sync_isaac
  if grep -q ISAAC_STARTED "$log" && ! grep -q "Failed to create any GPU devices" "$log"; then
    IMAGE=$candidate
    break
  fi
done
echo "selected image: ${IMAGE:-none}" | tee "$OUT/selected_image.txt"

if [ -n "$IMAGE" ]; then
  echo "== spike with $IMAGE (35 min limit)" > "$OUT/spike.log"
  timeout 35m "${RUN[@]}" "$IMAGE" /nerva_isaac/isaac_spike.py --urdf /robot/robot_local.urdf --mjcf /mjcf/open_duck_mini_v2.xml --out /out \
    --frames 120 --replay /nerva_isaac/replays/memory_s1.npz >> "$OUT/spike.log" 2>&1
  echo "spike exit=$?" >> "$OUT/spike.log"
fi
cp -r /work/isaac_logs "$OUT/kit_logs" 2>/dev/null || true
sync_isaac
ls -R "$OUT" | head -60
