#!/bin/bash
# Isaac Sim feasibility spike on an L4 (experiments/isaac/isaac_spike.py). Requires the user's explicit
# acceptance of the NVIDIA Omniverse License Agreement: the launcher passes ISAAC_ACCEPT_EULA only then.
set -euo pipefail
: "${ISAAC_ACCEPT_EULA:?the user has not accepted the NVIDIA Omniverse License Agreement}"
IMAGE=${ISAAC_IMAGE:-nvcr.io/nvidia/isaac-sim:5.0.0}
mkdir -p /work/out/isaac /work/isaac_cache
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
# robot model: Open Duck Mini v2 URDF + meshes (public), mesh paths made relative for the importer
git clone -q -b v2 --depth 1 https://github.com/apirrone/Open_Duck_Mini.git /work/odm
ROBOT=/work/odm/mini_bdx/robots/open_duck_mini_v2
sed 's#package:///##g' "$ROBOT/robot.urdf" > "$ROBOT/robot_local.urdf"
nvidia-smi | tee /work/out/isaac/nvidia-smi.txt
time docker pull -q "$IMAGE" | tee /work/out/isaac/pull.txt
docker run --rm --gpus all --network host \
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=N \
  -v /work/NERVA/experiments/isaac:/nerva_isaac:ro -v "$ROBOT":/robot:ro \
  -v /work/out/isaac:/out -v /work/isaac_cache:/root/.cache \
  --entrypoint /isaac-sim/python.sh "$IMAGE" /nerva_isaac/isaac_spike.py \
  --urdf /robot/robot_local.urdf --out /out --frames 120 --replay /nerva_isaac/replays/memory_s1.npz 2>&1 | tail -200 | tee /work/out/isaac/spike.log
ls -R /work/out/isaac | head -50
