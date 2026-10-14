#!/bin/sh
# Pull and start the OHH-204 GHCR sim image. Template only.
# This repository does not execute this script.
#
# Copy deploy/cloud/ to /opt/ohho and write /etc/ohho/cloud.env (mode 0600)
# before boot. Refuses placeholder tokens. Does not echo the token.
#
# TODO Varun: the guest needs the NVIDIA driver and nvidia-container-toolkit
# before `docker run --gpus all` works. This script does not install them.
# The image default is llvmpipe; the variables below match
# omnibot-digital-twin/docker/docker-compose.gpu.yml so EGL can use the GPU.

set -eu

ENV_FILE=/etc/ohho/cloud.env
if [ -f "$ENV_FILE" ]; then
    set -a
    # shellcheck disable=SC1090
    . "$ENV_FILE"
    set +a
fi

IMAGE="${OHHO_SIM_IMAGE:-ghcr.io/ohho-robotics/omnibot-sim:latest}"

case "${OHHO_GATEWAY_TOKEN:-}" in
    "" | REPLACE_ME | REPLACE_WITH_OHHO_GATEWAY_TOKEN)
        echo "OHHO_GATEWAY_TOKEN is not set. Refusing to start." >&2
        exit 1
        ;;
esac

case "$IMAGE" in
    *REPLACE* | "")
        echo "OHHO_SIM_IMAGE is a placeholder. Refusing to pull." >&2
        exit 1
        ;;
esac

echo "pulling ${IMAGE}"
docker pull "$IMAGE"
docker rm -f ohho-sim >/dev/null 2>&1 || true
# -e OHHO_GATEWAY_TOKEN passes the variable through without printing it.
docker run -d --name ohho-sim --gpus all \
    -p 127.0.0.1:9090:9090 \
    -p 127.0.0.1:8080:8080 \
    -p 127.0.0.1:8765:8765 \
    -e ROS_DOMAIN_ID=30 \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute \
    -e LIBGL_ALWAYS_SOFTWARE=0 \
    -e OHHO_GATEWAY_TOKEN \
    "$IMAGE" \
    ros2 launch omnibot_bringup sim_vr.launch.py \
    world:=apartment video:=true rosbridge_address:=0.0.0.0

mkdir -p /var/lib/ohho
touch /var/lib/ohho/heartbeat
echo "sim container started; gateway token was not printed"
