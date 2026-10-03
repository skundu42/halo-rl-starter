#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<'HELP'
Usage: ./run.sh [train [--smoke] [--config configs/default.toml] | doctor | verify | demo]
Builds the pinned Halo demo image if missing, then runs it on one NVIDIA GPU.

Environment:
  GPU_DEVICE=0                 Host GPU index or GPU UUID
  HALO_ARCH=blackwell           blackwell (also Ampere/Ada SDPA) or hopper
  HALO_IMAGE=...                Optional upstream base image override
  DEMO_IMAGE=halo-arithmetic:ARCH  Optional local image name
  REBUILD=1                    Rebuild after editing this repository
  CACHE_DIR=$PWD/.cache         Persistent Hugging Face cache
HELP
    exit 0
fi

if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
    echo 'This GPU launcher requires Linux x86_64. For a CPU environment demo: python3 -m halo_demo demo' >&2
    exit 1
fi
command -v docker >/dev/null || { echo 'Install Docker and NVIDIA Container Toolkit first.' >&2; exit 1; }
docker info >/dev/null
halo_arch="${HALO_ARCH:-blackwell}"
case "$halo_arch" in
    blackwell|hopper) ;;
    *) echo 'HALO_ARCH must be blackwell or hopper.' >&2; exit 1 ;;
esac
demo_image="${DEMO_IMAGE:-halo-arithmetic:$halo_arch}"
halo_image="${HALO_IMAGE:-public.ecr.aws/whitecircle/halo:$halo_arch-1.0.0}"
if [[ "${REBUILD:-0}" == 1 ]] || ! docker image inspect "$demo_image" >/dev/null 2>&1; then
    docker build --build-arg "HALO_IMAGE=$halo_image" -t "$demo_image" .
fi

cache_dir="${CACHE_DIR:-$PWD/.cache}"
mkdir -p "$cache_dir" outputs
cache_dir="$(cd -- "$cache_dir" && pwd)"
if [[ $# == 0 ]]; then set -- train; fi
exec docker run --rm --init \
    --gpus "device=${GPU_DEVICE:-0}" \
    --user "$(id -u):$(id -g)" \
    --shm-size=8g --ulimit memlock=-1 --ulimit stack=67108864 \
    -e HOME=/tmp -e HF_HOME=/cache/huggingface \
    -e HALO_DATA_ROOT=/tmp/halo-data -e XDG_CACHE_HOME=/cache \
    -e HF_TOKEN \
    -v "$cache_dir:/cache" \
    -v "$PWD/outputs:/app/outputs" \
    -v "$PWD/configs:/app/configs:ro" \
    "$demo_image" "$@"
