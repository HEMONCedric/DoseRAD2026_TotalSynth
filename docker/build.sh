#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
IMAGE_TAG=${DOSERAD_IMAGE_TAG:-doserad2026-proton:public}

docker build \
  --platform linux/amd64 \
  --file "$REPOSITORY_ROOT/docker/Dockerfile" \
  --tag "$IMAGE_TAG" \
  "$REPOSITORY_ROOT"
