#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 4 ]; then
  echo "Usage: $0 <proton-ct|proton-mri> <input-dir> <output-dir> <models-dir>" >&2
  exit 2
fi

TASK_NAME=$1
INPUT_DIR=$(realpath "$2")
OUTPUT_DIR=$(realpath "$3")
MODELS_DIR=$(realpath "$4")
IMAGE_TAG=${DOSERAD_IMAGE_TAG:-doserad2026-proton:public}

case "$TASK_NAME" in
  proton-ct|proton-mri) ;;
  *) echo "TASK must be proton-ct or proton-mri" >&2; exit 2 ;;
esac

docker run --rm \
  --gpus all \
  --shm-size 2g \
  --platform linux/amd64 \
  --env TASK="$TASK_NAME" \
  --env HF_HUB_OFFLINE=1 \
  --publish 4743:4743 \
  --volume "$INPUT_DIR:/input:ro" \
  --volume "$OUTPUT_DIR:/output" \
  --volume "$MODELS_DIR:/opt/ml/model:ro" \
  "$IMAGE_TAG"
