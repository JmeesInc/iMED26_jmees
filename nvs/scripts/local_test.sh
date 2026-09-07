#!/usr/bin/env bash
# Run the NVS image the way the evaluation server does.
# Usage: ./scripts/local_test.sh <image> <input_dir> <output_dir>
#
# The container REQUIRES CUDA: the challenge evaluator rejects CPU-only inference
# ("performs CPU-only inference and does not use CUDA") regardless of wall-clock time.
set -euo pipefail
IMAGE="${1:?Usage: $0 <image> <input_dir> <output_dir>}"
INPUT_DIR="${2:?missing input_dir}"
OUTPUT_DIR="${3:?missing output_dir}"
[ -d "$INPUT_DIR" ] || { echo "ERROR: input_dir does not exist: $INPUT_DIR" >&2; exit 2; }
mkdir -p "$OUTPUT_DIR"

# Every NVS_* knob set in the calling shell is forwarded, so ablations
# (instrument exclusion x hole filler x tone curve) run on ONE image, no rebuild.
ENVS=()
while IFS= read -r kv; do ENVS+=(-e "$kv"); done < <(env | grep -E "^NVS_" || true)
[ ${#ENVS[@]} -gt 0 ] && echo "env: ${ENVS[*]}"

docker run --rm \
  --gpus all \
  --network=none \
  --memory=20g \
  ${ENVS[@]+"${ENVS[@]}"} \
  -v "$(realpath "$INPUT_DIR")":/input:ro \
  -v "$(realpath "$OUTPUT_DIR")":/output \
  "$IMAGE" "${@:4}"

echo; echo "Outputs written to: $OUTPUT_DIR"
find "$OUTPUT_DIR" -maxdepth 3 -type d
