#!/usr/bin/env bash
# Run the PE image the way the evaluation server does.
# Usage: ./scripts/local_test.sh <image> <input_dir> <output_dir>
set -euo pipefail
IMAGE="${1:?Usage: $0 <image> <input_dir> <output_dir>}"
INPUT_DIR="${2:?missing input_dir}"
OUTPUT_DIR="${3:?missing output_dir}"
[ -d "$INPUT_DIR" ] || { echo "ERROR: input_dir does not exist: $INPUT_DIR" >&2; exit 2; }
mkdir -p "$OUTPUT_DIR"

# Sealed network, capped memory, read-only input, 10-minute budget for all sequences.
timeout 660 docker run --rm \
  --gpus all \
  --network=none \
  --memory=20g \
  --pids-limit=512 \
  -v "$(realpath "$INPUT_DIR")":/input:ro \
  -v "$(realpath "$OUTPUT_DIR")":/output \
  "$IMAGE" "${@:4}"

echo; echo "Outputs written to: $OUTPUT_DIR"; ls -R "$OUTPUT_DIR"
