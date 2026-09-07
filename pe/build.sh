#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
TAG="${1:-imed-pe-jmees:dev}"
docker build -t "$TAG" .
echo "Built $TAG"
docker images --format '{{.Repository}}:{{.Tag}} {{.ID}} {{.Size}}' | grep "${TAG%%:*}" || true
