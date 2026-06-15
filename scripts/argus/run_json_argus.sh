#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

log() {
  printf '[ARGUS %s] %s\n' "$(date '+%H:%M:%S')" "$*"
}

INPUT="${INPUT:-inputs/argus/json}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/argus/json}"
GPU="${GPU:-}"
LIMIT_ARGS=()

if [[ "${LIMIT:-0}" != "0" ]]; then
  LIMIT_ARGS+=(--limit "$LIMIT")
fi

mkdir -p "$OUTPUT_DIR"

if [[ -n "$GPU" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU"
fi
export ARGUS_TOOL_DEVICE="${ARGUS_TOOL_DEVICE:-auto}"

log "json ARGUS run"
log "input : $INPUT"
log "output: $OUTPUT_DIR"
log "limit : ${LIMIT:-0}"
log "gpu   : ${GPU:-unset}"
log "device: ARGUS_TOOL_DEVICE=${ARGUS_TOOL_DEVICE}"

python -u -m argus.json_runner \
  --input "$INPUT" \
  --output-dir "$OUTPUT_DIR" \
  ${LIMIT_ARGS+"${LIMIT_ARGS[@]}"}
