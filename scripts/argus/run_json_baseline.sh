#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

log() {
  printf '[ARGUS %s] %s\n' "$(date '+%H:%M:%S')" "$*"
}

BASELINE="${BASELINE:-single_post}"
INPUT="${INPUT:-inputs/argus/json}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/baselines/${BASELINE}}"
GPU="${GPU:-}"
MODEL_PROVIDER="${MODEL_PROVIDER:-gpt}"
MODEL="${MODEL:-}"
EXTRA_ARGS=()

case "$BASELINE" in
  text_only)
    MODULE="baselines.text_only"
    EXTRA_ARGS+=(--model-provider "$MODEL_PROVIDER")
    if [[ -n "$MODEL" ]]; then
      EXTRA_ARGS+=(--model "$MODEL")
    fi
    ;;
  single_post)
    MODULE="baselines.single_post"
    EXTRA_ARGS+=(--model-provider "$MODEL_PROVIDER")
    if [[ -n "$MODEL" ]]; then
      EXTRA_ARGS+=(--model "$MODEL")
    fi
    if [[ -n "${MAX_IMAGES:-}" ]]; then
      EXTRA_ARGS+=(--max-images "$MAX_IMAGES")
    fi
    ;;
  single_agent)
    MODULE="baselines.single_agent"
    if [[ -n "$MODEL" ]]; then
      EXTRA_ARGS+=(--model "$MODEL")
    fi
    if [[ -n "${MAX_ITERATIONS:-}" ]]; then
      EXTRA_ARGS+=(--max-iterations "$MAX_ITERATIONS")
    fi
    if [[ -n "${MAX_IMAGES:-}" ]]; then
      EXTRA_ARGS+=(--max-images "$MAX_IMAGES")
    fi
    if [[ -n "${MAX_PAYLOAD_CHARS:-}" ]]; then
      EXTRA_ARGS+=(--max-payload-chars "$MAX_PAYLOAD_CHARS")
    fi
    ;;
  self_disclosure_detector)
    MODULE="baselines.self_disclosure_detector"
    EXTRA_ARGS+=(--model-provider "$MODEL_PROVIDER")
    if [[ -n "$MODEL" ]]; then
      EXTRA_ARGS+=(--model "$MODEL")
    fi
    if [[ -n "${MAX_IMAGES:-}" ]]; then
      EXTRA_ARGS+=(--max-images "$MAX_IMAGES")
    fi
    ;;
  holmeseye_adapted)
    MODULE="baselines.holmeseye_adapted"
    EXTRA_ARGS+=(--model-provider "$MODEL_PROVIDER")
    if [[ -n "$MODEL" ]]; then
      EXTRA_ARGS+=(--model "$MODEL")
    fi
    if [[ -n "${GROUP_SIZE:-}" ]]; then
      EXTRA_ARGS+=(--group-size "$GROUP_SIZE")
    fi
    if [[ -n "${MAX_IMAGES_PER_POST:-}" ]]; then
      EXTRA_ARGS+=(--max-images-per-post "$MAX_IMAGES_PER_POST")
    fi
    if [[ -n "${MAX_IMAGES_PER_USER:-}" ]]; then
      EXTRA_ARGS+=(--max-images-per-user "$MAX_IMAGES_PER_USER")
    fi
    if [[ -n "${MAX_INQUIRY_QUESTIONS:-}" ]]; then
      EXTRA_ARGS+=(--max-inquiry-questions "$MAX_INQUIRY_QUESTIONS")
    fi
    if [[ -n "${INQUIRY_MAX_IMAGES:-}" ]]; then
      EXTRA_ARGS+=(--inquiry-max-images "$INQUIRY_MAX_IMAGES")
    fi
    ;;
  *)
    echo "Unknown BASELINE=$BASELINE" >&2
    echo "Supported: text_only, single_post, single_agent, self_disclosure_detector, holmeseye_adapted" >&2
    exit 2
    ;;
esac

mkdir -p "$OUTPUT_DIR"

if [[ -n "$GPU" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU"
fi
export ARGUS_TOOL_DEVICE="${ARGUS_TOOL_DEVICE:-auto}"

log "json baseline run"
log "baseline: $BASELINE"
log "input   : $INPUT"
log "output  : $OUTPUT_DIR"
log "gpu     : ${GPU:-unset}"
log "device  : ARGUS_TOOL_DEVICE=${ARGUS_TOOL_DEVICE}"
log "model   : provider=$MODEL_PROVIDER name=${MODEL:-default}"
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  log "extra   : ${EXTRA_ARGS[*]}"
fi

python -u -m "$MODULE" \
  --input "$INPUT" \
  --output-dir "$OUTPUT_DIR" \
  "${EXTRA_ARGS[@]}"
