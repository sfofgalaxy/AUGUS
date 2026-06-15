#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

export ARGUS_TOOL_DEVICE="${ARGUS_TOOL_DEVICE:-auto}"

python -B -m argus.tool_smoke "$@"

