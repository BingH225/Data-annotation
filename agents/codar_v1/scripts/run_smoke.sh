#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ $# -lt 2 ]]; then
  echo "Usage: scripts/run_smoke.sh <input_json> <scenario> [backend] [limit]"
  exit 1
fi

INPUT_JSON="$1"
SCENARIO="$2"
BACKEND="${3:-vllm}"
LIMIT="${4:-3}"

source .venv/bin/activate
python -m codar.cli smoke \
  --input "$INPUT_JSON" \
  --scenario "$SCENARIO" \
  --limit "$LIMIT" \
  --backend "$BACKEND" \
  --config config/runtime.yaml \
  --output-dir output/smoke

