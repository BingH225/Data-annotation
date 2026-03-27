#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "[bootstrap_remote] PROJECT_ROOT=$PROJECT_ROOT"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .
mkdir -p output cache/media
if [[ ! -f config/runtime.yaml ]]; then
  cp config/runtime.template.yaml config/runtime.yaml
  echo "[bootstrap_remote] created config/runtime.yaml from template"
fi

echo "[bootstrap_remote] done"

