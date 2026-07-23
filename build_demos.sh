#!/usr/bin/env bash
# Local helper: install build deps (if needed) and package Pong + Tetris.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

if [[ -x "$ROOT/.venv/bin/python" ]]; then
  PY="$ROOT/.venv/bin/python"
  PIP="$ROOT/.venv/bin/pip"
else
  PY="${PYTHON:-python3}"
  PIP="$PY -m pip"
fi

echo "Using: $PY"
$PIP install -q -r requirements-build.txt
$PY build_demos.py "$@"
