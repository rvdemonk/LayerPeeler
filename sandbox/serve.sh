#!/usr/bin/env bash
# Launch the appraisal sandbox dev preview. Stdlib python3 only — no venv,
# no deps, so it starts even when the hybrid venv is mid-rebuild.
# Usage: ./serve.sh [port]      default port 8642
set -euo pipefail
cd "$(dirname "$0")"
exec python3 serve.py "$@"
