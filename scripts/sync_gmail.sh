#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JARVIS_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/python_env.sh"
jarvis_select_python "$JARVIS_ROOT"
cd "$JARVIS_ROOT"
exec env PYTHONPATH="$JARVIS_ROOT" "$PYTHON_BIN" scripts/sync_gmail.py "$@"
