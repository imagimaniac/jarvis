#!/usr/bin/env bash
set -euo pipefail

JARVIS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$JARVIS_ROOT/scripts/python_env.sh"
jarvis_select_python "$JARVIS_ROOT"
cd "$JARVIS_ROOT"
exec "$PYTHON_BIN" "$JARVIS_ROOT/scripts/seed_canonical_memories.py" --apply "$@"
