#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

source "$SCRIPT_DIR/python_env.sh"
jarvis_select_python "$PROJECT_ROOT"
if [[ ! -f "$PROJECT_ROOT/app/server.py" || ! -f "$PROJECT_ROOT/requirements.txt" ]]; then
  echo "Jarvis deploy failed: project files are missing." >&2
  exit 2
fi
for command_name in "$PYTHON_BIN" lsof curl; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "Jarvis deploy failed: required command not found: $command_name" >&2
    exit 2
  fi
done
if [[ "${JARVIS_HOST:-127.0.0.1}" != "127.0.0.1" || "${JARVIS_PORT:-8123}" != "8123" ]]; then
  echo "Jarvis deploy failed: lifecycle is fixed to 127.0.0.1:8123." >&2
  exit 2
fi

echo "Running the complete Jarvis test suite..."
if ! "$PYTHON_BIN" -m pytest -q; then
  echo "Jarvis deploy aborted: tests failed; the running server was not restarted." >&2
  exit 1
fi
echo "Tests passed."
if [[ -z "${JARVIS_TOKEN:-}" ]]; then
  echo "Jarvis deploy aborted: JARVIS_TOKEN is unavailable in this shell; the running server was not restarted." >&2
  exit 2
fi
echo "Restarting Jarvis..."
"$SCRIPT_DIR/restart_jarvis.sh"
"$SCRIPT_DIR/status_jarvis.sh"
echo "Jarvis deployed successfully. Health check passed at 127.0.0.1:8123."
