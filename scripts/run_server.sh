#!/usr/bin/env bash
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Foreground server entrypoint used by start_jarvis.sh. Keep the local service
# pinned to its established loopback address and port.
if [[ "${JARVIS_HOST:-127.0.0.1}" != "127.0.0.1" ]]; then
  echo "Jarvis must bind to 127.0.0.1; refusing JARVIS_HOST override." >&2
  exit 2
fi
if [[ "${JARVIS_PORT:-8123}" != "8123" ]]; then
  echo "Jarvis lifecycle uses port 8123; refusing JARVIS_PORT override." >&2
  exit 2
fi

if [ -z "${JARVIS_TOKEN:-}" ]; then
  echo "warning: JARVIS_TOKEN is not set; all /api/* requests will be refused." >&2
fi

source "$SCRIPT_DIR/python_env.sh"
jarvis_select_python "$PROJECT_ROOT"
exec env PYTHONPATH="$PROJECT_ROOT" "$PYTHON_BIN" -m uvicorn app.server:app --host 127.0.0.1 --port 8123
