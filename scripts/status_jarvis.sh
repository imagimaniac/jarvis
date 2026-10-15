#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/jarvis_lifecycle_lib.sh"
jarvis_init
jarvis_validate_bind

# --quiet makes the exit code the only signal (0 = running and healthy,
# 1 = stopped or unhealthy), for scripting such as the `jarvis` command.
quiet=0
if [[ "${1:-}" == "--quiet" ]]; then quiet=1; fi

if jarvis_listener_pid; then
  pid="$JARVIS_FOUND_PID"
else
  result=$?
  if [[ "$result" -eq 2 ]]; then exit 1; fi
  if [[ "$quiet" -eq 1 ]]; then exit 1; fi
  echo "Jarvis is stopped (127.0.0.1:8123 is not listening)."
  exit 0
fi
if ! jarvis_assert_owned_listener "$pid"; then exit 1; fi
if ! jarvis_healthy; then
  if [[ "$quiet" -eq 1 ]]; then exit 1; fi
  jarvis_error "PID $pid owns 127.0.0.1:8123, but /health did not return healthy."
  exit 1
fi
if [[ "$quiet" -eq 1 ]]; then exit 0; fi
echo "Jarvis is running and healthy (PID $pid, 127.0.0.1:8123)."
