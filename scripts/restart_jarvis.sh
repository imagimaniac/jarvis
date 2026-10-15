#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/jarvis_lifecycle_lib.sh"
jarvis_init
jarvis_validate_bind

if jarvis_listener_pid; then
  pid="$JARVIS_FOUND_PID"
  jarvis_assert_owned_listener "$pid"
  if [[ -z "${JARVIS_TOKEN:-}" && ! -f "$JARVIS_LAUNCHD_PLIST" ]]; then
    jarvis_error "JARVIS_TOKEN is unavailable in this shell; leaving the running Jarvis untouched."
    exit 1
  fi
  "$SCRIPT_DIR/stop_jarvis.sh"
else
  result=$?
  if [[ "$result" -eq 2 ]]; then exit 1; fi
fi

"$SCRIPT_DIR/start_jarvis.sh"
"$SCRIPT_DIR/status_jarvis.sh"
