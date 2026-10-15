#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/jarvis_lifecycle_lib.sh"
jarvis_init
jarvis_validate_bind

# When launchd supervises Jarvis, unload the agent first: KeepAlive could
# otherwise revive the server right after the SIGTERM, so unloading the agent
# is what makes an intentional stop stick. RunAtLoad starts it again at the next
# login (or an explicit `jarvis start`).
if jarvis_launchd_loaded; then
  lpid="$(jarvis_launchd_pid || true)"
  if [[ -n "$lpid" ]] && jarvis_listener_pid && [[ "$JARVIS_FOUND_PID" == "$lpid" ]]; then
    jarvis_assert_owned_listener "$lpid"
  fi
  jarvis_launchd_bootout
  for _ in $(seq 1 20); do
    if ! jarvis_listener_pid; then
      rm -f "$JARVIS_PID_FILE"
      echo "Jarvis stopped (launchd agent unloaded; it restarts at next login)."
      exit 0
    fi
    sleep 1
  done
  jarvis_error "launchd agent unloaded but a listener still holds 127.0.0.1:8123."
  exit 1
fi

# Not launchd-managed: stop a server that was started directly (e.g. by hand).
if jarvis_listener_pid; then
  pid="$JARVIS_FOUND_PID"
else
  result=$?
  if [[ "$result" -eq 2 ]]; then exit 1; fi
  echo "Jarvis is stopped (no listener on 127.0.0.1:8123)."
  exit 0
fi
jarvis_assert_owned_listener "$pid"

if ! jarvis_pidfile_matches "$pid" && ! jarvis_healthy; then
  jarvis_error "Unmanaged listener did not identify as healthy Jarvis; refusing to stop PID $pid."
  exit 1
fi

kill -TERM "$pid"
for _ in $(seq 1 20); do
  if ! kill -0 "$pid" 2>/dev/null || \
     ! lsof -nP -a -p "$pid" -iTCP:8123 -sTCP:LISTEN >/dev/null 2>&1; then
    if jarvis_pidfile_matches "$pid"; then rm -f "$JARVIS_PID_FILE"; fi
    echo "Jarvis stopped (PID $pid)."
    exit 0
  fi
  sleep 1
done

jarvis_error "Jarvis PID $pid did not stop within 20 seconds; it was not force-killed."
exit 1
