#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/jarvis_lifecycle_lib.sh"
jarvis_init
jarvis_validate_bind

if jarvis_listener_pid; then
  pid="$JARVIS_FOUND_PID"
  jarvis_assert_owned_listener "$pid"
  if ! jarvis_wait_healthy "$pid" 2; then
    jarvis_error "Jarvis already owns 127.0.0.1:8123 but /health is not healthy."
    exit 1
  fi
  jarvis_write_pidfile "$pid"
  echo "Jarvis is already running and healthy (PID $pid, 127.0.0.1:8123)."
  exit 0
else
  result=$?
  if [[ "$result" -eq 2 ]]; then exit 1; fi
fi

if [[ -n "${JARVIS_TOKEN:-}" ]]; then
  : # Inherit the existing token without displaying or copying it to disk.
fi

# Prefer the launchd agent when its plist is installed: it reads the token from
# the login Keychain, keeps secrets out of argv/history, and gives crash
# recovery. Fall back to a direct nohup start only when no agent is installed.
if [[ -f "$JARVIS_LAUNCHD_PLIST" ]]; then
  if jarvis_launchd_loaded; then
    jarvis_launchd_kickstart
  else
    jarvis_launchd_bootstrap
  fi
  for _ in $(seq 1 "${JARVIS_START_TIMEOUT:-25}"); do
    if jarvis_listener_pid && jarvis_assert_owned_listener "$JARVIS_FOUND_PID" >/dev/null 2>&1 && jarvis_healthy; then
      jarvis_write_pidfile "$JARVIS_FOUND_PID"
      echo "Jarvis started via launchd and is healthy (PID $JARVIS_FOUND_PID, 127.0.0.1:8123)."
      exit 0
    fi
    sleep 1
  done
  jarvis_error "launchd agent did not bring Jarvis to a healthy state on 127.0.0.1:8123."
  exit 1
fi

if [[ -z "${JARVIS_TOKEN:-}" ]]; then
  echo "Warning: JARVIS_TOKEN is unset; API requests will fail closed." >&2
fi

mkdir -p "$JARVIS_DATA_DIR"
if [[ -e "$JARVIS_PID_FILE" ]]; then
  stale_pid="$(cat "$JARVIS_PID_FILE" 2>/dev/null || true)"
  if [[ "$stale_pid" =~ ^[0-9]+$ ]] && \
     ps -p "$stale_pid" -o command= 2>/dev/null | grep -Fq "uvicorn app.server:app"; then
    jarvis_error "PID file refers to a Jarvis server that is not listening on its expected socket; refusing to replace it."
    exit 1
  fi
  rm -f "$JARVIS_PID_FILE"
fi

umask 077
nohup "$SCRIPT_DIR/run_server.sh" >>"$JARVIS_LOG_FILE" 2>&1 </dev/null &
pid=$!
jarvis_write_pidfile "$pid"

if jarvis_wait_healthy "$pid" "${JARVIS_START_TIMEOUT:-25}"; then
  echo "Jarvis started and is healthy (PID $pid, 127.0.0.1:8123)."
  exit 0
fi

jarvis_error "Jarvis did not become healthy on 127.0.0.1:8123."
if ! kill -0 "$pid" 2>/dev/null; then
  rm -f "$JARVIS_PID_FILE"
  echo "See $JARVIS_LOG_FILE for server startup details." >&2
else
  echo "The Jarvis process is still running; it was left untouched for inspection." >&2
fi
exit 1
