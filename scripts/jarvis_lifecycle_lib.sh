#!/usr/bin/env bash
# Shared process checks for Jarvis lifecycle commands. Never act on a listener
# unless its PID, working directory, loopback socket, and Jarvis health agree.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/python_env.sh"

jarvis_init() {
  JARVIS_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
  JARVIS_DATA_DIR="$JARVIS_ROOT/data"
  JARVIS_PID_FILE="$JARVIS_DATA_DIR/jarvis-server.pid"
  JARVIS_LOG_FILE="$JARVIS_DATA_DIR/jarvis-server.log"
  JARVIS_URL="http://127.0.0.1:8123/health"
  jarvis_select_python "$JARVIS_ROOT"
}

jarvis_error() { echo "Jarvis error: $*" >&2; }

jarvis_validate_bind() {
  if [[ "${JARVIS_HOST:-127.0.0.1}" != "127.0.0.1" ]]; then
    jarvis_error "JARVIS_HOST must remain 127.0.0.1."
    return 1
  fi
  if [[ "${JARVIS_PORT:-8123}" != "8123" ]]; then
    jarvis_error "JARVIS_PORT must remain 8123."
    return 1
  fi
  for command_name in lsof curl "$PYTHON_BIN"; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
      jarvis_error "Required command not found: $command_name"
      return 1
    fi
  done
}

jarvis_listener_pids() {
  lsof -nP -t -iTCP:8123 -sTCP:LISTEN 2>/dev/null | sort -u
}

# Set JARVIS_FOUND_PID and return 0 for one listener, 1 for none, 2 for an
# ambiguous or uninspectable listener. lsof is used instead of broad process
# searches so unrelated Python/Uvicorn instances are never selected.
jarvis_listener_pid() {
  local pids count
  pids="$(jarvis_listener_pids)"
  if [[ -z "$pids" ]]; then
    JARVIS_FOUND_PID=""
    return 1
  fi
  count="$(printf '%s\n' "$pids" | wc -l | tr -d '[:space:]')"
  if [[ "$count" != "1" ]]; then
    jarvis_error "Multiple processes listen on port 8123; refusing to choose one."
    return 2
  fi
  JARVIS_FOUND_PID="$pids"
  return 0
}

jarvis_process_cwd() {
  lsof -a -p "$1" -d cwd -Fn 2>/dev/null | sed -n 's/^n//p' | head -n 1
}

jarvis_loopback_socket() {
  local details
  details="$(lsof -nP -a -p "$1" -iTCP:8123 -sTCP:LISTEN 2>/dev/null || true)"
  printf '%s\n' "$details" | grep -Fq '127.0.0.1:8123 (LISTEN)'
}

jarvis_owned_cwd() {
  [[ "$(jarvis_process_cwd "$1")" == "$JARVIS_ROOT" ]]
}

jarvis_healthy() {
  local body
  body="$(curl --fail --silent --show-error --max-time 2 "$JARVIS_URL" 2>/dev/null)" || return 1
  printf '%s' "$body" | "$PYTHON_BIN" -c \
    'import json,sys; d=json.load(sys.stdin); sys.exit(0 if d.get("ok") is True else 1)' \
    >/dev/null 2>&1
}

jarvis_assert_owned_listener() {
  local pid="$1"
  if ! jarvis_owned_cwd "$pid"; then
    jarvis_error "Port 8123 is owned by a process outside this Jarvis project; leaving it untouched."
    return 1
  fi
  if ! jarvis_loopback_socket "$pid"; then
    jarvis_error "PID $pid is not listening on exactly 127.0.0.1:8123; leaving it untouched."
    return 1
  fi
}

jarvis_write_pidfile() {
  mkdir -p "$JARVIS_DATA_DIR"
  umask 077
  printf '%s\n' "$1" > "$JARVIS_PID_FILE"
}

jarvis_wait_healthy() {
  local pid="$1" timeout="$2" start now
  start="$(date +%s)"
  while true; do
    if jarvis_listener_pid && [[ "$JARVIS_FOUND_PID" == "$pid" ]] && \
       jarvis_assert_owned_listener "$pid" >/dev/null 2>&1 && jarvis_healthy; then
      return 0
    fi
    now="$(date +%s)"
    if (( now - start >= timeout )); then return 1; fi
    sleep 1
  done
}

jarvis_pidfile_matches() {
  local recorded
  [[ -r "$JARVIS_PID_FILE" ]] || return 1
  recorded="$(cat "$JARVIS_PID_FILE" 2>/dev/null || true)"
  [[ "$recorded" =~ ^[0-9]+$ && "$recorded" == "$1" ]]
}

# launchd integration. Jarvis is normally supervised by the user LaunchAgent
# "${JARVIS_LAUNCHD_LABEL}"; these helpers let the lifecycle scripts treat an
# intentional stop as "unload the agent" so KeepAlive cannot revive it.
JARVIS_LAUNCHD_LABEL="${JARVIS_LAUNCHD_LABEL:-com.example.jarvis}"
JARVIS_LAUNCHD_DOMAIN="gui/$(id -u)"
JARVIS_LAUNCHD_PLIST="${JARVIS_LAUNCHD_PLIST:-$HOME/Library/LaunchAgents/${JARVIS_LAUNCHD_LABEL}.plist}"

jarvis_launchd_loaded() {
  command -v launchctl >/dev/null 2>&1 || return 1
  launchctl print "$JARVIS_LAUNCHD_DOMAIN/$JARVIS_LAUNCHD_LABEL" >/dev/null 2>&1
}

jarvis_launchd_pid() {
  launchctl print "$JARVIS_LAUNCHD_DOMAIN/$JARVIS_LAUNCHD_LABEL" 2>/dev/null \
    | sed -n 's/^[[:space:]]*pid = \([0-9][0-9]*\).*/\1/p' | head -n 1
}

jarvis_launchd_bootout() {
  launchctl bootout "$JARVIS_LAUNCHD_DOMAIN/$JARVIS_LAUNCHD_LABEL" >/dev/null 2>&1 || true
}

jarvis_launchd_bootstrap() {
  launchctl bootstrap "$JARVIS_LAUNCHD_DOMAIN" "$JARVIS_LAUNCHD_PLIST"
}

jarvis_launchd_kickstart() {
  launchctl kickstart "$JARVIS_LAUNCHD_DOMAIN/$JARVIS_LAUNCHD_LABEL"
}
