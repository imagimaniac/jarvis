#!/usr/bin/env bash
# launchd entrypoint for Jarvis.
#
# This is the process the user-level launchd agent runs. It:
#   1. refuses to start a second server when a healthy Jarvis is already up,
#   2. reads the API token from the login Keychain (never from the plist,
#      argv, logs, or shell history) and fails closed if it is unavailable,
#   3. preserves the established loopback-only bind and feature flags,
#   4. waits briefly (never indefinitely) for Ollama, then starts anyway,
#   5. execs the foreground server so launchd supervises it directly.
#
# The plist supplies JARVIS_PYTHON, JARVIS_WEB, JARVIS_MEMORY_CANDIDATES and a
# PATH containing the project interpreter; this script also defaults them so it
# behaves the same when run by hand for testing.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/jarvis_lifecycle_lib.sh"
jarvis_init
jarvis_validate_bind

# Bounded TCP readiness probe. Returns 0 as soon as the port accepts a
# connection, 1 if it never does within the timeout. Non-fatal by design.
jarvis_wait_for_tcp() {
  local host="$1" port="$2" timeout="$3" start now
  start="$(date +%s)"
  while true; do
    if (exec 3<>"/dev/tcp/$host/$port") 2>/dev/null; then
      return 0
    fi
    now="$(date +%s)"
    if (( now - start >= timeout )); then return 1; fi
    sleep 1
  done
}

# 1. Never create a duplicate: if an owned, healthy Jarvis already listens on
#    127.0.0.1:8123 (for example one started by hand), leave it untouched.
if jarvis_listener_pid; then
  pid="$JARVIS_FOUND_PID"
  if jarvis_assert_owned_listener "$pid" >/dev/null 2>&1 && jarvis_healthy; then
    echo "Jarvis launchd: an owned, healthy server is already running (PID $pid); nothing to do."
    exit 0
  fi
  jarvis_error "127.0.0.1:8123 is held by a listener that is not a healthy Jarvis; refusing to start."
  exit 1
fi

# 2. Retrieve the token from the login Keychain. Fail closed rather than serve
#    an unauthenticated API. The value never appears in argv or on stdout.
token="$(security find-generic-password -a "$USER" -s jarvis-token -w 2>/dev/null || true)"
if [[ -z "$token" ]]; then
  jarvis_error "could not read the 'jarvis-token' Keychain item; refusing to start an unauthenticated server."
  exit 1
fi
export JARVIS_TOKEN="$token"
unset token

# 3. Preserve the established feature settings and interpreter.
export JARVIS_WEB="${JARVIS_WEB:-1}"
export JARVIS_MEMORY_CANDIDATES="${JARVIS_MEMORY_CANDIDATES:-1}"
export JARVIS_PYTHON="${JARVIS_PYTHON:-$PYTHON_BIN}"

# 4. Wait briefly for Ollama so login-time ordering is smooth, but never block
#    startup: retrieval and /health work without the model; only chat needs it.
jarvis_wait_for_tcp 127.0.0.1 11434 "${JARVIS_OLLAMA_WAIT:-30}" || \
  echo "Jarvis launchd: Ollama not reachable yet; starting anyway (chat will fail until it is up)." >&2

# 5. Run the server in the foreground so launchd owns and supervises the PID.
#    exec keeps the PID, so the pidfile stays accurate for the lifecycle scripts.
jarvis_write_pidfile "$$"
exec "$SCRIPT_DIR/run_server.sh"
