#!/usr/bin/env bash
# Shared interpreter selection for tests, lifecycle checks, and the server.
# An explicit override wins, followed by a project-local venv, then the active
# environment's `python` command (for example, an activated Conda environment).
jarvis_select_python() {
  local project_root="$1" candidate=""

  if [[ -n "${PYTHON_BIN:-}" ]]; then
    candidate="$PYTHON_BIN"
  elif [[ -n "${JARVIS_PYTHON:-}" ]]; then
    candidate="$JARVIS_PYTHON"
  elif [[ -x "$project_root/.venv/bin/python" ]]; then
    candidate="$project_root/.venv/bin/python"
  elif command -v python >/dev/null 2>&1; then
    candidate="python"
  elif command -v python3 >/dev/null 2>&1; then
    candidate="python3"
  else
    echo "Jarvis error: no Python interpreter found; set JARVIS_PYTHON or activate the project environment." >&2
    return 1
  fi

  if ! command -v "$candidate" >/dev/null 2>&1; then
    echo "Jarvis error: configured Python interpreter is unavailable." >&2
    return 1
  fi
  PYTHON_BIN="$(command -v "$candidate")"
  export PYTHON_BIN
}
