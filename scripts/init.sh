#!/usr/bin/env bash
set -e
mkdir -p memory/raw data tests app/core app/providers app/importers
python3 -m pip install -q typer rich pyyaml pydantic python-dateutil httpx 2>/dev/null || true
echo "init done"
