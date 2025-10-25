#!/usr/bin/env bash
set -euo pipefail

if command -v pytest >/dev/null 2>&1; then
  PYTEST=pytest
else
  PYTEST=python -m pytest
fi

exec $PYTEST "$@"
