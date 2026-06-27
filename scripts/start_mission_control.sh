#!/usr/bin/env bash
# Launch the Harvy Mission Control bridge server.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

# shellcheck source=/dev/null
source .venv/bin/activate

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

exec python -m mission_control.server
