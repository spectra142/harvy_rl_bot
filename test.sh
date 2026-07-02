#!/usr/bin/env bash
set -euo pipefail

# Use the virtual environment's Python if it exists.
if [ -d ".venv" ] && [ -f ".venv/bin/python" ]; then
  PYTHON=".venv/bin/python"
else
  PYTHON="python"
fi

echo "==> Python syntax checks"
$PYTHON -m py_compile src/train.py src/eval.py src/common/config.py
$PYTHON -m py_compile src/env/*.py
$PYTHON -m py_compile src/agent/*.py

echo "==> JavaScript syntax checks"
node --check bot/index.js
for f in bot/src/*.js; do
    node --check "$f"
done

echo "==> Python tests"
$PYTHON -m pytest tests/ -q

echo "==> All checks passed"
