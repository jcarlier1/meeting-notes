#!/usr/bin/env bash
set -euo pipefail

echo "[bootstrap] Installing project in editable mode..."
python3 -m pip install -U pip wheel setuptools
python3 -m pip install -e .
echo "[bootstrap] Done. Run 'meeting-notes --help' to see commands."

