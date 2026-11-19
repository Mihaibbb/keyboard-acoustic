#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== [1/4] Install ==="
python -m pip install -q -r requirements.txt

echo "=== [2/4] Pytest ==="
python -m pytest tests/ -v

echo "=== [3/4] Demo end-to-end ==="
python scripts/demo.py

echo "=== [4/4] Confirmare fișiere rapoarte ==="
ls -la reports/
echo "OK – proiectul este verde."
