#!/usr/bin/env bash
set -euo pipefail
cd "."
exec "/home/mark-agnew/Documents/Masters-phishD/.venv/bin/python3" -m src.evaluation.run_experiment "$@"
