#!/usr/bin/env bash
set -euo pipefail

uv run python pipeline.py prep
uv run python pipeline.py train --epochs 15 --bs 512 --lr 3e-4
