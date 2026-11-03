#!/usr/bin/env bash
set -euo pipefail

uv run python pipeline.py prep
uv run python pipeline.py train --epochs 30 --bs 512
