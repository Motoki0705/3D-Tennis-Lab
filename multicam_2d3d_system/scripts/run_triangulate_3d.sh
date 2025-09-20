#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON:-python3}

"${PYTHON_BIN}" -m multicam_2d3d_system.src.main pipeline=triangulate3d "$@"
