#!/usr/bin/env bash
# Run a Python file in the fvdb_tile GPU environment.
# Usage: fvdb_tile/run_gpu.sh <python_file> [args...]
export CUDA_HOME=/usr/local/cuda-13.1
source ~/.venvs/fvdb_cutile/bin/activate
exec python "$@"
