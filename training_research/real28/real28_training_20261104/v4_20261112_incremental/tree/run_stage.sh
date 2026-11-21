#!/usr/bin/env bash
# Launch one stage of run 20261112_v4 in a throwaway container of the existing
# phase2-demo-deps image (torch-cpu, sklearn 1.8.0). Raw data is mounted read-only;
# only the independent run directory is writable. No network, no GPU, CPU-capped.
# usage: run_stage.sh <name> <cpus> <workdir under work/real28> <script> [args...] [-- KEY=VAL ...]
set -euo pipefail
BASE=/home/dev/Zhang/private/real28_incremental_20261112
name=$1; cpus=$2; wd=$3; script=$4; shift 4
args=(); envs=()
while [ $# -gt 0 ] && [ "$1" != "--" ]; do args+=("$1"); shift; done
if [ $# -gt 0 ]; then shift; for kv in "$@"; do envs+=(-e "$kv"); done; fi
exec docker run -d --name "real28-v4-$name" --network none --user "$(id -u):$(id -g)" \
  --cpus "$cpus" --memory 28g -e HOME=/tmp -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1 \
  -e OMP_NUM_THREADS=4 -e MKL_NUM_THREADS=4 -e OPENBLAS_NUM_THREADS=4 "${envs[@]}" \
  -v "/home/dev/Zhang/private/Real-time Data:/work/Real-time Data:ro" \
  -v "$BASE/work/real28:/work/real28" -w "/work/real28/$wd" \
  --entrypoint python phase2-demo-deps:0f2ab59a533a3dd2 -B "$script" "${args[@]}"
