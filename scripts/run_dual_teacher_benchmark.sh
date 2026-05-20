#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
NUM_GPUS="${NUM_GPUS:-8}"
BASE_PORT="${BASE_PORT:-29900}"
EPOCHS="${EPOCHS:-150}"
BACKBONE="${BACKBONE:-mit_b1}"
NUM_WORKERS="${NUM_WORKERS:-8}"
VAL_NUM_WORKERS="${VAL_NUM_WORKERS:-8}"
PIN_MEMORY="${PIN_MEMORY:-true}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-true}"
AMP="${AMP:-false}"
PREP_DATA="${PREP_DATA:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export DIST_BACKEND="${DIST_BACKEND:-gloo}"
export PYTHONPATH="$ROOT/dual_teacher:${PYTHONPATH:-}"

NVRTC_LIB="$(python3 - <<'PY'
import pathlib
import sysconfig

print(pathlib.Path(sysconfig.get_paths()["purelib"]) / "nvidia" / "cuda_nvrtc" / "lib")
PY
)"
if [[ -d "$NVRTC_LIB" ]]; then
  export LD_LIBRARY_PATH="$NVRTC_LIB:${LD_LIBRARY_PATH:-}"
fi

if [[ "$PREP_DATA" == "1" ]]; then
  python3 "$ROOT/tools/prepare_method_datasets.py"
fi

mkdir -p "$ROOT/run_logs/dual_teacher"
cd "$ROOT/dual_teacher"
port="$BASE_PORT"
for dataset in rescuenet floodnet; do
  for split in 12_5 25 50; do
    if [[ "$dataset" == "rescuenet" ]]; then
      train_file="tools/train-rescue.py"
    else
      train_file="tools/train-flood.py"
    fi
    work_dir="work_dirs/benchmark_${dataset}_${split}_${RUN_ID}"
    log_path="$ROOT/run_logs/dual_teacher/${dataset}_${split}_${RUN_ID}.log"
    echo "==> Dual-Teacher ${dataset} split ${split}; work dir: dual_teacher/${work_dir}"
    torchrun \
      --nproc_per_node="$NUM_GPUS" \
      --master_port="$port" \
      "$train_file" \
      --ddp \
      --dual_teacher \
      --backbone "$BACKBONE" \
      --split "$split" \
      --epochs "$EPOCHS" \
      --num-workers "$NUM_WORKERS" \
      --val-num-workers "$VAL_NUM_WORKERS" \
      --pin-memory "$PIN_MEMORY" \
      --prefetch-factor "$PREFETCH_FACTOR" \
      --persistent-workers "$PERSISTENT_WORKERS" \
      --amp "$AMP" \
      --port "$port" \
      --work-dir "$work_dir" 2>&1 | tee "$log_path"
    port=$((port + 1))
  done
done
