#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
NUM_GPUS="${NUM_GPUS:-8}"
BASE_PORT="${BASE_PORT:-29800}"
SEED="${SEED:-42}"
AMP="${AMP:-true}"
PREP_DATA="${PREP_DATA:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

if [[ "$PREP_DATA" == "1" ]]; then
  python3 "$ROOT/tools/prepare_method_datasets.py"
fi

mkdir -p "$ROOT/run_logs/s4mc"
cd "$ROOT/s4mc"
port="$BASE_PORT"
for dataset in rescuenet floodnet; do
  for split in 12_5 25 50; do
    name="${dataset}_${split}_${RUN_ID}"
    log_path="$ROOT/run_logs/s4mc/${name}.log"
    echo "==> S4MC ${dataset} split ${split}; checkpoint dir: s4mc/checkpoints/${name}"
    torchrun --nproc_per_node="$NUM_GPUS" --master_port="$port" train_semi.py \
      --config "config_${dataset}_${split}.yaml" \
      --seed "$SEED" \
      --name "$name" \
      --amp "$AMP" \
      --port "$port" 2>&1 | tee "$log_path"
    port=$((port + 1))
  done
done
