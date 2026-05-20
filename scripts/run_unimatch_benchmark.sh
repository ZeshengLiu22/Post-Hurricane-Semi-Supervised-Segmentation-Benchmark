#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
NUM_GPUS="${NUM_GPUS:-4}"
BASE_PORT="${BASE_PORT:-29500}"
METHOD="${METHOD:-unimatch}"
EXP="${EXP:-r101}"
PREP_DATA="${PREP_DATA:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

if [[ "$PREP_DATA" == "1" ]]; then
  python3 "$ROOT/tools/prepare_method_datasets.py"
fi

cd "$ROOT/UniMatch"
port="$BASE_PORT"
for dataset in rescuenet floodnet; do
  for split in 12_5 25 50; do
    save_path="exp/${dataset}/${METHOD}/${EXP}/${split}/${RUN_ID}"
    echo "==> UniMatch ${dataset} split ${split}; outputs: UniMatch/${save_path}"
    DATASET="$dataset" \
    SPLIT="$split" \
    METHOD="$METHOD" \
    EXP="$EXP" \
    SAVE_PATH="$save_path" \
    NUM_GPUS="$NUM_GPUS" \
    PORT="$port" \
      bash scripts/train.sh "$NUM_GPUS" "$port"
    port=$((port + 1))
  done
done
