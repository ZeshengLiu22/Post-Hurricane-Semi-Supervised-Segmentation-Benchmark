#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
NUM_GPUS="${NUM_GPUS:-8}"
BASE_PORT="${BASE_PORT:-29700}"
EPOCHS="${EPOCHS:-150}"
SEED="${SEED:-0}"
NUM_LABELS="${NUM_LABELS:-15}"
NUM_WORKERS="${NUM_WORKERS:-4}"
VAL_NUM_WORKERS="${VAL_NUM_WORKERS:-4}"
PIN_MEMORY="${PIN_MEMORY:-false}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-true}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/reco}"
PREP_DATA="${PREP_DATA:-1}"

if [[ "$PREP_DATA" == "1" ]]; then
  python3 "$ROOT/tools/prepare_method_datasets.py"
fi

mkdir -p "$ROOT/run_logs/reco" "$ROOT/reco/$OUTPUT_ROOT"
cd "$ROOT/reco"
port="$BASE_PORT"
for dataset in rescuenet floodnet; do
  for split in 12_5 25 50; do
    log_path="$ROOT/run_logs/reco/${dataset}_${split}_${RUN_ID}.log"
    echo "==> ReCo ${dataset} split ${split}; output root: reco/${OUTPUT_ROOT}; run id: ${RUN_ID}"
    accelerate launch --num_processes "$NUM_GPUS" --main_process_port "$port" train_semisup_acc.py \
      --dataset "$dataset" \
      --split "$split" \
      --num_labels "$NUM_LABELS" \
      --apply_aug classmix \
      --apply_reco \
      --backbone deeplabv3p \
      --epochs "$EPOCHS" \
      --seed "$SEED" \
      --num-workers "$NUM_WORKERS" \
      --val-num-workers "$VAL_NUM_WORKERS" \
      --pin-memory "$PIN_MEMORY" \
      --prefetch-factor "$PREFETCH_FACTOR" \
      --persistent-workers "$PERSISTENT_WORKERS" \
      --output-root "$OUTPUT_ROOT" \
      --run-id "$RUN_ID" 2>&1 | tee "$log_path"
    port=$((port + 1))
  done
done
