#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
NUM_GPUS="${NUM_GPUS:-4}"
BASE_PORT="${BASE_PORT:-29600}"
SAVE_IMAGES="${SAVE_IMAGES:-false}"
AMP="${AMP:-true}"
AMP_DTYPE="${AMP_DTYPE:-bf16}"
PREP_DATA="${PREP_DATA:-1}"

if [[ "$PREP_DATA" == "1" ]]; then
  python3 "$ROOT/tools/prepare_method_datasets.py"
fi

mkdir -p "$ROOT/run_logs/classmix"
cd "$ROOT/ClassMix"
port="$BASE_PORT"
for dataset in RescueNet FloodNet; do
  lower="$(echo "$dataset" | tr 'A-Z' 'a-z')"
  for split in 12_5 25 50; do
    name="${lower}_${split}_${RUN_ID}"
    log_path="$ROOT/run_logs/classmix/${name}.log"
    echo "==> ClassMix ${lower} split ${split}; run name: ${name}"
    torchrun --nproc_per_node="$NUM_GPUS" --master_port="$port" trainSSL.py \
      --gpus "$NUM_GPUS" \
      --config "./configs/config${dataset}${split}.json" \
      --name "$name" \
      --amp "$AMP" \
      --amp-dtype "$AMP_DTYPE" \
      --save-images "$SAVE_IMAGES" 2>&1 | tee "$log_path"
    port=$((port + 1))
  done
done
