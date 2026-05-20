#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 {12_5|25|50} [extra run_benchmark_sweep.py args...]" >&2
  exit 2
fi

SPLIT="$1"
shift

SHOW_HELP=0
for arg in "$@"; do
  case "$arg" in
    -h|--help) SHOW_HELP=1 ;;
  esac
done

case "$SPLIT" in
  12_5|25|50) ;;
  *)
    echo "Unsupported split: $SPLIT. Expected one of: 12_5, 25, 50." >&2
    exit 2
    ;;
esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV_ROOT="${CONDA_ENV_ROOT:-${ENV_ROOT:-$HOME/.conda/envs}}"
RUNNER_PYTHON="${RUNNER_PYTHON:-$CONDA_ENV_ROOT/unimatch/bin/python}"
RUN_ID="${RUN_ID:-benchmark_${SPLIT}_$(date -u +%Y%m%d_%H%M%S)}"
EPOCHS="${EPOCHS:-150}"
NUM_GPUS="${NUM_GPUS:-4}"
DEVICES="${DEVICES:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}"
BASE_PORT="${BASE_PORT:-35000}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/benchmark_runs}"
if [[ "$OUTPUT_ROOT" != /* ]]; then
  OUTPUT_ROOT="$ROOT/$OUTPUT_ROOT"
fi
PREP_DATA="${PREP_DATA:-1}"
KEEP_CHECKPOINTS="${KEEP_CHECKPOINTS:-1}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-0}"
FORCE="${FORCE:-0}"
USE_TMUX="${USE_TMUX:-1}"

read -r -a METHOD_ARGS <<< "${METHODS:-s4mc classmix unimatch reco dual_teacher}"
read -r -a DATASET_ARGS <<< "${DATASETS:-floodnet rescuenet}"

declare -A METHOD_BINS=(
  [s4mc]="$CONDA_ENV_ROOT/s4mc/bin/torchrun"
  [classmix]="$CONDA_ENV_ROOT/classmix/bin/torchrun"
  [unimatch]="$CONDA_ENV_ROOT/unimatch/bin/python"
  [reco]="$CONDA_ENV_ROOT/reco/bin/accelerate"
  [dual_teacher]="$CONDA_ENV_ROOT/dual_teacher/bin/torchrun"
)

if [[ ! -x "$RUNNER_PYTHON" ]]; then
  echo "Missing runner Python: $RUNNER_PYTHON" >&2
  echo "Set RUNNER_PYTHON or CONDA_ENV_ROOT before launching this script." >&2
  exit 1
fi

for method in "${METHOD_ARGS[@]}"; do
  if [[ -z "${METHOD_BINS[$method]+set}" ]]; then
    echo "Unsupported method in METHODS: $method" >&2
    exit 2
  fi
  if [[ ! -x "${METHOD_BINS[$method]}" ]]; then
    echo "Missing executable for $method: ${METHOD_BINS[$method]}" >&2
    echo "Set CONDA_ENV_ROOT to the directory containing the method conda envs." >&2
    exit 1
  fi
done

export CONDA_ENV_ROOT

cmd=(
  "$RUNNER_PYTHON" "$ROOT/tools/run_benchmark_sweep.py"
  --epochs "$EPOCHS"
  --num-gpus "$NUM_GPUS"
  --devices "$DEVICES"
  --base-port "$BASE_PORT"
  --run-id "$RUN_ID"
  --output-root "$OUTPUT_ROOT"
  --methods "${METHOD_ARGS[@]}"
  --datasets "${DATASET_ARGS[@]}"
  --splits "$SPLIT"
)

if [[ "$PREP_DATA" == "1" ]]; then
  cmd+=(--prepare-data)
fi
if [[ "$KEEP_CHECKPOINTS" == "1" ]]; then
  cmd+=(--keep-checkpoints)
fi
if [[ "$CONTINUE_ON_ERROR" == "1" ]]; then
  cmd+=(--continue-on-error)
fi
if [[ "$FORCE" == "1" ]]; then
  cmd+=(--force)
fi

if [[ "$SHOW_HELP" == "0" && "$USE_TMUX" == "1" && "${NO_TMUX:-0}" != "1" && -z "${TMUX:-}" && "${_RATIO_BENCHMARK_IN_TMUX:-0}" != "1" ]]; then
  if ! command -v tmux >/dev/null 2>&1; then
    echo "tmux is not available. Install tmux or run with USE_TMUX=0." >&2
    exit 1
  fi

  INSTANCE_NAME="$(basename "$(dirname "$ROOT")")"
  SESSION_BASE="${TMUX_SESSION:-${INSTANCE_NAME}_${RUN_ID}}"
  SESSION_BASE="$(printf "%s" "$SESSION_BASE" | tr -c 'A-Za-z0-9_.-' '_')"
  TMUX_SESSION="$SESSION_BASE"
  suffix=2
  while tmux has-session -t "$TMUX_SESSION" 2>/dev/null; do
    TMUX_SESSION="${SESSION_BASE}_${suffix}"
    suffix=$((suffix + 1))
  done

  run_root="$OUTPUT_ROOT/$RUN_ID"
  tmux_log="$run_root/tmux.log"
  mkdir -p "$run_root"

  tmux_cmd=(
    env
    _RATIO_BENCHMARK_IN_TMUX=1
    CONDA_ENV_ROOT="$CONDA_ENV_ROOT"
    RUNNER_PYTHON="$RUNNER_PYTHON"
    RUN_ID="$RUN_ID"
    EPOCHS="$EPOCHS"
    NUM_GPUS="$NUM_GPUS"
    DEVICES="$DEVICES"
    BASE_PORT="$BASE_PORT"
    OUTPUT_ROOT="$OUTPUT_ROOT"
    PREP_DATA="$PREP_DATA"
    KEEP_CHECKPOINTS="$KEEP_CHECKPOINTS"
    CONTINUE_ON_ERROR="$CONTINUE_ON_ERROR"
    FORCE="$FORCE"
    METHODS="${METHOD_ARGS[*]}"
    DATASETS="${DATASET_ARGS[*]}"
  )
  if [[ -n "${DATASET_ROOT:-}" ]]; then
    tmux_cmd+=(DATASET_ROOT="$DATASET_ROOT")
  fi
  if [[ -n "${CLASSMIX_COCO_PRETRAIN:-}" ]]; then
    tmux_cmd+=(CLASSMIX_COCO_PRETRAIN="$CLASSMIX_COCO_PRETRAIN")
  fi
  tmux_cmd+=("$0" "$SPLIT" "$@")

  printf -v quoted_cmd "%q " "${tmux_cmd[@]}"
  quoted_cmd="${quoted_cmd% }"
  quoted_log="$(printf "%q" "$tmux_log")"
  tmux_shell="set -o pipefail; $quoted_cmd 2>&1 | tee -a $quoted_log"
  tmux new-session -d -s "$TMUX_SESSION" -c "$ROOT" "bash -lc $(printf "%q" "$tmux_shell")"

  echo "==> Started ratio benchmark in tmux"
  echo "    session=${TMUX_SESSION}"
  echo "    attach=tmux attach -t ${TMUX_SESSION}"
  echo "    log=${tmux_log}"
  echo "    output=${run_root}"
  if [[ "${TMUX_ATTACH:-0}" == "1" ]]; then
    exec tmux attach -t "$TMUX_SESSION"
  fi
  exit 0
fi

echo "==> Ratio benchmark split=${SPLIT}"
echo "    run_id=${RUN_ID}"
echo "    epochs=${EPOCHS}, num_gpus=${NUM_GPUS}, devices=${DEVICES}"
echo "    methods=${METHOD_ARGS[*]}"
echo "    datasets=${DATASET_ARGS[*]}"
echo "    conda_env_root=${CONDA_ENV_ROOT}"
echo "    runner_python=${RUNNER_PYTHON}"
echo "    output=${OUTPUT_ROOT}/${RUN_ID}"

cd "$ROOT"
exec "${cmd[@]}" "$@"
