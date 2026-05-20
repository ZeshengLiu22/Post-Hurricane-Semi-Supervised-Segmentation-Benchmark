# Commands For 12.5/25/50 Split Runs

Run this once from the project root after moving the repo or changing the source dataset path:

```bash
python3 tools/prepare_method_datasets.py
python3 tools/check_pretrained_weights.py
```

The setup script creates method-local `dataset/` folders with symlinks into `DATASET_ROOT`, or into a `Dataset/` directory next to this repository by default, plus split files/configs for `12_5`, `25`, and `50`.
Dual-Teacher is the exception in shape only: it gets split-specific folders under `dual_teacher/data/{dataset}_{split}` because its dataset loader scans folders instead of split txt files.
The setup script also copies pretrained weights from `PRETRAIN_SOURCE_ROOT` (default: `../Pretrain_Weights`) into each method's expected local folder.

Benchmark training-budget policy:

- Epochs: `150` for all methods.
- Batch: `4` in each method config/script.
- Input crop: `750x750` for all methods.
- LR, scheduler, optimizer, and weight decay stay method-native.
- ClassMix is iteration-based, so each generated config sets `num_iterations = floor(num_labeled / 4) * 150`.
- ClassMix, ReCo, and S4MC use mixed precision by default on CUDA. Dual-Teacher defaults to full precision to match the previously working runs; set `AMP=true` only for an explicit precision experiment.

## Recommended Sequential Launchers

Each script runs `2 datasets x 3 ratios` sequentially for one method. A `RUN_ID` timestamp is added to output paths, so a second script launch will not overwrite the first launch.

Activate the matching conda environment first:

```bash
conda activate unimatch
bash scripts/run_unimatch_benchmark.sh

conda activate classmix
bash scripts/run_classmix_benchmark.sh

conda activate reco
bash scripts/run_reco_benchmark.sh

conda activate s4mc
bash scripts/run_s4mc_benchmark.sh

conda activate dual_teacher
bash scripts/run_dual_teacher_benchmark.sh
```

Useful overrides:

```bash
RUN_ID=trial1 NUM_GPUS=4 CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_unimatch_benchmark.sh
PREP_DATA=0 RUN_ID=trial1 NUM_GPUS=4 BASE_PORT=29600 bash scripts/run_classmix_benchmark.sh
BASE_PORT=31000 RUN_ID=trial1 bash scripts/run_s4mc_benchmark.sh
AMP=true RUN_ID=trial1 bash scripts/run_dual_teacher_benchmark.sh
```

## Ratio Launchers

Each script runs all five methods sequentially for one ratio across both datasets. These scripts call the unified runner directly, so they can be launched from the project root without switching conda environments between methods.
Full benchmark outputs go under `benchmark_runs/{RUN_ID}` by default.

```bash
bash scripts/run_12_5.sh
bash scripts/run_25.sh
bash scripts/run_50.sh
```

Useful overrides:

```bash
RUN_ID=ratio12_5_full NUM_GPUS=4 DEVICES=0,1,2,3 bash scripts/run_12_5.sh
EPOCHS=150 BASE_PORT=36000 METHODS="s4mc classmix unimatch reco dual_teacher" bash scripts/run_25.sh
PREP_DATA=0 KEEP_CHECKPOINTS=1 DATASETS="floodnet rescuenet" bash scripts/run_50.sh
```

The ratio launchers use `tools/run_benchmark_sweep.py`, default to `EPOCHS=150`, and preflight the method conda env executables under `CONDA_ENV_ROOT` before training starts.

Output layout:

| Method | Main outputs |
| --- | --- |
| UniMatch | `UniMatch/exp/{dataset}/{method}/{exp}/{split}/{RUN_ID}/latest.pth`, `best.pth`, TensorBoard events, timestamp log |
| ClassMix | `ClassMix/checkpoints/{dataset}_{split}/{timestamp}-{dataset}_{split}_{RUN_ID}/checkpoint-*.pth`, `best_model.pth`, `config.json`, `train_split.pkl`, validation outputs |
| ReCo | `reco/outputs/reco/model_weights/*_{RUN_ID}.pth`, `reco/outputs/reco/logging/*_{RUN_ID}.npy`, terminal logs in `run_logs/reco/` |
| S4MC | `s4mc/checkpoints/{dataset}_{split}_{RUN_ID}/ckpt.pth`, `ckpt_best.pth`, TensorBoard events in `s4mc/log/events_seg/` |
| Dual-Teacher | `dual_teacher/work_dirs/benchmark_{dataset}_{split}_{RUN_ID}/best_weights.pth`, config dump, timestamp log |

The older manual commands below are still useful for one-off runs.

## UniMatch

Run from `UniMatch/`. Adjust GPU count and `CUDA_VISIBLE_DEVICES` as needed.

```bash
cd UniMatch
export CUDA_VISIBLE_DEVICES=0,1,2,3
port=29500
for dataset in rescuenet floodnet; do
  for split in 12_5 25 50; do
    DATASET="$dataset" SPLIT="$split" METHOD=unimatch EXP=r101 bash scripts/train.sh 4 "$port"
    port=$((port + 1))
  done
done
```

## ClassMix

Run from `ClassMix/`. Multi-GPU ClassMix now uses DDP, so launch with `torchrun`; do not use plain `python3 trainSSL.py --gpus 8`.

```bash
cd ClassMix
port=29600
for dataset in RescueNet FloodNet; do
  for split in 12_5 25 50; do
    name=$(echo "${dataset}_${split}" | tr 'A-Z' 'a-z')
    torchrun --nproc_per_node=8 --master_port="$port" trainSSL.py \
      --gpus 8 \
      --config "./configs/config${dataset}${split}.json" \
      --name "$name" \
      --amp true \
      --amp-dtype bf16
    port=$((port + 1))
  done
done
```

## ReCo

Run from `reco/`. The training path uses Accelerate BF16, anomaly detection is disabled, tensor augmentation avoids PIL CPU round-trips, and only the main process writes checkpoints/log arrays.

```bash
cd reco
port=29700
for dataset in rescuenet floodnet; do
  for split in 12_5 25 50; do
    accelerate launch --num_processes 8 --main_process_port "$port" train_semisup_acc.py \
      --dataset "$dataset" \
      --split "$split" \
      --apply_aug classmix \
      --apply_reco \
      --backbone deeplabv3p \
      --epochs 150
    port=$((port + 1))
  done
done
```

Evaluate a ReCo checkpoint:

```bash
cd reco
python eval.py --dataset rescuenet --split 25
python eval.py --dataset floodnet --split 25
```

## S4MC

Run from `s4mc/`. Adjust `--nproc_per_node` for your machine. BF16 AMP is enabled by default, and the unreliable-pixel threshold path uses `torch.quantile`.

```bash
cd s4mc
for dataset in rescuenet floodnet; do
  for split in 12_5 25 50; do
    torchrun --nproc_per_node=8 train_semi.py \
      --config "config_${dataset}_${split}.yaml" \
      --seed 42 \
      --name "${dataset}_${split}" \
      --amp true
  done
done
```

Evaluate an S4MC checkpoint:

```bash
cd s4mc
python eval-rescuenet.py --config config_rescuenet_25.yaml --ckpt checkpoints/rescuenet_25/ckpt_best.pth --save_dir results/eval_rescuenet_25
python eval-floodnet.py --config config_floodnet_25.yaml --ckpt checkpoints/floodnet_25/ckpt_best.pth --save_dir results/eval_floodnet_25
```

## Dual-Teacher

Run from `dual_teacher/`. The training scripts now accept `--split`; default work dirs append `split12_5`, `split25`, or `split50`. Full precision is the default to match the previously working runs, and `torch.backends.cudnn.benchmark` stays `False`. The Dual-Teacher train scripts add the local project root to `sys.path` before importing `mmseg`, so manual `PYTHONPATH` setup is not required.

```bash
cd dual_teacher
port=29600
for split in 12_5 25 50; do
  torchrun --nproc_per_node=8 --master_port="$port" tools/train-flood.py \
    --ddp \
    --dual_teacher \
    --backbone mit_b1 \
    --split "$split" \
    --epochs 150 \
    --amp false \
    --port "$port"
  port=$((port + 1))
done

for split in 12_5 25 50; do
  torchrun --nproc_per_node=8 --master_port="$port" tools/train-rescue.py \
    --ddp \
    --dual_teacher \
    --backbone mit_b1 \
    --split "$split" \
    --epochs 150 \
    --amp false \
    --port "$port"
  port=$((port + 1))
done
```
