# S4MC

This directory contains the adapted S4MC code used in the post-hurricane semi-supervised segmentation benchmark.

Use the root project README for the full benchmark workflow. This file only covers direct S4MC commands.

## Setup

From the repository root:

```bash
python3 tools/prepare_method_datasets.py
python3 tools/check_pretrained_weights.py
```

The setup creates `s4mc/dataset/{FloodNet,RescueNet}/` symlinks, generated split files, generated `config_{dataset}_{split}.yaml` files, and expects `s4mc/resnet101.pth`.

Activate the S4MC environment before running:

```bash
conda activate s4mc
```

## Run All Benchmark Splits

From the repository root:

```bash
bash scripts/run_s4mc_benchmark.sh
```

Useful overrides:

```bash
RUN_ID=trial1 NUM_GPUS=4 CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_s4mc_benchmark.sh
PREP_DATA=0 AMP=true BASE_PORT=29820 bash scripts/run_s4mc_benchmark.sh
SEED=7 RUN_ID=seed7 bash scripts/run_s4mc_benchmark.sh
```

## Direct Training

Run from `s4mc/`:

```bash
cd s4mc
torchrun --nproc_per_node=4 --master_port=29800 train_semi.py \
  --config config_rescuenet_25.yaml \
  --seed 42 \
  --name rescuenet_25_manual \
  --amp true \
  --port 29800
```

Use `config_floodnet_12_5.yaml`, `config_floodnet_25.yaml`, `config_floodnet_50.yaml`, `config_rescuenet_12_5.yaml`, `config_rescuenet_25.yaml`, or `config_rescuenet_50.yaml`.

## Direct Evaluation

```bash
cd s4mc
python eval-rescuenet.py \
  --config config_rescuenet_25.yaml \
  --ckpt checkpoints/rescuenet_25_manual/ckpt_best.pth \
  --save_dir results/eval_rescuenet_25

python eval-floodnet.py \
  --config config_floodnet_25.yaml \
  --ckpt checkpoints/floodnet_25_manual/ckpt_best.pth \
  --save_dir results/eval_floodnet_25
```

For comparable cross-method metrics, prefer the root `tools/evaluate_semi_supervised_final.py` evaluator.

## Upstream Reference

Original project: <https://github.com/s4mcontext/s4mc>
