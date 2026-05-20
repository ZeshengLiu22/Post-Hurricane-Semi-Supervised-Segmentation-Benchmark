# UniMatch

This directory contains the adapted UniMatch/FixMatch code used in the post-hurricane semi-supervised segmentation benchmark.

Use the root project README for the full benchmark workflow. This file only covers direct UniMatch commands.

## Setup

From the repository root:

```bash
python3 tools/prepare_method_datasets.py
python3 tools/check_pretrained_weights.py
```

The setup creates `UniMatch/dataset/` symlinks, `UniMatch/splits/{dataset}/{split}/` files, and expects `UniMatch/pretrained/resnet101.pth`.

Activate the UniMatch environment before running:

```bash
conda activate unimatch
```

## Run All Benchmark Splits

From the repository root:

```bash
bash scripts/run_unimatch_benchmark.sh
```

Useful overrides:

```bash
RUN_ID=trial1 NUM_GPUS=4 CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_unimatch_benchmark.sh
METHOD=fixmatch RUN_ID=fixmatch_trial bash scripts/run_unimatch_benchmark.sh
PREP_DATA=0 BASE_PORT=29520 bash scripts/run_unimatch_benchmark.sh
```

## Direct Training

Run from `UniMatch/`:

```bash
cd UniMatch
DATASET=rescuenet SPLIT=25 METHOD=unimatch EXP=r101 \
SAVE_PATH=exp/rescuenet/unimatch/r101/25/manual \
bash scripts/train.sh 4 29500
```

Change `DATASET` to `floodnet`, `SPLIT` to `12_5`, `25`, or `50`, and `METHOD` to `unimatch`, `fixmatch`, or `supervised`.

## Direct Evaluation

UniMatch's local evaluator reads `best.pth` from `--save-path`:

```bash
cd UniMatch
torchrun --nproc_per_node=1 --master_port=29510 evaluate.py \
  --config configs/rescuenet.yaml \
  --labeled-id-path splits/rescuenet/25/labeled.txt \
  --unlabeled-id-path splits/rescuenet/25/unlabeled.txt \
  --save-path exp/rescuenet/unimatch/r101/25/manual \
  --port 29510
```

For comparable cross-method metrics, prefer the root `tools/evaluate_semi_supervised_final.py` evaluator.

## Upstream Reference

Original project: <https://github.com/LiheYoung/UniMatch>
