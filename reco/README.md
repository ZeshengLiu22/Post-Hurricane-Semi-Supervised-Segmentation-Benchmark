# ReCo

This directory contains the adapted ReCo code used in the post-hurricane semi-supervised segmentation benchmark.

Use the root project README for the full benchmark workflow. This file only covers direct ReCo commands.

## Setup

From the repository root:

```bash
python3 tools/prepare_method_datasets.py
python3 tools/check_pretrained_weights.py
```

The setup creates `reco/dataset/{dataset}/` symlinks and split files for `12_5`, `25`, and `50`.

Activate the ReCo environment before running:

```bash
conda activate reco
```

ReCo uses Hugging Face Accelerate for multi-process training.

## Run All Benchmark Splits

From the repository root:

```bash
bash scripts/run_reco_benchmark.sh
```

Useful overrides:

```bash
RUN_ID=trial1 NUM_GPUS=4 CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_reco_benchmark.sh
PREP_DATA=0 EPOCHS=150 BASE_PORT=29720 bash scripts/run_reco_benchmark.sh
OUTPUT_ROOT=outputs/reco_trial RUN_ID=trial1 bash scripts/run_reco_benchmark.sh
```

## Direct Training

Run from `reco/`:

```bash
cd reco
accelerate launch --num_processes 4 --main_process_port 29700 train_semisup_acc.py \
  --dataset rescuenet \
  --split 25 \
  --num_labels 15 \
  --apply_aug classmix \
  --apply_reco \
  --backbone deeplabv3p \
  --epochs 150 \
  --output-root outputs/reco \
  --run-id manual
```

Change `--dataset` to `floodnet` and `--split` to `12_5`, `25`, or `50` as needed.

## Direct Evaluation

```bash
cd reco
python eval.py --dataset rescuenet --split 25 --output-root outputs/reco --run-id manual
```

Pass `--model-path path/to/checkpoint.pth` to evaluate a specific checkpoint. For comparable cross-method metrics, prefer the root `tools/evaluate_semi_supervised_final.py` evaluator.

## Upstream Reference

Original project: <https://github.com/lorenmt/reco>
