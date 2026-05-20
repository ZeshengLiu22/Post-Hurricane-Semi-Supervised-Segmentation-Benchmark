# Dual-Teacher

This directory contains the adapted Dual-Teacher code used in the post-hurricane semi-supervised segmentation benchmark.

Use the root project README for the full benchmark workflow. This file only covers direct Dual-Teacher commands.

## Setup

From the repository root:

```bash
python3 tools/prepare_method_datasets.py
python3 tools/check_pretrained_weights.py
```

The setup creates split-specific symlink trees under `dual_teacher/data/{dataset}_{split}/` and expects `dual_teacher/pretrained/mit_b1.pth`.

Activate the Dual-Teacher environment before running:

```bash
conda activate dual_teacher
```

After installing dependencies, install the local package if needed:

```bash
cd dual_teacher
pip install -e . --user
```

This project has been run with Python 3.8, PyTorch 1.10.1+cu113, CUDA 11.3, MMCV 1.3.17, and MMSegmentation 0.11.0. Headless OpenCV is recommended on servers.

## Run All Benchmark Splits

From the repository root:

```bash
bash scripts/run_dual_teacher_benchmark.sh
```

Useful overrides:

```bash
RUN_ID=trial1 NUM_GPUS=4 CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_dual_teacher_benchmark.sh
PREP_DATA=0 EPOCHS=150 AMP=false BASE_PORT=29920 bash scripts/run_dual_teacher_benchmark.sh
AMP=true RUN_ID=amp_trial bash scripts/run_dual_teacher_benchmark.sh
```

Dual-Teacher defaults to full precision in the benchmark launcher because that matched the previously stable runs. Set `AMP=true` only for an explicit precision experiment.

## Direct Training

Run from `dual_teacher/`:

```bash
cd dual_teacher
torchrun --nproc_per_node=4 --master_port=29900 tools/train-rescue.py \
  --ddp \
  --dual_teacher \
  --backbone mit_b1 \
  --split 25 \
  --epochs 150 \
  --amp false \
  --port 29900 \
  --work-dir work_dirs/benchmark_rescuenet_25_manual

torchrun --nproc_per_node=4 --master_port=29901 tools/train-flood.py \
  --ddp \
  --dual_teacher \
  --backbone mit_b1 \
  --split 25 \
  --epochs 150 \
  --amp false \
  --port 29901 \
  --work-dir work_dirs/benchmark_floodnet_25_manual
```

Change `--split` to `12_5`, `25`, or `50`.

## Direct Evaluation

Metric scripts:

```bash
cd dual_teacher
python tools/eval-rescuenet.py \
  --config work_dirs/benchmark_rescuenet_25_manual/segformer.b1.512x512.rescuenet.160k.py \
  --checkpoint work_dirs/benchmark_rescuenet_25_manual/best_weights.pth \
  --backbone mit_b1 \
  --num-classes 11

python tools/eval-floodnet.py \
  --config work_dirs/benchmark_floodnet_25_manual/segformer.b1.512x512.floodnet.160k.py \
  --checkpoint work_dirs/benchmark_floodnet_25_manual/best_weights.pth \
  --backbone mit_b1 \
  --num-classes 10
```

Prediction export scripts:

```bash
cd dual_teacher
python tools/eval-dual-teacher-rescuenet.py \
  --val-img-dir data/rescuenet_25/images/val-org-img \
  --val-mask-dir data/rescuenet_25/annotations/val-label-img \
  --checkpoint work_dirs/benchmark_rescuenet_25_manual/best_weights.pth \
  --save-dir predictions/rescuenet_25

python tools/eval-dual-teacher-floodnet.py \
  --val-img-dir data/floodnet_25/images/val-org-img \
  --val-mask-dir data/floodnet_25/annotations/val-label-img \
  --checkpoint work_dirs/benchmark_floodnet_25_manual/best_weights.pth \
  --save-dir predictions/floodnet_25
```

For comparable cross-method metrics, prefer the root `tools/evaluate_semi_supervised_final.py` evaluator.

## Upstream Reference

Original project: <https://github.com/NaJaeMin92/Dual-Teacher>
