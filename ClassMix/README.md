# ClassMix

This directory contains the adapted ClassMix code used in the post-hurricane semi-supervised segmentation benchmark.

Use the root project README for the full benchmark workflow. This file only covers direct ClassMix commands.

## Setup

From the repository root:

```bash
python3 tools/prepare_method_datasets.py
python3 tools/check_pretrained_weights.py
```

The setup creates `ClassMix/dataset/` symlinks, `ClassMix/data/{dataset}/splits/` files, generated `configs/config{Dataset}{Split}.json` files, and expects `ClassMix/pretrained/resnet101COCO-41f33a49.pth`.

Activate the ClassMix environment before running:

```bash
conda activate classmix
```

## Run All Benchmark Splits

From the repository root:

```bash
bash scripts/run_classmix_benchmark.sh
```

Useful overrides:

```bash
RUN_ID=trial1 NUM_GPUS=4 CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_classmix_benchmark.sh
PREP_DATA=0 AMP=true AMP_DTYPE=bf16 BASE_PORT=29620 bash scripts/run_classmix_benchmark.sh
SAVE_IMAGES=true RUN_ID=debug_images bash scripts/run_classmix_benchmark.sh
```

## Direct Training

Run from `ClassMix/`. Multi-GPU training uses DDP through `torchrun`.

```bash
cd ClassMix
torchrun --nproc_per_node=4 --master_port=29600 trainSSL.py \
  --gpus 4 \
  --config ./configs/configRescueNet25.json \
  --name rescuenet_25_manual \
  --amp true \
  --amp-dtype bf16
```

Use `configFloodNet12_5.json`, `configFloodNet25.json`, `configFloodNet50.json`, `configRescueNet12_5.json`, `configRescueNet25.json`, or `configRescueNet50.json`.

## Direct Evaluation

```bash
cd ClassMix
python evaluateSSL.py --model-path checkpoints/path/to/best_model.pth
```

For comparable cross-method metrics, prefer the root `tools/evaluate_semi_supervised_final.py` evaluator.

## Upstream Reference

Original project: <https://github.com/WilhelmT/ClassMix>
