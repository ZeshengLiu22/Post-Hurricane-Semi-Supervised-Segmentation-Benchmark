# Post-Hurricane Semi-Supervised Segmentation

[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Code release for **Benchmarking Semi-Supervised Semantic Segmentation Models for Post-Disaster Scene Understanding**.

This repository adapts five semi-supervised semantic segmentation methods to FloodNet and RescueNet post-hurricane imagery. It includes shared data preparation, split generation, benchmark launchers, and final evaluation scripts so the methods can be run under one common protocol.

## Benchmark Scope

| Item | Setting |
| --- | --- |
| Datasets | FloodNet, RescueNet |
| Label ratios | 12.5%, 25%, 50% labeled data |
| Methods | UniMatch, ClassMix, ReCo, S4MC, Dual-Teacher |
| Training budget | 150 epochs, batch size 4, 750x750 crop |
| Method policy | Optimizer, scheduler, learning rate, and core method logic stay method-native |
| Split files | Root split IDs are tracked in `splits/` |

ClassMix is iteration-based, so generated ClassMix configs set `num_iterations = floor(num_labeled / 4) * 150`. The setup script creates method-local split/config files from the root split IDs.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `README.md` | Project-level overview and benchmark workflow |
| `command.md` | Longer command notebook for manual and historical runs |
| `splits/` | Canonical FloodNet and RescueNet labeled/unlabeled IDs |
| `scripts/` | Root launchers for per-method and per-ratio benchmark sweeps |
| `tools/prepare_method_datasets.py` | Creates method-local dataset symlinks, split files, configs, and pretrained weight copies |
| `tools/check_pretrained_weights.py` | Checks required method-local pretrained weights |
| `tools/run_benchmark_sweep.py` | Unified ratio runner used by `scripts/run_12_5.sh`, `scripts/run_25.sh`, and `scripts/run_50.sh` |
| `tools/evaluate_semi_supervised_final.py` | Shared final evaluator for trained benchmark checkpoints |
| `UniMatch/`, `ClassMix/`, `reco/`, `s4mc/`, `dual_teacher/` | Adapted method code and short method-specific run guides |

## Data And Weights

The setup script expects datasets in `DATASET_ROOT`, defaulting to `../Dataset` relative to this repository:

```text
Dataset/
+-- FloodNet/
|   +-- Train/
|   |   +-- train-org-img/
|   |   +-- train-label-img/
|   +-- Validation/
|       +-- val-org-img/
|       +-- val-label-img/
+-- RescueNet/
    +-- Train/
    |   +-- train-org-img/
    |   +-- train-label-img/
    +-- Validation/
        +-- val-org-img/
        +-- val-label-img/
```

Required pretrained weights are included in the repository with Git LFS:

```text
+-- UniMatch/pretrained/resnet101.pth
+-- ClassMix/pretrained/resnet101COCO-41f33a49.pth
+-- s4mc/resnet101.pth
+-- dual_teacher/pretrained/mit_b1.pth
```

Run the shared preparation once after cloning, moving the repo, changing dataset paths, or changing pretrained paths:

```bash
python3 tools/prepare_method_datasets.py
python3 tools/check_pretrained_weights.py
```

Useful overrides:

```bash
DATASET_ROOT=/path/to/Dataset PRETRAIN_SOURCE_ROOT=/path/to/Pretrain_Weights python3 tools/prepare_method_datasets.py
```

`PRETRAIN_SOURCE_ROOT` is only needed if you want to replace the committed Git LFS weights from another local source. The setup script creates symlinks for image/label folders and generated files under each method directory. Large datasets, checkpoints, and run outputs should stay out of git.

## Running The Benchmark

Each method has its own conda environment. Activate the matching environment before using a per-method launcher:

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

Common launcher overrides:

```bash
RUN_ID=trial1 NUM_GPUS=4 CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_unimatch_benchmark.sh
PREP_DATA=0 RUN_ID=trial1 NUM_GPUS=4 BASE_PORT=29600 bash scripts/run_classmix_benchmark.sh
BASE_PORT=31000 RUN_ID=trial1 bash scripts/run_s4mc_benchmark.sh
AMP=true RUN_ID=trial1 bash scripts/run_dual_teacher_benchmark.sh
```

To run all methods for a single label ratio, use the ratio launchers from the repository root:

```bash
bash scripts/run_12_5.sh
bash scripts/run_25.sh
bash scripts/run_50.sh
```

The ratio launchers call `tools/run_benchmark_sweep.py`, preflight the method environments under `CONDA_ENV_ROOT`, and write outputs under `benchmark_runs/{RUN_ID}` by default. They use `tmux` by default for long runs; set `USE_TMUX=0` to run in the current shell.

Useful ratio-run overrides:

```bash
CONDA_ENV_ROOT=$HOME/.conda/envs RUN_ID=ratio25 NUM_GPUS=4 DEVICES=0,1,2,3 bash scripts/run_25.sh
METHODS="s4mc classmix unimatch reco dual_teacher" DATASETS="floodnet rescuenet" EPOCHS=150 bash scripts/run_50.sh
PREP_DATA=0 KEEP_CHECKPOINTS=1 USE_TMUX=0 bash scripts/run_12_5.sh
```

## Output Locations

| Method | Main outputs |
| --- | --- |
| UniMatch | `UniMatch/exp/{dataset}/{method}/{exp}/{split}/{RUN_ID}/latest.pth`, `best.pth`, TensorBoard events, logs |
| ClassMix | `ClassMix/checkpoints/{dataset}_{split}/{timestamp}-{dataset}_{split}_{RUN_ID}/checkpoint-*.pth`, `best_model.pth`, validation outputs |
| ReCo | `reco/outputs/reco/model_weights/*_{RUN_ID}.pth`, `reco/outputs/reco/logging/*_{RUN_ID}.npy`, logs in `run_logs/reco/` |
| S4MC | `s4mc/checkpoints/{dataset}_{split}_{RUN_ID}/ckpt.pth`, `ckpt_best.pth`, TensorBoard events |
| Dual-Teacher | `dual_teacher/work_dirs/benchmark_{dataset}_{split}_{RUN_ID}/best_weights.pth`, config dump, logs |

## Final Evaluation

Use the shared evaluator for comparable final metrics across methods:

```bash
python3 tools/evaluate_semi_supervised_final.py \
  --benchmark-root benchmark_runs \
  --run RUN_ID_OR_RUN_DIR \
  --methods unimatch classmix reco s4mc dual_teacher \
  --datasets floodnet rescuenet \
  --splits 12_5 25 50 \
  --eval-split test \
  --skip-completed
```

Related reporting helpers:

```bash
python3 tools/export_image_gt_750.py --split test --overwrite
python3 tools/generate_critical_class_metrics.py
```

## Method Guides

Each method directory now keeps a short README focused on direct method-level commands:

- [UniMatch](UniMatch/README.md)
- [ClassMix](ClassMix/README.md)
- [ReCo](reco/README.md)
- [S4MC](s4mc/README.md)
- [Dual-Teacher](dual_teacher/README.md)

For the complete command notebook, see [command.md](command.md).

## Upstream Code

This benchmark builds on the original public implementations of UniMatch, ClassMix, ReCo, S4MC, and Dual-Teacher. Please cite the original method papers when using their code or comparing against them.
