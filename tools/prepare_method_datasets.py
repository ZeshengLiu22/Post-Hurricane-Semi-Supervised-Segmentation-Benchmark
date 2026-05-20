#!/usr/bin/env python3
"""Prepare method-local dataset trees for the 12.5/25/50 post-disaster splits.

The heavy image/label directories are symlinked from DATASET_ROOT, or from a
Dataset directory next to this repository by default.
Split/config files are generated in the formats expected by each method.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

try:
    from PIL import Image
except ImportError as exc:  # pragma: no cover - this is an environment setup helper.
    raise SystemExit("Pillow is required to create blank unlabeled masks.") from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = Path(os.environ.get("DATASET_ROOT", PROJECT_ROOT.parent / "Dataset")).expanduser()
PRETRAIN_SOURCE_ROOT = Path(os.environ.get("PRETRAIN_SOURCE_ROOT", PROJECT_ROOT.parent / "Pretrain_Weights")).expanduser()
RATIOS = ("12_5", "25", "50")
STALE_RATIOS = ("75",)
BENCHMARK_EPOCHS = 150
BENCHMARK_BATCH_SIZE = 4
PRETRAINED_WEIGHTS = (
    (
        "UniMatch ResNet-101 backbone",
        PRETRAIN_SOURCE_ROOT / "unimatch" / "resnet101.pth",
        PROJECT_ROOT / "UniMatch" / "pretrained" / "resnet101.pth",
    ),
    (
        "S4MC ResNet-101 ImageNet backbone",
        PRETRAIN_SOURCE_ROOT / "s4mc" / "resnet101.pth",
        PROJECT_ROOT / "s4mc" / "resnet101.pth",
    ),
    (
        "Dual-Teacher MiT-B1 backbone",
        PRETRAIN_SOURCE_ROOT / "dualteacher" / "mit_b1.pth",
        PROJECT_ROOT / "dual_teacher" / "pretrained" / "mit_b1.pth",
    ),
    (
        "ClassMix COCO ResNet-101 backbone",
        PRETRAIN_SOURCE_ROOT / "classmix" / "resnet101COCO-41f33a49.pth",
        PROJECT_ROOT / "ClassMix" / "pretrained" / "resnet101COCO-41f33a49.pth",
    ),
)

DATASETS = {
    "rescuenet": {
        "title": "RescueNet",
        "nclass": 11,
        "classmix_config": "configRescueNet",
        "train_alias": "train-set",
        "reco_train_alias": "trainset_all",
        "s4mc_title": "RescueNet",
        "s4mc_lr": 0.005,
        "source": SOURCE_ROOT / "RescueNet",
    },
    "floodnet": {
        "title": "FloodNet",
        "nclass": 10,
        "classmix_config": "configFloodNet",
        "train_alias": "train",
        "reco_train_alias": "trainset",
        "s4mc_title": "FloodNet",
        "s4mc_lr": 0.01,
        "source": SOURCE_ROOT / "FloodNet",
    },
}


def read_ids(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def validation_ids(source: Path) -> list[str]:
    ids = [p.stem for p in (source / "Validation" / "val-org-img").glob("*.jpg")]
    return sorted(ids, key=lambda x: int(x) if x.isdigit() else x)


def write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def is_lfs_pointer(path: Path) -> bool:
    if not path.exists() or not path.is_file():
        return False
    return path.read_bytes()[:80].startswith(b"version https://git-lfs.github.com/spec")


def copy_pretrained_weights() -> None:
    copied = []
    for label, source, target in PRETRAINED_WEIGHTS:
        target_ok = target.exists() and target.is_file() and not is_lfs_pointer(target)
        if target_ok and (not source.exists() or target.stat().st_size == source.stat().st_size):
            continue
        if not source.exists():
            raise SystemExit(
                f"Missing source pretrained weight for {label}: {source}\n"
                f"Set PRETRAIN_SOURCE_ROOT or place the file at the method-local target: {target}"
            )
        if is_lfs_pointer(source):
            raise SystemExit(f"Source pretrained weight is only a Git LFS pointer: {source}")

        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied.append(target.relative_to(PROJECT_ROOT))

    if copied:
        print("Copied pretrained weights:")
        for path in copied:
            print(f"- {path}")


def cleanup_stale_ratio_artifacts() -> None:
    for key, meta in DATASETS.items():
        title = meta["s4mc_title"]
        classmix_prefix = meta["classmix_config"]
        for ratio in STALE_RATIOS:
            paths = [
                PROJECT_ROOT / "splits" / f"{key}-labeled-{ratio}.txt",
                PROJECT_ROOT / "splits" / f"{key}-unlabeled-{ratio}.txt",
                PROJECT_ROOT / "UniMatch" / "splits" / key / ratio,
                PROJECT_ROOT / "ClassMix" / "data" / key / "splits" / ratio,
                PROJECT_ROOT / "ClassMix" / "configs" / f"{classmix_prefix}{ratio}.json",
                PROJECT_ROOT / "reco" / "dataset" / key / f"train_labeled_{ratio}.txt",
                PROJECT_ROOT / "reco" / "dataset" / key / f"train_unlabeled_{ratio}.txt",
                PROJECT_ROOT / "s4mc" / "dataset" / title / f"{key}-labeled-{ratio}.txt",
                PROJECT_ROOT / "s4mc" / "dataset" / title / f"{key}-unlabeled-{ratio}.txt",
                PROJECT_ROOT / "s4mc" / f"config_{key}_{ratio}.yaml",
                PROJECT_ROOT / "dual_teacher" / "data" / f"{key}_{ratio}",
            ]
            for path in paths:
                if path.exists() or path.is_symlink():
                    remove_path(path)


def ensure_symlink(link: Path, target: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink():
        if link.resolve() == target.resolve():
            return
        link.unlink()
    elif link.exists():
        raise RuntimeError(f"Refusing to replace non-symlink path: {link}")
    link.symlink_to(target)


def clear_generated_symlink_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for entry in path.iterdir():
        if entry.is_symlink():
            entry.unlink()
        else:
            raise RuntimeError(f"Refusing to replace non-symlink path in generated split dir: {entry}")


def sync_file_symlinks(link_dir: Path, target_dir: Path, filenames: list[str]) -> None:
    clear_generated_symlink_dir(link_dir)
    for filename in filenames:
        target = target_dir / filename
        if not target.exists():
            raise FileNotFoundError(f"Expected dataset file not found: {target}")
        (link_dir / filename).symlink_to(target)


def ensure_blank_mask(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    Image.new("L", (4000, 3000), color=0).save(path)


def setup_unimatch_or_classmix(method_dir: Path, key: str, meta: dict[str, object]) -> None:
    source = meta["source"]
    train_alias = meta["train_alias"]
    dataset_root = method_dir / "dataset" / key

    ensure_symlink(dataset_root / train_alias / "train-org-img", source / "Train" / "train-org-img")
    ensure_symlink(dataset_root / train_alias / "train-label-img", source / "Train" / "train-label-img")
    ensure_blank_mask(dataset_root / train_alias / "train-unlabel-img" / "blank.png")
    ensure_symlink(dataset_root / "val" / "val-org-img", source / "Validation" / "val-org-img")
    ensure_symlink(dataset_root / "val" / "val-label-img", source / "Validation" / "val-label-img")


def path_split_lines(key: str, ids: list[str], labeled: bool) -> list[str]:
    train_alias = DATASETS[key]["train_alias"]
    img_prefix = f"{train_alias}/train-org-img"
    if labeled:
        return [f"{img_prefix}/{item}.jpg {train_alias}/train-label-img/{item}_lab.png" for item in ids]
    return [f"{img_prefix}/{item}.jpg {train_alias}/train-unlabel-img/blank.png" for item in ids]


def val_path_lines(key: str, ids: list[str]) -> list[str]:
    return [f"val/val-org-img/{item}.jpg val/val-label-img/{item}_lab.png" for item in ids]


def setup_unimatch_splits(key: str, split_data: dict[str, dict[str, list[str]]], val_ids: list[str]) -> None:
    split_root = PROJECT_ROOT / "UniMatch" / "splits" / key
    write_lines(split_root / "val.txt", val_path_lines(key, val_ids))
    for ratio in RATIOS:
        write_lines(split_root / ratio / "labeled.txt", path_split_lines(key, split_data[ratio]["labeled"], True))
        write_lines(split_root / ratio / "unlabeled.txt", path_split_lines(key, split_data[ratio]["unlabeled"], False))


def setup_classmix_splits(key: str, split_data: dict[str, dict[str, list[str]]], val_ids: list[str]) -> None:
    split_root = PROJECT_ROOT / "ClassMix" / "data" / key
    write_lines(split_root / "val.txt", val_path_lines(key, val_ids))
    for ratio in RATIOS:
        write_lines(split_root / "splits" / ratio / "labeled.txt", path_split_lines(key, split_data[ratio]["labeled"], True))
        write_lines(split_root / "splits" / ratio / "unlabeled.txt", path_split_lines(key, split_data[ratio]["unlabeled"], False))

    # Legacy paths remain useful for quick smoke tests; make them mirror the 25% split.
    train_alias = DATASETS[key]["train_alias"]
    write_lines(split_root / train_alias / "labeled.txt", path_split_lines(key, split_data["25"]["labeled"], True))
    write_lines(split_root / train_alias / "unlabeled.txt", path_split_lines(key, split_data["25"]["unlabeled"], False))


def setup_reco_dataset(key: str, split_data: dict[str, dict[str, list[str]]], val_ids: list[str]) -> None:
    meta = DATASETS[key]
    source = meta["source"]
    dataset_root = PROJECT_ROOT / "reco" / "dataset" / key
    train_alias = meta["reco_train_alias"]

    ensure_symlink(dataset_root / train_alias / "train-org-img", source / "Train" / "train-org-img")
    ensure_symlink(dataset_root / train_alias / "train-label-img", source / "Train" / "train-label-img")
    ensure_symlink(dataset_root / "validationset" / "val-org-img", source / "Validation" / "val-org-img")
    ensure_symlink(dataset_root / "validationset" / "val-label-img", source / "Validation" / "val-label-img")
    write_lines(dataset_root / "val.txt", val_ids)
    for ratio in RATIOS:
        write_lines(dataset_root / f"train_labeled_{ratio}.txt", split_data[ratio]["labeled"])
        write_lines(dataset_root / f"train_unlabeled_{ratio}.txt", split_data[ratio]["unlabeled"])

    write_lines(dataset_root / "train_labeled.txt", split_data["25"]["labeled"])
    write_lines(dataset_root / "train_unlabeled.txt", split_data["25"]["unlabeled"])
    (PROJECT_ROOT / "reco" / "model_weights").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "reco" / "logging").mkdir(parents=True, exist_ok=True)


def setup_s4mc_dataset(key: str, split_data: dict[str, dict[str, list[str]]], val_ids: list[str]) -> None:
    meta = DATASETS[key]
    source = meta["source"]
    title = meta["s4mc_title"]
    dataset_root = PROJECT_ROOT / "s4mc" / "dataset" / title

    ensure_symlink(dataset_root / "Train" / "org-img", source / "Train" / "train-org-img")
    ensure_symlink(dataset_root / "Train" / "label-img", source / "Train" / "train-label-img")
    ensure_symlink(dataset_root / "Validation" / "org-img", source / "Validation" / "val-org-img")
    ensure_symlink(dataset_root / "Validation" / "label-img", source / "Validation" / "val-label-img")
    write_lines(dataset_root / f"{key}-val.txt", val_ids)
    for ratio in RATIOS:
        write_lines(dataset_root / f"{key}-labeled-{ratio}.txt", split_data[ratio]["labeled"])
        write_lines(dataset_root / f"{key}-unlabeled-{ratio}.txt", split_data[ratio]["unlabeled"])


def setup_dual_teacher_dataset(key: str, split_data: dict[str, dict[str, list[str]]]) -> None:
    source = DATASETS[key]["source"]
    train_img_dir = source / "Train" / "train-org-img"
    train_label_dir = source / "Train" / "train-label-img"

    for ratio in RATIOS:
        dataset_root = PROJECT_ROOT / "dual_teacher" / "data" / f"{key}_{ratio}"
        images_root = dataset_root / "images"
        labels_root = dataset_root / "annotations"
        labeled_ids = split_data[ratio]["labeled"]
        unlabeled_ids = split_data[ratio]["unlabeled"]

        ensure_symlink(images_root / "train", train_img_dir)
        ensure_symlink(labels_root / "train", train_label_dir)
        ensure_symlink(images_root / "val-org-img", source / "Validation" / "val-org-img")
        ensure_symlink(labels_root / "val-label-img", source / "Validation" / "val-label-img")

        sync_file_symlinks(
            images_root / "train-org-img-l",
            train_img_dir,
            [f"{item}.jpg" for item in labeled_ids],
        )
        sync_file_symlinks(
            labels_root / "train-label-img-l",
            train_label_dir,
            [f"{item}_lab.png" for item in labeled_ids],
        )
        sync_file_symlinks(
            images_root / "train-org-img-u",
            train_img_dir,
            [f"{item}.jpg" for item in unlabeled_ids],
        )
        sync_file_symlinks(
            labels_root / "train-label-img-u",
            train_label_dir,
            [f"{item}_lab.png" for item in unlabeled_ids],
        )


def classmix_config(key: str, ratio: str, labeled_count: int) -> dict[str, object]:
    meta = DATASETS[key]
    steps_per_epoch = max(1, labeled_count // BENCHMARK_BATCH_SIZE)
    return {
        "dataset": key,
        "ignore_label": 255,
        "model": "DeepLab",
        "pretrained": "coco",
        "seed": 1,
        "training": {
            "batch_size": BENCHMARK_BATCH_SIZE,
            "epochs": BENCHMARK_EPOCHS,
            "data": {
                "crop": True,
                "input_size": "750,750",
                "labeled_samples": labeled_count,
                "scale": True,
                "split_id_list": 1000,
                "split_percent": ratio,
            },
            "learning_rate": 0.00025,
            "lr_schedule": "Poly",
            "lr_schedule_power": 0.9,
            "momentum": 0.9,
            "num_iterations": steps_per_epoch * BENCHMARK_EPOCHS,
            "num_workers": 4,
            "optimizer": "SGD",
            "unlabeled": {
                "blur": False,
                "color_jitter": False,
                "consistency_loss": "CE",
                "consistency_weight": 1,
                "flip": False,
                "pixel_weight": "threshold_uniform",
                "mix_mask": "class",
                "train_unlabeled": True,
            },
            "use_sync_batchnorm": True,
            "weight_decay": 0.0005,
        },
        "utils": {
            "checkpoint_dir": f"./checkpoints/{key}_{ratio}",
            "log_per_iter": 200,
            "save_best_model": True,
            "save_checkpoint_every": 1000,
            "tensorboard": True,
            "val_per_iter": 5000,
        },
    }


def write_classmix_configs(key: str, split_data: dict[str, dict[str, list[str]]]) -> None:
    prefix = DATASETS[key]["classmix_config"]
    config_dir = PROJECT_ROOT / "ClassMix" / "configs"
    for ratio in RATIOS:
        config = classmix_config(key, ratio, len(split_data[ratio]["labeled"]))
        path = config_dir / f"{prefix}{ratio}.json"
        path.write_text(json.dumps(config, indent=4) + "\n", encoding="utf-8")
    legacy_config = classmix_config(key, "25", len(split_data["25"]["labeled"]))
    (config_dir / f"{prefix}.json").write_text(json.dumps(legacy_config, indent=4) + "\n", encoding="utf-8")


def s4mc_config_text(key: str, ratio: str, labeled_count: int) -> str:
    meta = DATASETS[key]
    title = meta["s4mc_title"]
    lr = meta["s4mc_lr"]
    nclass = meta["nclass"]
    return f"""dataset:
  type: {key}_semi
  train:
    data_root: dataset/{title}/Train
    data_list: dataset/{title}/{key}-labeled-{ratio}.txt
    flip: True
    GaussianBlur: False
    rand_resize: [0.5, 2.0]
    crop:
      type: rand
      size: [750, 750]
  val:
    data_root: dataset/{title}/Validation
    data_list: dataset/{title}/{key}-val.txt
    crop:
      type: center
      size: [3000, 4000]
  batch_size: {BENCHMARK_BATCH_SIZE}
  n_sup: {labeled_count}
  noise_std: 0.1
  workers: 2
  mean: [123.675, 116.28, 103.53]
  std: [58.395, 57.12, 57.375]
  ignore_label: 255

trainer:
  epochs: {BENCHMARK_EPOCHS}
  eval_on: True
  optimizer:
    type: SGD
    kwargs:
      lr: {lr}
      momentum: 0.9
      weight_decay: 0.0001
  lr_scheduler:
    mode: poly
    kwargs:
      power: 0.9
  unsupervised:
    TTA: False
    drop_percent: 60
    apply_aug: cutmix
    two_sample: True
    neigborhood_size: 4
    n_neigbors: 1
    indicator: "margin"
  contrastive:
    low_rank: 3
    high_rank: 20
    current_class_threshold: 0.3
    current_class_negative_threshold: 1
    unsupervised_entropy_ignore: 80
    low_entropy_threshold: 20
    num_negatives: 50
    num_queries: 256
    temperature: 0.5

distributed_params:

main_mode:
  eval: False
  compare: False
  compared_pretrain:

saver:
  snapshot_dir: checkpoints
  pretrain: ''

criterion:
  type: CELoss
  kwargs:
    use_weight: False
  cons:
    sample: True
    gamma: 2

net:
  num_classes: {nclass}
  sync_bn: True
  ema_decay: 0.99
  encoder:
    type: s4mc_utils.models.resnet.resnet101
    kwargs:
      multi_grid: True
      zero_init_residual: True
      fpn: True
      replace_stride_with_dilation: [False, True, True]
  decoder:
    type: s4mc_utils.models.decoder.dec_deeplabv3_plus
    kwargs:
      inner_planes: 256
      dilations: [12, 24, 36]
"""


def write_s4mc_configs(key: str, split_data: dict[str, dict[str, list[str]]]) -> None:
    for ratio in RATIOS:
        path = PROJECT_ROOT / "s4mc" / f"config_{key}_{ratio}.yaml"
        path.write_text(s4mc_config_text(key, ratio, len(split_data[ratio]["labeled"])), encoding="utf-8")


def main() -> None:
    if not SOURCE_ROOT.exists():
        raise SystemExit(f"Dataset source not found: {SOURCE_ROOT}")

    cleanup_stale_ratio_artifacts()

    for key, meta in DATASETS.items():
        source = meta["source"]
        if not source.exists():
            raise SystemExit(f"Dataset source not found: {source}")

        split_data: dict[str, dict[str, list[str]]] = {}
        for ratio in RATIOS:
            split_data[ratio] = {
                "labeled": read_ids(PROJECT_ROOT / "splits" / f"{key}-labeled-{ratio}.txt"),
                "unlabeled": read_ids(PROJECT_ROOT / "splits" / f"{key}-unlabeled-{ratio}.txt"),
            }
        val_ids = validation_ids(source)

        setup_unimatch_or_classmix(PROJECT_ROOT / "UniMatch", key, meta)
        setup_unimatch_or_classmix(PROJECT_ROOT / "ClassMix", key, meta)
        setup_unimatch_splits(key, split_data, val_ids)
        setup_classmix_splits(key, split_data, val_ids)
        setup_reco_dataset(key, split_data, val_ids)
        setup_s4mc_dataset(key, split_data, val_ids)
        setup_dual_teacher_dataset(key, split_data)
        write_classmix_configs(key, split_data)
        write_s4mc_configs(key, split_data)

    copy_pretrained_weights()
    print("Prepared method-local datasets, split files, and generated configs.")


if __name__ == "__main__":
    main()
