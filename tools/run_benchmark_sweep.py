#!/usr/bin/env python3
"""Run the all-method post-disaster segmentation benchmark sweep."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_ROOT = Path(
    os.environ.get("CONDA_ENV_ROOT", os.environ.get("ENV_ROOT", Path.home() / ".conda" / "envs"))
).expanduser()
CLASSMIX_COCO_PRETRAIN = Path(
    os.environ.get(
        "CLASSMIX_COCO_PRETRAIN",
        PROJECT_ROOT / "ClassMix" / "pretrained" / "resnet101COCO-41f33a49.pth",
    )
).expanduser()

METHODS = ["unimatch", "classmix", "reco", "s4mc", "dual_teacher"]
DATASETS = ["floodnet", "rescuenet"]
SPLITS = ["12_5", "25", "50"]

LOADER_SETTINGS = {
    "unimatch": {
        "num_workers": 4,
        "val_num_workers": 4,
        "pin_memory": True,
        "prefetch_factor": 2,
        "persistent_workers": True,
    },
    "classmix": {
        "num_workers": 8,
        "val_num_workers": 8,
        "pin_memory": True,
        "prefetch_factor": 4,
        "persistent_workers": True,
    },
    "reco": {
        "num_workers": 4,
        "val_num_workers": 4,
        "pin_memory": False,
        "prefetch_factor": 4,
        "persistent_workers": True,
    },
    "s4mc": {
        "num_workers": 8,
        "val_num_workers": 8,
        "pin_memory": True,
        "prefetch_factor": 2,
        "persistent_workers": True,
    },
    "dual_teacher": {
        "num_workers": 8,
        "val_num_workers": 8,
        "pin_memory": True,
        "prefetch_factor": 2,
        "persistent_workers": True,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--devices", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3"))
    parser.add_argument("--base-port", type=int, default=34200)
    parser.add_argument("--run-id", default=time.strftime("benchmark_%Y%m%d_%H%M%S"))
    parser.add_argument("--output-root", default=str(PROJECT_ROOT / "benchmark_runs"))
    parser.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    parser.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    parser.add_argument("--splits", nargs="+", default=SPLITS, choices=SPLITS)
    parser.add_argument("--prepare-data", action="store_true")
    parser.add_argument("--keep-checkpoints", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--force", action="store_true", help="rerun rows that already completed in this run root")
    return parser.parse_args()


def bool_arg(value: bool) -> str:
    return "true" if value else "false"


def title_dataset(dataset: str) -> str:
    return "FloodNet" if dataset == "floodnet" else "RescueNet"


def ensure_crop_750(value: object, label: str) -> None:
    if value not in (750, "750,750", [750, 750], (750, 750)):
        raise RuntimeError(f"Refusing to run {label} with crop={value}; expected 750.")


def write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def write_unimatch_config(dataset: str, split: str, epochs: int, config_dir: Path) -> Path:
    source = PROJECT_ROOT / "UniMatch" / "configs" / f"{dataset}.yaml"
    with source.open("r", encoding="utf-8") as f:
        cfg = yaml.load(f, Loader=yaml.Loader)

    ensure_crop_750(cfg.get("crop_size"), "UniMatch")
    settings = LOADER_SETTINGS["unimatch"]
    cfg.update(
        {
            "epochs": epochs,
            "crop_size": 750,
            "eval_interval": 1,
            "num_workers": settings["num_workers"],
            "val_num_workers": settings["val_num_workers"],
            "pin_memory": settings["pin_memory"],
            "prefetch_factor": settings["prefetch_factor"],
            "persistent_workers": settings["persistent_workers"],
        }
    )

    out = config_dir / f"unimatch_{dataset}_{split}_{epochs}epoch.yaml"
    write_yaml(out, cfg)
    return out


def write_classmix_config(dataset: str, split: str, epochs: int, num_gpus: int, config_dir: Path, exp_dir: Path) -> Path:
    source = PROJECT_ROOT / "ClassMix" / "configs" / f"config{title_dataset(dataset)}{split}.json"
    with source.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    if not CLASSMIX_COCO_PRETRAIN.exists():
        raise FileNotFoundError(f"Missing ClassMix COCO checkpoint: {CLASSMIX_COCO_PRETRAIN}")
    cfg["pretrained"] = str(CLASSMIX_COCO_PRETRAIN)

    training = cfg["training"]
    ensure_crop_750(training["data"]["input_size"], "ClassMix")
    settings = LOADER_SETTINGS["classmix"]
    training["num_workers"] = settings["num_workers"]
    training["unlabeled_num_workers"] = settings["num_workers"]
    training["val_num_workers"] = settings["val_num_workers"]
    training["pin_memory"] = settings["pin_memory"]
    training["prefetch_factor"] = settings["prefetch_factor"]
    training["persistent_workers"] = settings["persistent_workers"]

    # DDP consumes only the per-rank slice of the labeled set per epoch.
    local_samples = math.ceil(int(training["data"]["labeled_samples"]) / num_gpus)
    steps_per_epoch = max(1, math.ceil(local_samples / int(training["batch_size"])))
    training["epochs"] = None
    training["num_iterations"] = steps_per_epoch * epochs
    cfg["utils"]["val_per_iter"] = steps_per_epoch
    cfg["utils"]["checkpoint_dir"] = str(exp_dir)

    out = config_dir / f"classmix_{dataset}_{split}_{epochs}epoch.json"
    write_json(out, cfg)
    return out


def write_s4mc_config(dataset: str, split: str, epochs: int, config_dir: Path, exp_dir: Path) -> Path:
    source = PROJECT_ROOT / "s4mc" / f"config_{dataset}_{split}.yaml"
    with source.open("r", encoding="utf-8") as f:
        cfg = yaml.load(f, Loader=yaml.Loader)

    ensure_crop_750(cfg["dataset"]["train"]["crop"]["size"], "S4MC")
    settings = LOADER_SETTINGS["s4mc"]
    cfg["trainer"]["epochs"] = epochs
    cfg["trainer"]["eval_on"] = True
    cfg["dataset"]["workers"] = settings["num_workers"]
    cfg["dataset"]["val_workers"] = settings["val_num_workers"]
    cfg["dataset"]["pin_memory"] = settings["pin_memory"]
    cfg["dataset"]["prefetch_factor"] = settings["prefetch_factor"]
    cfg["dataset"]["persistent_workers"] = settings["persistent_workers"]
    cfg["saver"]["snapshot_dir"] = str(exp_dir / "checkpoints")

    out = config_dir / f"s4mc_{dataset}_{split}_{epochs}epoch.yaml"
    write_yaml(out, cfg)
    return out


def command_for_experiment(
    method: str,
    dataset: str,
    split: str,
    args: argparse.Namespace,
    run_root: Path,
    port: int,
) -> tuple[list[str], Path]:
    name = f"{method}_{dataset}_{split}"
    config_dir = run_root / "configs" / method
    exp_dir = run_root / "exp" / method / f"{dataset}_{split}"

    if method == "unimatch":
        cfg = write_unimatch_config(dataset, split, args.epochs, config_dir)
        return [
            str(ENV_ROOT / "unimatch" / "bin" / "python"),
            "-m",
            "torch.distributed.run",
            "--nproc_per_node",
            str(args.num_gpus),
            "--master_addr",
            "localhost",
            "--master_port",
            str(port),
            "unimatch.py",
            "--config",
            str(cfg),
            "--labeled-id-path",
            f"splits/{dataset}/{split}/labeled.txt",
            "--unlabeled-id-path",
            f"splits/{dataset}/{split}/unlabeled.txt",
            "--save-path",
            str(exp_dir),
            "--port",
            str(port),
        ], PROJECT_ROOT / "UniMatch"

    if method == "classmix":
        cfg = write_classmix_config(dataset, split, args.epochs, args.num_gpus, config_dir, exp_dir)
        return [
            str(ENV_ROOT / "classmix" / "bin" / "torchrun"),
            "--nproc_per_node",
            str(args.num_gpus),
            "--master_addr",
            "localhost",
            "--master_port",
            str(port),
            "trainSSL.py",
            "--gpus",
            str(args.num_gpus),
            "--config",
            str(cfg),
            "--name",
            name,
            "--amp",
            "true",
            "--amp-dtype",
            "bf16",
            "--save-images",
            "false",
        ], PROJECT_ROOT / "ClassMix"

    if method == "reco":
        settings = LOADER_SETTINGS["reco"]
        return [
            str(ENV_ROOT / "reco" / "bin" / "accelerate"),
            "launch",
            "--num_processes",
            str(args.num_gpus),
            "--main_process_port",
            str(port),
            "train_semisup_acc.py",
            "--dataset",
            dataset,
            "--split",
            split,
            "--num_labels",
            "15",
            "--apply_aug",
            "classmix",
            "--apply_reco",
            "--backbone",
            "deeplabv3p",
            "--epochs",
            str(args.epochs),
            "--seed",
            "0",
            "--num-workers",
            str(settings["num_workers"]),
            "--val-num-workers",
            str(settings["val_num_workers"]),
            "--pin-memory",
            bool_arg(settings["pin_memory"]),
            "--prefetch-factor",
            str(settings["prefetch_factor"]),
            "--persistent-workers",
            bool_arg(settings["persistent_workers"]),
            "--output-root",
            str(exp_dir),
            "--run-id",
            args.run_id,
        ], PROJECT_ROOT / "reco"

    if method == "s4mc":
        cfg = write_s4mc_config(dataset, split, args.epochs, config_dir, exp_dir)
        return [
            str(ENV_ROOT / "s4mc" / "bin" / "torchrun"),
            "--nproc_per_node",
            str(args.num_gpus),
            "--master_port",
            str(port),
            "train_semi.py",
            "--config",
            str(cfg),
            "--seed",
            "42",
            "--name",
            name,
            "--amp",
            "true",
            "--port",
            str(port),
        ], PROJECT_ROOT / "s4mc"

    settings = LOADER_SETTINGS["dual_teacher"]
    train_file = "tools/train-flood.py" if dataset == "floodnet" else "tools/train-rescue.py"
    return [
        str(ENV_ROOT / "dual_teacher" / "bin" / "torchrun"),
        f"--nproc_per_node={args.num_gpus}",
        f"--master_port={port}",
        train_file,
        "--ddp",
        "--dual_teacher",
        "--backbone",
        "mit_b1",
        "--split",
        split,
        "--epochs",
        str(args.epochs),
        "--num-workers",
        str(settings["num_workers"]),
        "--val-num-workers",
        str(settings["val_num_workers"]),
        "--pin-memory",
        bool_arg(settings["pin_memory"]),
        "--prefetch-factor",
        str(settings["prefetch_factor"]),
        "--persistent-workers",
        bool_arg(settings["persistent_workers"]),
        "--amp",
        "false",
        "--port",
        str(port),
        "--work-dir",
        str(exp_dir),
    ], PROJECT_ROOT / "dual_teacher"


def env_for_method(method: str, args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.devices
    env["PYTHONUNBUFFERED"] = "1"
    env.pop("CUDA_LAUNCH_BLOCKING", None)

    if method == "dual_teacher":
        dual_teacher_root = PROJECT_ROOT / "dual_teacher"
        nvrtc_lib = ENV_ROOT / "dual_teacher" / "lib" / "python3.8" / "site-packages" / "nvidia" / "cuda_nvrtc" / "lib"
        env["DIST_BACKEND"] = env.get("DIST_BACKEND", "gloo")
        old_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{dual_teacher_root}:{old_pythonpath}" if old_pythonpath else str(dual_teacher_root)
        if nvrtc_lib.exists():
            old_ld = env.get("LD_LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = f"{nvrtc_lib}:{old_ld}" if old_ld else str(nvrtc_lib)
    return env


def remove_checkpoints(path: Path) -> None:
    if not path.exists():
        return
    for pattern in ("*.pth", "events.out.tfevents.*", "*.npy"):
        for artifact in path.rglob(pattern):
            artifact.unlink(missing_ok=True)


def completed_keys(summary_path: Path) -> set[tuple[str, str, str]]:
    if not summary_path.exists():
        return set()
    with summary_path.open("r", encoding="utf-8", newline="") as f:
        return {
            (row["method"], row["dataset"], row["split"])
            for row in csv.DictReader(f)
            if row.get("returncode") == "0"
        }


def append_row(summary_path: Path, row: dict[str, object]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    exists = summary_path.exists()
    with summary_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def run_prepare_data(run_root: Path) -> None:
    log_path = run_root / "prepare_method_datasets.log"
    cmd = [sys.executable, str(PROJECT_ROOT / "tools" / "prepare_method_datasets.py")]
    with log_path.open("w", encoding="utf-8") as log:
        log.write("COMMAND: " + " ".join(cmd) + "\n\n")
        log.flush()
        subprocess.run(cmd, cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)


def resolve_output_root(path: str) -> Path:
    output_root = Path(path).expanduser()
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    return output_root


def main() -> int:
    args = parse_args()
    run_root = resolve_output_root(args.output_root) / args.run_id
    log_dir = run_root / "logs"
    summary_path = run_root / "summary.csv"
    run_root.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"Benchmark root: {run_root}", flush=True)
    print(f"Epochs per experiment: {args.epochs}", flush=True)
    print(f"Devices: {args.devices}; num_gpus={args.num_gpus}", flush=True)
    print("Constraints: crop_size=750; eval every epoch; tuned loader settings per method.", flush=True)

    if args.prepare_data:
        print("Preparing method-specific datasets before benchmark...", flush=True)
        run_prepare_data(run_root)

    done = set() if args.force else completed_keys(summary_path)
    total = len(args.methods) * len(args.datasets) * len(args.splits)
    experiment_index = 0

    for method in args.methods:
        for dataset in args.datasets:
            for split in args.splits:
                experiment_index += 1
                key = (method, dataset, split)
                if key in done:
                    print(f"[{experiment_index}/{total}] SKIP {method} {dataset} {split}: already complete", flush=True)
                    continue

                port = args.base_port + experiment_index
                log_path = log_dir / method / f"{dataset}_{split}.log"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                cmd, cwd = command_for_experiment(method, dataset, split, args, run_root, port)
                env = env_for_method(method, args)

                print(f"\n[{experiment_index}/{total}] {method} {dataset} split {split}", flush=True)
                print(f"Log: {log_path}", flush=True)
                started = time.monotonic()
                with log_path.open("w", encoding="utf-8") as log:
                    log.write("COMMAND: " + " ".join(cmd) + "\n")
                    log.write(f"METHOD: {method}\nDATASET: {dataset}\nSPLIT: {split}\nEPOCHS: {args.epochs}\n")
                    log.write(f"LOADER_SETTINGS: {LOADER_SETTINGS[method]}\n\n")
                    log.flush()
                    completed = subprocess.run(cmd, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
                elapsed = time.monotonic() - started
                sec_per_epoch = elapsed / args.epochs

                exp_path = run_root / "exp" / method / f"{dataset}_{split}"
                if not args.keep_checkpoints:
                    remove_checkpoints(exp_path)

                settings = LOADER_SETTINGS[method]
                row = {
                    "method": method,
                    "dataset": dataset,
                    "split": split,
                    "epochs": args.epochs,
                    "elapsed_sec": f"{elapsed:.2f}",
                    "elapsed_hours": f"{elapsed / 3600:.2f}",
                    "sec_per_epoch": f"{sec_per_epoch:.2f}",
                    "projected_150_epoch_hours": f"{(sec_per_epoch * 150) / 3600:.2f}",
                    "projected_200_epoch_hours": f"{(sec_per_epoch * 200) / 3600:.2f}",
                    "returncode": completed.returncode,
                    "num_workers": settings["num_workers"],
                    "val_num_workers": settings["val_num_workers"],
                    "pin_memory": settings["pin_memory"],
                    "prefetch_factor": settings["prefetch_factor"],
                    "persistent_workers": settings["persistent_workers"],
                    "log_path": str(log_path),
                }
                append_row(summary_path, row)

                print(
                    f"Completed {method} {dataset} {split}: returncode={completed.returncode}, "
                    f"elapsed={elapsed:.2f}s, sec/epoch={sec_per_epoch:.2f}",
                    flush=True,
                )
                if completed.returncode != 0 and not args.continue_on_error:
                    print(f"Stopping after failure. See {log_path}", flush=True)
                    return completed.returncode

    print(f"\nSummary: {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
