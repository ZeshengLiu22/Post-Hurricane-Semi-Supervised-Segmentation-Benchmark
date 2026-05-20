#!/usr/bin/env python3
"""Final metric and prediction export for semi-supervised benchmark runs.

The method folders contain several useful evaluation scripts, but their metric
definitions are not fully aligned. This script uses method-local code only to
load a checkpoint and produce a prediction. All reported metrics are computed
here from the same confusion-matrix implementation.

Default reporting:
* class 0/background is included in classwise IoU, mIoU, FWIoU, and per-image mIoU
* mIoU_no_background and per_image_mIoU_no_background are also emitted
* only --ignore-index pixels are excluded; default is none because these
  FloodNet/RescueNet masks use class 0 as background, not void
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image
try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - fallback for minimal method environments
    def tqdm(iterable, *args, **kwargs):
        return iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARK_ROOT = PROJECT_ROOT / "benchmark_runs"
DEFAULT_DATASET_ROOT = PROJECT_ROOT.parent / "Dataset"
DEFAULT_ENV_ROOT = Path(
    os.environ.get("CONDA_ENV_ROOT", os.environ.get("ENV_ROOT", Path.home() / ".conda" / "envs"))
).expanduser()

METHODS = ("unimatch", "classmix", "reco", "s4mc", "dual_teacher")
DATASETS = ("floodnet", "rescuenet")
SPLITS = ("12_5", "25", "50")
METHOD_DIRS = {
    "unimatch": "UniMatch",
    "classmix": "ClassMix",
    "reco": "reco",
    "s4mc": "s4mc",
    "dual_teacher": "dual_teacher",
}


DATASET_INFO = {
    "floodnet": {
        "title": "FloodNet",
        "num_classes": 10,
        "classes": [
            "background",
            "building flooded",
            "building non-flooded",
            "road flooded",
            "road non-flooded",
            "water",
            "tree",
            "vehicle",
            "pool",
            "grass",
        ],
        "palette": [
            (0, 0, 0),
            (255, 0, 0),
            (180, 120, 120),
            (160, 150, 20),
            (140, 140, 140),
            (61, 230, 250),
            (0, 82, 255),
            (255, 0, 245),
            (255, 235, 0),
            (4, 250, 7),
        ],
    },
    "rescuenet": {
        "title": "RescueNet",
        "num_classes": 11,
        "classes": [
            "background",
            "water",
            "building no damage",
            "building medium damage",
            "building major damage",
            "building total destruction",
            "vehicle",
            "road clear",
            "road blocked",
            "tree",
            "pool",
        ],
        "palette": [
            (0, 0, 0),
            (61, 230, 250),
            (180, 120, 120),
            (235, 255, 7),
            (255, 184, 6),
            (255, 0, 0),
            (255, 0, 245),
            (140, 140, 140),
            (160, 150, 20),
            (4, 250, 7),
            (255, 235, 0),
        ],
    },
}


@dataclass(frozen=True)
class Sample:
    image_id: str
    image_path: Path
    label_path: Path


@dataclass(frozen=True)
class Task:
    run_dir: Path
    method: str
    dataset: str
    split: str
    checkpoint: Path
    config: Path | None
    output_dir: Path


class MetricAccumulator:
    def __init__(self, num_classes: int, ignore_index: int | None):
        self.num_classes = int(num_classes)
        self.ignore_index = ignore_index
        self.confusion = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        self.per_image_rows: list[dict[str, object]] = []

    def _valid_mask(self, pred: np.ndarray, target: np.ndarray) -> np.ndarray:
        valid = (target >= 0) & (target < self.num_classes)
        valid &= (pred >= 0) & (pred < self.num_classes)
        if self.ignore_index is not None:
            valid &= target != self.ignore_index
        return valid

    def update(self, image_id: str, pred: np.ndarray, target: np.ndarray) -> None:
        if pred.shape != target.shape:
            raise ValueError(f"{image_id}: prediction shape {pred.shape} != label shape {target.shape}")
        pred = pred.astype(np.int64, copy=False)
        target = target.astype(np.int64, copy=False)
        valid = self._valid_mask(pred, target)

        per_cm = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        if valid.any():
            bins = target[valid] * self.num_classes + pred[valid]
            per_cm = np.bincount(
                bins, minlength=self.num_classes * self.num_classes
            ).reshape(self.num_classes, self.num_classes)
            self.confusion += per_cm

        per = compute_metrics_from_confusion(per_cm)
        present = per["target_pixels"] > 0
        present_no_bg = present.copy()
        if len(present_no_bg):
            present_no_bg[0] = False
        self.per_image_rows.append(
            {
                "image_id": image_id,
                "valid_pixels": int(valid.sum()),
                "num_present_classes": int(present.sum()),
                "per_image_mIoU": nanmean_or_nan(per["iou"][present]),
                "per_image_mIoU_no_background": nanmean_or_nan(per["iou"][present_no_bg]),
            }
        )

    def summary(self) -> dict[str, object]:
        metrics = compute_metrics_from_confusion(self.confusion)
        no_bg = np.ones(self.num_classes, dtype=bool)
        if self.num_classes:
            no_bg[0] = False
        per_vals = np.array([float(row["per_image_mIoU"]) for row in self.per_image_rows], dtype=float)
        per_vals_no_bg = np.array(
            [float(row["per_image_mIoU_no_background"]) for row in self.per_image_rows], dtype=float
        )
        return {
            "mIoU": nanmean_or_nan(metrics["iou"]),
            "mIoU_no_background": nanmean_or_nan(metrics["iou"][no_bg]),
            "FWIoU": metrics["FWIoU"],
            "per_image_mIoU_mean": nanmean_or_nan(per_vals),
            "per_image_mIoU_std": nanstd_or_nan(per_vals),
            "per_image_mIoU_no_background_mean": nanmean_or_nan(per_vals_no_bg),
            "per_image_mIoU_no_background_std": nanstd_or_nan(per_vals_no_bg),
            "num_images": len(self.per_image_rows),
            "num_classes": self.num_classes,
            "valid_pixels": int(metrics["target_pixels"].sum()),
        }


def nanmean_or_nan(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else float("nan")


def nanstd_or_nan(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(values.std(ddof=0)) if values.size else float("nan")


def compute_metrics_from_confusion(cm: np.ndarray) -> dict[str, np.ndarray | float]:
    cm = cm.astype(np.float64, copy=False)
    tp = np.diag(cm)
    pred_pixels = cm.sum(axis=0)
    target_pixels = cm.sum(axis=1)
    union = target_pixels + pred_pixels - tp
    iou = np.full(cm.shape[0], np.nan, dtype=np.float64)
    np.divide(tp, union, out=iou, where=union > 0)
    total_target = target_pixels.sum()
    fwiou = float(np.nansum(target_pixels * iou) / total_target) if total_target > 0 else float("nan")
    return {
        "intersection": tp,
        "union": union,
        "target_pixels": target_pixels,
        "predicted_pixels": pred_pixels,
        "iou": iou,
        "FWIoU": fwiou,
    }


def classwise_iou_rows(class_names: list[str], iou: np.ndarray) -> list[dict[str, object]]:
    return [
        {"class_id": class_id, "class_name": name, "IoU": float(iou[class_id])}
        for class_id, name in enumerate(class_names)
    ]


def metric_string(value: object) -> str:
    value = float(value)
    return f"{value:.6f}" if np.isfinite(value) else "nan"


def confusion_column_name(class_id: int, class_name: str) -> str:
    slug = "".join(ch if ch.isalnum() else "_" for ch in class_name).strip("_").lower()
    while "__" in slug:
        slug = slug.replace("__", "_")
    return f"pred_{class_id}_{slug}" if slug else f"pred_{class_id}"


def natural_key(value: str) -> tuple[int, object]:
    return (0, int(value)) if value.isdigit() else (1, value)


def collect_samples(dataset_root: Path, dataset: str, eval_split: str) -> list[Sample]:
    split_key = eval_split.lower()
    if split_key in {"val", "validation"}:
        split_dir = "Validation"
        prefix = "val"
    elif split_key == "test":
        split_dir = "Test"
        prefix = "test"
    else:
        raise ValueError(f"Unsupported eval split: {eval_split}")

    root = dataset_root / DATASET_INFO[dataset]["title"] / split_dir
    img_dir = root / f"{prefix}-org-img"
    label_dir = root / f"{prefix}-label-img"
    if not img_dir.is_dir() or not label_dir.is_dir():
        raise FileNotFoundError(f"Missing image/label directories: {img_dir} or {label_dir}")

    samples: list[Sample] = []
    for image_path in sorted(img_dir.iterdir(), key=lambda p: natural_key(p.stem)):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        label_path = label_dir / f"{image_path.stem}_lab.png"
        if not label_path.exists():
            raise FileNotFoundError(f"Missing label for {image_path.name}: {label_path}")
        samples.append(Sample(image_path.stem, image_path, label_path))
    if not samples:
        raise RuntimeError(f"No evaluation samples found in {img_dir}")
    return samples


def resize_pair(image: Image.Image, label: Image.Image, eval_size: int | None) -> tuple[Image.Image, Image.Image]:
    if eval_size is None:
        return image, label
    size = (int(eval_size), int(eval_size))
    return image.resize(size, Image.BILINEAR), label.resize(size, Image.NEAREST)


def save_index_png(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8), mode="L").save(path)


def save_color_png(mask: np.ndarray, palette: list[tuple[int, int, int]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_id, color in enumerate(palette):
        rgb[mask == class_id] = color
    Image.fromarray(rgb, mode="RGB").save(path)


def write_class_csv(path: Path, class_names: list[str], metrics: dict[str, np.ndarray | float]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "class_id",
                "class_name",
                "intersection",
                "union",
                "target_pixels",
                "predicted_pixels",
                "frequency",
                "IoU",
            ],
        )
        writer.writeheader()
        target_total = float(np.asarray(metrics["target_pixels"]).sum())
        for class_id, name in enumerate(class_names):
            target = float(metrics["target_pixels"][class_id])
            writer.writerow(
                {
                    "class_id": class_id,
                    "class_name": name,
                    "intersection": int(metrics["intersection"][class_id]),
                    "union": int(metrics["union"][class_id]),
                    "target_pixels": int(target),
                    "predicted_pixels": int(metrics["predicted_pixels"][class_id]),
                    "frequency": target / target_total if target_total > 0 else float("nan"),
                    "IoU": float(metrics["iou"][class_id]),
                }
            )


def write_confusion_matrix_csv(path: Path, class_names: list[str], confusion: np.ndarray) -> None:
    pred_columns = [confusion_column_name(class_id, name) for class_id, name in enumerate(class_names)]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["target_class_id", "target_class_name", "target_total", *pred_columns],
        )
        writer.writeheader()
        for target_class_id, target_name in enumerate(class_names):
            row = {
                "target_class_id": target_class_id,
                "target_class_name": target_name,
                "target_total": int(confusion[target_class_id].sum()),
            }
            for pred_class_id, column in enumerate(pred_columns):
                row[column] = int(confusion[target_class_id, pred_class_id])
            writer.writerow(row)


def write_per_image_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_id",
                "valid_pixels",
                "num_present_classes",
                "per_image_mIoU",
                "per_image_mIoU_no_background",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def print_result_report(row: dict[str, object]) -> None:
    print(
        f"{row['run']} {row['method']} {row['dataset']}_{row['split']}: "
        f"mIoU={metric_string(row['mIoU'])}, "
        f"mIoU_no_background={metric_string(row['mIoU_no_background'])}, "
        f"FWIoU={metric_string(row['FWIoU'])}, "
        f"per-image mIoU={metric_string(row['per_image_mIoU_mean'])}, "
        f"per-image mIoU_no_background={metric_string(row['per_image_mIoU_no_background_mean'])}"
    )
    print("Classwise IoU:")
    for class_row in row.get("classwise_IoU", []):
        print(f"  {int(class_row['class_id']):02d} {class_row['class_name']}: {metric_string(class_row['IoU'])}")
    print(f"Confusion matrix: {row['confusion_matrix_csv']} (rows=target, columns=prediction)")


@contextmanager
def method_context(method_root: Path):
    old_cwd = Path.cwd()
    old_sys_path = list(sys.path)
    os.chdir(method_root)
    sys.path.insert(0, str(method_root))
    try:
        yield
    finally:
        os.chdir(old_cwd)
        sys.path[:] = old_sys_path


def strip_module_prefix(state: dict) -> dict:
    return {key.replace("module.", "", 1): value for key, value in state.items()}


class LogitAdapter:
    def __init__(
        self,
        model,
        preprocess: Callable[[Image.Image], "object"],
        num_classes: int,
        device: "object",
        amp: bool,
        amp_dtype: "object",
        align_corners: bool,
        tile_size: int | None,
        tile_stride: int | None,
    ):
        self.model = model
        self.preprocess = preprocess
        self.num_classes = num_classes
        self.device = device
        self.amp = amp
        self.amp_dtype = amp_dtype
        self.align_corners = align_corners
        self.tile_size = tile_size
        self.tile_stride = tile_stride or tile_size

    def _extract_logits(self, output):
        if isinstance(output, dict):
            output = output.get("pred", output.get("out", output))
        if isinstance(output, (tuple, list)):
            output = output[0]
        return output

    def _predict_logits_one(self, image: Image.Image):
        import torch
        import torch.nn.functional as F

        tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            with torch.cuda.amp.autocast(
                enabled=self.amp and getattr(self.device, "type", "cpu") == "cuda",
                dtype=self.amp_dtype,
            ):
                logits = self._extract_logits(self.model(tensor))
        logits = F.interpolate(
            logits.float(),
            size=(image.height, image.width),
            mode="bilinear",
            align_corners=self.align_corners,
        )
        return logits.squeeze(0).cpu().numpy()

    def predict(self, image: Image.Image) -> np.ndarray:
        if (
            self.tile_size is None
            or image.width <= self.tile_size
            and image.height <= self.tile_size
        ):
            logits = self._predict_logits_one(image)
            return logits.argmax(axis=0).astype(np.uint8)
        return self._predict_sliding(image)

    def _predict_sliding(self, image: Image.Image) -> np.ndarray:
        tile = int(self.tile_size)
        stride = int(self.tile_stride or tile)
        height, width = image.height, image.width
        logits_sum = np.zeros((self.num_classes, height, width), dtype=np.float32)
        counts = np.zeros((height, width), dtype=np.float32)
        y_starts = sliding_starts(height, tile, stride)
        x_starts = sliding_starts(width, tile, stride)
        for y0 in y_starts:
            for x0 in x_starts:
                x1 = min(x0 + tile, width)
                y1 = min(y0 + tile, height)
                crop = image.crop((x0, y0, x1, y1))
                logits = self._predict_logits_one(crop)
                logits_sum[:, y0:y1, x0:x1] += logits[:, : y1 - y0, : x1 - x0]
                counts[y0:y1, x0:x1] += 1.0
        logits_sum /= np.maximum(counts[None, :, :], 1.0)
        return logits_sum.argmax(axis=0).astype(np.uint8)


def sliding_starts(length: int, tile: int, stride: int) -> list[int]:
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


def imagenet_preprocess(image: Image.Image):
    import torch

    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr.transpose(2, 0, 1))
    mean = tensor.new_tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = tensor.new_tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (tensor - mean) / std


def rgb255_preprocess(image: Image.Image, mean: list[float], std: list[float]):
    import torch

    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    tensor = torch.from_numpy(arr.transpose(2, 0, 1))
    mean_t = tensor.new_tensor(mean).view(3, 1, 1)
    std_t = tensor.new_tensor(std).view(3, 1, 1)
    return (tensor - mean_t) / std_t


def classmix_preprocess(image: Image.Image):
    import torch

    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    tensor = torch.from_numpy(arr.transpose(2, 0, 1))
    # Equivalent to cv2 BGR input, BGR mean subtraction, then channel reversal.
    mean = tensor.new_tensor([122.67891434, 116.66876762, 104.00698793]).view(3, 1, 1)
    return tensor - mean


def load_adapter(task: Task, args: argparse.Namespace) -> LogitAdapter:
    import torch

    info = DATASET_INFO[task.dataset]
    num_classes = info["num_classes"]
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    amp_dtype = torch.bfloat16 if str(args.amp_dtype).lower() in {"bf16", "bfloat16"} else torch.float16

    method_root = PROJECT_ROOT / METHOD_DIRS[task.method]
    with method_context(method_root):
        if task.method == "unimatch":
            import yaml
            from model.semseg.deeplabv3plus import DeepLabV3Plus

            cfg_path = task.config or (method_root / "configs" / f"{task.dataset}.yaml")
            cfg = yaml.load(open(cfg_path, "r", encoding="utf-8"), Loader=yaml.Loader)
            cfg["nclass"] = num_classes
            model = DeepLabV3Plus(cfg)
            ckpt = torch.load(task.checkpoint, map_location="cpu")
            state = ckpt.get("model", ckpt)
            model.load_state_dict(strip_module_prefix(state), strict=True)
            preprocess = imagenet_preprocess
            align_corners = True

        elif task.method == "classmix":
            from model.deeplabv2 import Res_Deeplab

            model = Res_Deeplab(num_classes=num_classes)
            ckpt = torch.load(task.checkpoint, map_location="cpu")
            state = ckpt.get("model", ckpt.get("state_dict", ckpt))
            model.load_state_dict(strip_module_prefix(state), strict=True)
            preprocess = classmix_preprocess
            align_corners = True

        elif task.method == "reco":
            import torchvision.models as models
            from network.deeplabv2 import DeepLabv2
            from network.deeplabv3.deeplabv3 import DeepLabv3Plus

            if args.reco_backbone == "deeplabv2":
                model = DeepLabv2(models.resnet101(pretrained=False), num_classes=num_classes, output_dim=args.reco_output_dim)
                align_corners = True
            else:
                model = DeepLabv3Plus(
                    models.resnet101(pretrained=False),
                    num_classes=num_classes,
                    output_dim=args.reco_output_dim,
                )
                align_corners = True
            ckpt = torch.load(task.checkpoint, map_location="cpu")
            state = ckpt.get("state_dict", ckpt.get("model", ckpt))
            model.load_state_dict(strip_module_prefix(state), strict=True)
            preprocess = imagenet_preprocess

        elif task.method == "s4mc":
            import yaml
            from s4mc_utils.models.model_helper import ModelBuilder
            from s4mc_utils.utils.utils import convert_state_dict

            if task.config is None:
                raise FileNotFoundError(f"S4MC requires a config for {task.run_dir}")
            cfg = yaml.load(open(task.config, "r", encoding="utf-8"), Loader=yaml.Loader)
            model = ModelBuilder(cfg["net"])
            ckpt = torch.load(task.checkpoint, map_location="cpu")
            key = "teacher_state" if "teacher_state" in ckpt else "model_state"
            state = convert_state_dict(ckpt[key])
            model.load_state_dict(state, strict=False)
            mean = cfg["dataset"].get("mean", [123.675, 116.28, 103.53])
            std = cfg["dataset"].get("std", [58.395, 57.12, 57.375])
            preprocess = lambda img: rgb255_preprocess(img, mean, std)
            align_corners = True

        elif task.method == "dual_teacher":
            from seg_core.model import MiT_SegFormer

            backbone = args.dual_teacher_backbone
            model = MiT_SegFormer(backbone=backbone, num_classes=num_classes, embedding_dim=256, pretrained=False)
            state = torch.load(task.checkpoint, map_location="cpu")
            model.load_state_dict(strip_module_prefix(state), strict=True)
            preprocess = lambda img: rgb255_preprocess(
                img, [123.675, 116.28, 103.53], [58.395, 57.12, 57.375]
            )
            align_corners = False
        else:
            raise ValueError(f"Unsupported method: {task.method}")

    model.to(device)
    model.eval()
    return LogitAdapter(
        model=model,
        preprocess=preprocess,
        num_classes=num_classes,
        device=device,
        amp=args.amp,
        amp_dtype=amp_dtype,
        align_corners=align_corners,
        tile_size=args.tile_size,
        tile_stride=args.tile_stride,
    )


def evaluate_task(task: Task, args: argparse.Namespace) -> dict[str, object]:
    info = DATASET_INFO[task.dataset]
    task.output_dir.mkdir(parents=True, exist_ok=True)
    index_dir = task.output_dir / "predictions_index"
    color_dir = task.output_dir / "predictions_color"

    samples = collect_samples(args.dataset_root, task.dataset, args.eval_split)
    if args.limit:
        samples = samples[: args.limit]

    adapter = load_adapter(task, args)
    accumulator = MetricAccumulator(info["num_classes"], args.ignore_index)

    for sample in tqdm(samples, desc=f"{task.method}/{task.dataset}_{task.split}", unit="img"):
        image = Image.open(sample.image_path).convert("RGB")
        label = Image.open(sample.label_path)
        if label.mode != "L":
            label = label.convert("L")
        image_eval, label_eval = resize_pair(image, label, args.eval_size)
        pred = adapter.predict(image_eval)
        target = np.asarray(label_eval, dtype=np.int64)
        accumulator.update(sample.image_id, pred, target)
        save_index_png(pred, index_dir / f"{sample.image_id}.png")
        save_color_png(pred, info["palette"], color_dir / f"{sample.image_id}.png")
        if args.save_original_size and args.eval_size is not None:
            original_pred = Image.fromarray(pred, mode="L").resize(image.size, Image.NEAREST)
            original_np = np.asarray(original_pred)
            save_index_png(original_np, task.output_dir / "predictions_index_original_size" / f"{sample.image_id}.png")
            save_color_png(
                original_np,
                info["palette"],
                task.output_dir / "predictions_color_original_size" / f"{sample.image_id}.png",
            )

    summary = accumulator.summary()
    class_metrics = compute_metrics_from_confusion(accumulator.confusion)
    classwise_rows = classwise_iou_rows(info["classes"], class_metrics["iou"])
    class_csv_path = task.output_dir / "classwise_metrics.csv"
    per_image_csv_path = task.output_dir / "per_image_metrics.csv"
    confusion_csv_path = task.output_dir / "confusion_matrix.csv"
    confusion_npy_path = task.output_dir / "confusion_matrix.npy"

    write_class_csv(class_csv_path, info["classes"], class_metrics)
    write_per_image_csv(per_image_csv_path, accumulator.per_image_rows)
    write_confusion_matrix_csv(confusion_csv_path, info["classes"], accumulator.confusion)
    np.save(confusion_npy_path, accumulator.confusion)

    result = {
        "run": task.run_dir.name,
        "method": task.method,
        "dataset": task.dataset,
        "split": task.split,
        "eval_split": args.eval_split,
        "eval_size": args.eval_size,
        "checkpoint": str(task.checkpoint),
        "config": str(task.config) if task.config else "",
        "output_dir": str(task.output_dir),
        "classwise_IoU": classwise_rows,
        "classwise_metrics_csv": str(class_csv_path),
        "per_image_metrics_csv": str(per_image_csv_path),
        "confusion_matrix_csv": str(confusion_csv_path),
        "confusion_matrix_npy": str(confusion_npy_path),
        **summary,
    }
    (task.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, allow_nan=True) + "\n", encoding="utf-8"
    )
    return result


def find_one(patterns: list[str], root: Path) -> Path | None:
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(root.glob(pattern))
    matches = [p for p in matches if p.exists()]
    if not matches:
        return None
    return sorted(matches, key=lambda p: str(p))[-1]


def discover_checkpoint(run_dir: Path, method: str, dataset: str, split: str) -> Path | None:
    exp = run_dir / "exp" / method / f"{dataset}_{split}"
    if method == "unimatch":
        return find_one(["best.pth"], exp)
    if method == "dual_teacher":
        return find_one(["best_weights.pth"], exp)
    if method == "reco":
        return find_one(["model_weights/*.pth"], exp)
    if method == "classmix":
        return find_one(["*/best_model.pth", "best_model.pth"], exp)
    if method == "s4mc":
        return find_one([f"checkpoints/s4mc_{dataset}_{split}/ckpt_best.pth", "**/ckpt_best.pth"], exp)
    return None


def discover_config(run_dir: Path, method: str, dataset: str, split: str) -> Path | None:
    if method == "unimatch":
        return find_one([f"unimatch_{dataset}_{split}_*epoch.yaml"], run_dir / "configs" / "unimatch")
    if method == "s4mc":
        return find_one([f"s4mc_{dataset}_{split}_*epoch.yaml"], run_dir / "configs" / "s4mc")
    if method == "classmix":
        return find_one([f"classmix_{dataset}_{split}_*epoch.json"], run_dir / "configs" / "classmix")
    return None


def discover_run_dirs(args: argparse.Namespace) -> list[Path]:
    if args.run:
        out = []
        for item in args.run:
            path = Path(item).expanduser()
            if not path.is_absolute():
                path = args.benchmark_root / path
            out.append(path.resolve())
        return out
    return sorted(args.benchmark_root.glob("benchmark_*"), key=lambda p: p.name)


def default_output_dir(args: argparse.Namespace, run_dir: Path, method: str, dataset: str, split: str) -> Path:
    suffix = f"{args.eval_split}_{args.eval_size}"
    if args.output_root is not None:
        return args.output_root / run_dir.name / method / f"{dataset}_{split}" / suffix
    return run_dir / "final_evaluation" / method / f"{dataset}_{split}" / suffix


def build_tasks(args: argparse.Namespace) -> list[Task]:
    if args.task_method:
        return [
            Task(
                run_dir=args.task_run_dir.resolve(),
                method=args.task_method,
                dataset=args.task_dataset,
                split=args.task_split,
                checkpoint=args.task_checkpoint.resolve(),
                config=args.task_config.resolve() if args.task_config else None,
                output_dir=args.task_output_dir.resolve(),
            )
        ]

    tasks: list[Task] = []
    for run_dir in discover_run_dirs(args):
        if not run_dir.is_dir():
            print(f"Skipping missing run dir: {run_dir}", file=sys.stderr)
            continue
        run_splits = {split for split in SPLITS if f"benchmark_{split}_" in run_dir.name}
        for method in args.methods:
            for dataset in args.datasets:
                for split in args.splits:
                    if run_splits and split not in run_splits:
                        continue
                    checkpoint = discover_checkpoint(run_dir, method, dataset, split)
                    if checkpoint is None:
                        print(f"Skipping missing checkpoint: {run_dir.name} {method} {dataset}_{split}", file=sys.stderr)
                        continue
                    tasks.append(
                        Task(
                            run_dir=run_dir,
                            method=method,
                            dataset=dataset,
                            split=split,
                            checkpoint=checkpoint.resolve(),
                            config=(discover_config(run_dir, method, dataset, split) or None),
                            output_dir=default_output_dir(args, run_dir, method, dataset, split).resolve(),
                        )
                    )
    return tasks


def env_python(method: str, env_root: Path) -> Path:
    return env_root / method / "bin" / "python"


def run_task_subprocess(task: Task, args: argparse.Namespace) -> dict[str, object]:
    python = env_python(task.method, args.env_root)
    if not python.exists():
        python = Path(sys.executable)

    cmd = [
        str(python),
        str(Path(__file__).resolve()),
        "--no-dispatch",
        "--task-run-dir",
        str(task.run_dir),
        "--task-method",
        task.method,
        "--task-dataset",
        task.dataset,
        "--task-split",
        task.split,
        "--task-checkpoint",
        str(task.checkpoint),
        "--task-output-dir",
        str(task.output_dir),
        "--dataset-root",
        str(args.dataset_root),
        "--eval-split",
        args.eval_split,
        "--eval-size",
        str(args.eval_size),
        "--ignore-index",
        str(args.ignore_index if args.ignore_index is not None else "none"),
        "--amp-dtype",
        args.amp_dtype,
        "--reco-backbone",
        args.reco_backbone,
        "--reco-output-dim",
        str(args.reco_output_dim),
        "--dual-teacher-backbone",
        args.dual_teacher_backbone,
    ]
    if task.config:
        cmd.extend(["--task-config", str(task.config)])
    if args.device:
        cmd.extend(["--device", args.device])
    if args.amp:
        cmd.append("--amp")
    if args.limit:
        cmd.extend(["--limit", str(args.limit)])
    if args.save_original_size:
        cmd.append("--save-original-size")
    if args.tile_size:
        cmd.extend(["--tile-size", str(args.tile_size)])
    if args.tile_stride:
        cmd.extend(["--tile-stride", str(args.tile_stride)])

    task.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = task.output_dir / "eval.log"
    env = os.environ.copy()
    if task.method == "dual_teacher":
        env["DIST_BACKEND"] = env.get("DIST_BACKEND", "gloo")
        nvrtc_lib = (
            args.env_root
            / "dual_teacher"
            / "lib"
            / "python3.8"
            / "site-packages"
            / "nvidia"
            / "cuda_nvrtc"
            / "lib"
        )
        if nvrtc_lib.exists():
            old_ld = env.get("LD_LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = f"{nvrtc_lib}:{old_ld}" if old_ld else str(nvrtc_lib)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("COMMAND: " + " ".join(cmd) + "\n\n")
        log.flush()
        completed = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, text=True, env=env)
    if completed.returncode != 0:
        raise RuntimeError(f"Task failed with code {completed.returncode}; see {log_path}")
    return json.loads((task.output_dir / "summary.json").read_text(encoding="utf-8"))


def write_summary_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "run",
        "method",
        "dataset",
        "split",
        "eval_split",
        "eval_size",
        "checkpoint",
        "config",
        "output_dir",
        "classwise_metrics_csv",
        "per_image_metrics_csv",
        "confusion_matrix_csv",
        "confusion_matrix_npy",
        "mIoU",
        "mIoU_no_background",
        "FWIoU",
        "per_image_mIoU_mean",
        "per_image_mIoU_std",
        "per_image_mIoU_no_background_mean",
        "per_image_mIoU_no_background_std",
        "num_images",
        "num_classes",
        "valid_pixels",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def completed_summary_for_task(task: Task) -> dict[str, object] | None:
    summary_path = task.output_dir / "summary.json"
    if not summary_path.exists():
        return None
    return json.loads(summary_path.read_text(encoding="utf-8"))


def parse_eval_size(value: str) -> int | None:
    value = str(value).strip().lower()
    if value in {"original", "none", "full"}:
        raise argparse.ArgumentTypeError("Original-size final evaluation has been retired; use --eval-size 750.")
    size = int(value)
    if size != 750:
        raise argparse.ArgumentTypeError("Final evaluation now only supports --eval-size 750.")
    return size


def parse_ignore_index(value: str) -> int | None:
    value = str(value).strip().lower()
    if value in {"none", "null", "no"}:
        return None
    return int(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--run", action="append", help="Benchmark run directory name/path. Repeat for selected runs.")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument(
        "--eval-split",
        choices=["val", "validation", "test"],
        default="test",
        help="Dataset split to evaluate. Default: test.",
    )
    parser.add_argument(
        "--eval-size",
        type=parse_eval_size,
        default=750,
        help="Final square evaluation size. Only 750 is supported.",
    )
    parser.add_argument(
        "--ignore-index",
        type=parse_ignore_index,
        default=None,
        help="Void label to ignore. Use 'none' to include class 0/background. Default: none.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--amp-dtype", default="bf16", choices=["bf16", "bfloat16", "fp16", "float16"])
    parser.add_argument("--env-root", type=Path, default=DEFAULT_ENV_ROOT)
    parser.add_argument("--no-dispatch", action="store_true", help="Run in the current Python process.")
    parser.add_argument("--dry-run", action="store_true", help="List discovered tasks, then exit without inference.")
    parser.add_argument("--save-original-size", action="store_true")
    parser.add_argument("--tile-size", type=int, default=None, help="Use sliding-window inference with the given tile size.")
    parser.add_argument("--tile-stride", type=int, default=None)
    parser.add_argument("--reco-backbone", choices=["deeplabv3p", "deeplabv2"], default="deeplabv3p")
    parser.add_argument("--reco-output-dim", type=int, default=256)
    parser.add_argument("--dual-teacher-backbone", default="mit_b1")
    parser.add_argument("--skip-completed", action="store_true", help="Skip tasks whose output summary.json already exists.")

    parser.add_argument("--task-run-dir", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--task-method", choices=METHODS, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--task-dataset", choices=DATASETS, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--task-split", choices=SPLITS, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--task-checkpoint", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--task-config", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--task-output-dir", type=Path, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.benchmark_root = args.benchmark_root.expanduser().resolve()
    args.dataset_root = args.dataset_root.expanduser().resolve()
    args.env_root = args.env_root.expanduser().resolve()
    if args.output_root is not None:
        args.output_root = args.output_root.expanduser().resolve()

    tasks = build_tasks(args)
    skipped_rows: list[dict[str, object]] = []
    if args.skip_completed and not args.task_output_dir:
        pending_tasks = []
        for task in tasks:
            completed = completed_summary_for_task(task)
            if completed is None:
                pending_tasks.append(task)
            else:
                skipped_rows.append(completed)
                print(f"Skipping completed task: {task.run_dir.name} {task.method} {task.dataset}_{task.split}")
        tasks = pending_tasks
    if not tasks and not skipped_rows:
        raise SystemExit("No semi-supervised evaluation tasks found.")
    if args.dry_run:
        for task in tasks:
            print(
                f"{task.run_dir.name} {task.method} {task.dataset}_{task.split} "
                f"ckpt={task.checkpoint} config={task.config} out={task.output_dir}"
            )
        return 0

    rows: list[dict[str, object]] = list(skipped_rows)
    for task in tasks:
        if args.no_dispatch:
            row = evaluate_task(task, args)
        else:
            row = run_task_subprocess(task, args)
        rows.append(row)
        print_result_report(row)

    if args.task_output_dir:
        return 0

    if args.output_root is not None:
        summary_path = args.output_root / "summary.csv"
    elif len({row["run"] for row in rows}) == 1:
        summary_path = Path(rows[0]["output_dir"]).parents[2] / "summary.csv"
    else:
        summary_path = args.benchmark_root / "final_evaluation_summary.csv"
    write_summary_csv(summary_path, rows)
    print(f"\nWrote summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
