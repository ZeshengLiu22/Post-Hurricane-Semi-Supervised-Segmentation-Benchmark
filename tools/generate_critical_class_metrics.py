#!/usr/bin/env python3
"""Generate critical-class mIoU/recall tables from final evaluation outputs.

This script intentionally leaves the main final evaluation files untouched. It
prefers saved confusion matrices from ``evaluate_semi_supervised_final.py`` and
falls back to recomputing the same confusion matrix from exported index
predictions only when a matrix is missing.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from evaluate_semi_supervised_final import (
    DATASET_INFO,
    MetricAccumulator,
    collect_samples,
    compute_metrics_from_confusion,
    nanmean_or_nan,
    resize_pair,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_ROOT = PROJECT_ROOT.parent / "final_results" / "semi_supervised" / "test_750"
DEFAULT_SUPERVISED_INPUT_ROOT = PROJECT_ROOT.parent / "final_results" / "fully_supervised" / "test_750"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results"

DATASET_ORDER = ("floodnet", "rescuenet")
SPLIT_ORDER = ("12_5", "25", "50", "full")
METHOD_ORDER = (
    "mean_teacher",
    "unimatch",
    "classmix",
    "reco",
    "s4mc",
    "dual_teacher",
    "segformer",
    "mask2former",
)

RATIO_LABELS = {
    "12_5": "12.5%",
    "25": "25%",
    "50": "50%",
    "full": "100%",
}

METHOD_LABELS = {
    "mean_teacher": "Mean Teacher",
    "unimatch": "UniMatch",
    "classmix": "ClassMix",
    "reco": "ReCo",
    "s4mc": "S4MC",
    "dual_teacher": "Dual Teacher",
    "segformer": "SegFormer",
    "mask2former": "Mask2Former",
}

CRITICAL_CLASS_ALIASES = {
    "floodnet": [
        ("Building Flooded", ("building flooded", "building-flooded")),
        ("Building Non-Flooded", ("building non-flooded", "building-non-flooded")),
        ("Road Flooded", ("road flooded", "road-flooded")),
        ("Road Non-Flooded", ("road non-flooded", "road-non-flooded")),
    ],
    "rescuenet": [
        ("Building No Damage", ("building no damage", "building-no-damage", "building_no_damage")),
        (
            "Building Minor Damage",
            ("building minor damage", "building medium damage", "building-minor-damage", "building-medium-damage"),
        ),
        ("Building Major Damage", ("building major damage", "building-major-damage", "building_major_damage")),
        (
            "Building Destroyed",
            (
                "building destroyed",
                "building total destruction",
                "building-total-destruction",
                "building_total_destruction",
            ),
        ),
        ("Road Clear", ("road clear", "road-clear")),
        ("Road Blocked", ("road blocked", "road-blocked")),
    ],
}

TABLE_COLUMNS = [
    "Dataset",
    "Labeled Ratio",
    "Method",
    "mIoU",
    "Critical-class mIoU",
    "Critical-class Recall",
]

LATEX_CAPTION = (
    "All-class mIoU, critical-class mIoU, and recall on disaster-relevant categories "
    "for FloodNet and RescueNet."
)


@dataclass(frozen=True)
class ResultDir:
    path: Path
    method: str
    dataset: str
    split: str


def normalize_class_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()


def critical_indices(dataset: str) -> list[int]:
    class_names = DATASET_INFO[dataset]["classes"]
    normalized = {normalize_class_name(name): idx for idx, name in enumerate(class_names)}
    indices: list[int] = []
    missing: list[str] = []
    for display_name, aliases in CRITICAL_CLASS_ALIASES[dataset]:
        index = None
        for alias in aliases:
            index = normalized.get(normalize_class_name(alias))
            if index is not None:
                break
        if index is None:
            missing.append(display_name)
        else:
            indices.append(index)
    if missing:
        raise ValueError(f"Missing critical class definitions for {DATASET_INFO[dataset]['title']}: {missing}")
    return indices


def parse_dataset_split(value: str) -> tuple[str, str] | None:
    match = re.fullmatch(r"(floodnet|rescuenet)_(12_5|25|50)", value)
    if not match:
        return None
    return match.group(1), match.group(2)


def parse_supervised_run(value: str) -> tuple[str, str, str] | None:
    match = re.fullmatch(r"(segformer|mask2former)_(floodnet|rescuenet)_(12_5|25|50|full)", value)
    if not match:
        return None
    return match.group(1), match.group(2), match.group(3)


def parse_result_dir(path: Path) -> ResultDir | None:
    dataset_split = path.parent.name
    parsed = parse_dataset_split(dataset_split)
    if parsed is not None:
        dataset, split = parsed
        method = path.parent.parent.name
        if method in METHOD_LABELS:
            return ResultDir(path=path, method=method, dataset=dataset, split=split)

    supervised = parse_supervised_run(dataset_split)
    if supervised is not None:
        method, dataset, split = supervised
        return ResultDir(path=path, method=method, dataset=dataset, split=split)

    return None


def discover_result_dirs(input_roots: list[Path]) -> list[ResultDir]:
    dirs = set()
    for input_root in input_roots:
        if not input_root.exists():
            continue
        dirs.update(path.parent for path in input_root.glob("**/summary.json"))
        dirs.update(path.parent for path in input_root.glob("**/confusion_matrix.npy"))

    parsed_dirs: dict[tuple[str, str, str], ResultDir] = {}
    duplicates: dict[tuple[str, str, str], list[Path]] = {}
    for path in sorted(dirs):
        result = parse_result_dir(path)
        if result is None:
            continue
        key = (result.dataset, result.split, result.method)
        if key in parsed_dirs:
            duplicates.setdefault(key, [parsed_dirs[key].path]).append(result.path)
            continue
        parsed_dirs[key] = result

    if duplicates:
        details = "; ".join(f"{key}: {paths}" for key, paths in sorted(duplicates.items()))
        raise RuntimeError(f"Duplicate result directories found. Narrow --input-root or remove duplicates: {details}")

    return sorted(
        parsed_dirs.values(),
        key=lambda item: (
            DATASET_ORDER.index(item.dataset),
            SPLIT_ORDER.index(item.split),
            METHOD_ORDER.index(item.method),
        ),
    )


def load_confusion_matrix(result: ResultDir, args: argparse.Namespace) -> np.ndarray:
    confusion_path = result.path / "confusion_matrix.npy"
    if confusion_path.exists():
        return np.load(confusion_path)

    summary_path = result.path / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary_confusion = Path(str(summary.get("confusion_matrix_npy", ""))).expanduser()
        if summary_confusion.exists():
            return np.load(summary_confusion)

    return compute_confusion_from_predictions(result, args)


def compute_confusion_from_predictions(result: ResultDir, args: argparse.Namespace) -> np.ndarray:
    prediction_dir = result.path / "predictions_index"
    if not prediction_dir.is_dir():
        raise FileNotFoundError(
            f"No confusion matrix or predictions_index directory found for {result.path}"
        )

    info = DATASET_INFO[result.dataset]
    samples = collect_samples(args.dataset_root, result.dataset, args.eval_split)
    accumulator = MetricAccumulator(info["num_classes"], args.ignore_index)

    for sample in samples:
        pred_path = prediction_dir / f"{sample.image_id}.png"
        if not pred_path.exists():
            raise FileNotFoundError(f"Missing prediction mask for {sample.image_id}: {pred_path}")
        pred = np.asarray(Image.open(pred_path).convert("L"), dtype=np.int64)
        image = Image.open(sample.image_path).convert("RGB")
        label = Image.open(sample.label_path)
        if label.mode != "L":
            label = label.convert("L")
        _, label_eval = resize_pair(image, label, args.eval_size)
        target = np.asarray(label_eval, dtype=np.int64)
        accumulator.update(sample.image_id, pred, target)

    return accumulator.confusion


def all_class_miou(cm: np.ndarray) -> float:
    metrics = compute_metrics_from_confusion(cm)
    return nanmean_or_nan(np.asarray(metrics["iou"], dtype=float))


def critical_metrics(cm: np.ndarray, dataset: str) -> tuple[float, float]:
    metrics = compute_metrics_from_confusion(cm)
    indices = np.asarray(critical_indices(dataset), dtype=int)
    iou = np.asarray(metrics["iou"], dtype=float)
    target_pixels = np.asarray(metrics["target_pixels"], dtype=float)
    intersection = np.asarray(metrics["intersection"], dtype=float)
    recall = np.full_like(target_pixels, np.nan, dtype=float)
    np.divide(intersection, target_pixels, out=recall, where=target_pixels > 0)
    return nanmean_or_nan(iou[indices]), nanmean_or_nan(recall[indices])


def percent_string(value: float) -> str:
    return f"{value * 100.0:.2f}" if np.isfinite(value) else "nan"


def build_rows(result_dirs: list[ResultDir], args: argparse.Namespace) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for result in result_dirs:
        cm = load_confusion_matrix(result, args)
        expected_classes = int(DATASET_INFO[result.dataset]["num_classes"])
        if cm.shape != (expected_classes, expected_classes):
            raise ValueError(
                f"{result.path}: expected confusion matrix shape "
                f"{(expected_classes, expected_classes)}, got {cm.shape}"
            )
        miou = all_class_miou(cm)
        critical_miou, critical_recall = critical_metrics(cm, result.dataset)
        rows.append(
            {
                "Dataset": DATASET_INFO[result.dataset]["title"],
                "Labeled Ratio": RATIO_LABELS[result.split],
                "Method": METHOD_LABELS[result.method],
                "mIoU": percent_string(miou),
                "Critical-class mIoU": percent_string(critical_miou),
                "Critical-class Recall": percent_string(critical_recall),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TABLE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, str]]) -> str:
    lines = [
        "| Dataset | Labeled Ratio | Method | mIoU | Critical-class mIoU | Critical-class Recall |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {Dataset} | {Labeled Ratio} | {Method} | {mIoU} | {Critical-class mIoU} | {Critical-class Recall} |".format(
                **row
            )
        )
    return "\n".join(lines) + "\n"


def latex_escape(value: str) -> str:
    return value.replace("_", r"\_").replace("%", r"\%")


def latex_table(rows: list[dict[str, str]]) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{LATEX_CAPTION}}}",
        r"\begin{tabular}{ll l rrr}",
        r"\toprule",
        r"Dataset & Labeled Ratio & Method & mIoU & Critical-class mIoU & Critical-class Recall \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            " & ".join(
                latex_escape(row[column])
                for column in TABLE_COLUMNS
            )
            + r" \\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(rows: list[dict[str, str]], output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    write_csv(output_root / "critical_class_metrics.csv", rows)
    (output_root / "critical_class_metrics.md").write_text(markdown_table(rows), encoding="utf-8")
    (output_root / "critical_class_metrics.tex").write_text(latex_table(rows), encoding="utf-8")


def parse_ignore_index(value: str) -> int | None:
    value = str(value).strip().lower()
    if value in {"none", "null", "no"}:
        return None
    return int(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--supervised-input-root", type=Path, default=DEFAULT_SUPERVISED_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dataset-root", type=Path, default=PROJECT_ROOT.parent / "Dataset")
    parser.add_argument("--eval-split", choices=["val", "validation", "test"], default="test")
    parser.add_argument("--eval-size", type=int, default=750)
    parser.add_argument("--ignore-index", type=parse_ignore_index, default=None)
    parser.add_argument("--print-indices", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.input_root = args.input_root.expanduser().resolve()
    args.supervised_input_root = args.supervised_input_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.dataset_root = args.dataset_root.expanduser().resolve()

    if args.print_indices:
        for dataset in DATASET_ORDER:
            names = DATASET_INFO[dataset]["classes"]
            indices = critical_indices(dataset)
            labels = ", ".join(f"{idx}:{names[idx]}" for idx in indices)
            print(f"{DATASET_INFO[dataset]['title']}: [{', '.join(map(str, indices))}] ({labels})")

    result_dirs = discover_result_dirs([args.input_root, args.supervised_input_root])
    if not result_dirs:
        raise SystemExit(
            "No final evaluation result directories found under "
            f"{args.input_root} or {args.supervised_input_root}"
        )

    rows = build_rows(result_dirs, args)
    write_outputs(rows, args.output_root)

    print(f"Wrote {len(rows)} rows to {args.output_root / 'critical_class_metrics.csv'}")
    print(f"Wrote {args.output_root / 'critical_class_metrics.md'}")
    print(f"Wrote {args.output_root / 'critical_class_metrics.tex'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
