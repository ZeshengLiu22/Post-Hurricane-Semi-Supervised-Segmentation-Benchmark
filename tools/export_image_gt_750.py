#!/usr/bin/env python3
"""Export resized RGB images and ground-truth masks for final evaluation sets."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = PROJECT_ROOT.parent / "Dataset"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT.parent / "final_results" / "reference"
RESAMPLING = getattr(Image, "Resampling", Image)

DATASET_INFO = {
    "floodnet": {
        "title": "FloodNet",
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


def natural_key(path: Path) -> tuple[int, object]:
    return (0, int(path.stem)) if path.stem.isdigit() else (1, path.stem)


def colorize(mask: np.ndarray, palette: list[tuple[int, int, int]]) -> Image.Image:
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_id, color in enumerate(palette):
        rgb[mask == class_id] = color
    return Image.fromarray(rgb, mode="RGB")


def export_dataset(args: argparse.Namespace, dataset: str) -> int:
    info = DATASET_INFO[dataset]
    split_dir = "Validation" if args.split in {"val", "validation"} else "Test"
    prefix = "val" if split_dir == "Validation" else "test"
    root = args.dataset_root / info["title"] / split_dir
    image_dir = root / f"{prefix}-org-img"
    label_dir = root / f"{prefix}-label-img"
    out_dir = args.output_root / f"{prefix}_{args.size}" / dataset

    image_out = out_dir / "images_rgb"
    gt_index_out = out_dir / "ground_truth_index"
    gt_color_out = out_dir / "ground_truth_color"
    for path in (image_out, gt_index_out, gt_color_out):
        path.mkdir(parents=True, exist_ok=True)

    image_paths = [
        path
        for path in sorted(image_dir.iterdir(), key=natural_key)
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]
    if args.limit is not None:
        image_paths = image_paths[: args.limit]

    size = (args.size, args.size)
    written = 0
    for image_path in image_paths:
        image_id = image_path.stem
        label_path = label_dir / f"{image_id}_lab.png"
        if not label_path.exists():
            raise FileNotFoundError(f"Missing label for {image_path}: {label_path}")

        rgb_path = image_out / f"{image_id}.png"
        index_path = gt_index_out / f"{image_id}.png"
        color_path = gt_color_out / f"{image_id}.png"
        if not args.overwrite and rgb_path.exists() and index_path.exists() and color_path.exists():
            continue

        image = Image.open(image_path).convert("RGB").resize(size, RESAMPLING.BILINEAR)
        label = Image.open(label_path).convert("L").resize(size, RESAMPLING.NEAREST)
        mask = np.asarray(label, dtype=np.uint8)

        image.save(rgb_path)
        Image.fromarray(mask, mode="L").save(index_path)
        colorize(mask, info["palette"]).save(color_path)
        written += 1

    print(f"{dataset}: wrote {written} samples to {out_dir}")
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASET_INFO), default=sorted(DATASET_INFO))
    parser.add_argument("--split", choices=["test", "val", "validation"], default="test")
    parser.add_argument("--size", type=int, default=750)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.dataset_root = args.dataset_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    total = sum(export_dataset(args, dataset) for dataset in args.datasets)
    print(f"Done. Wrote {total} files per output kind.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
