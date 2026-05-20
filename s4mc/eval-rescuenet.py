import os
import yaml
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
from argparse import ArgumentParser
from tqdm import tqdm

from s4mc_utils.models.model_helper import ModelBuilder
from s4mc_utils.dataset.builder import get_loader
from s4mc_utils.utils.utils import convert_state_dict


def nanmean_or_nan(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else float("nan")


def nanstd_or_nan(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(values.std(ddof=0)) if values.size else float("nan")


def metric_percent(value):
    value = float(value)
    return f"{value * 100:.2f}%" if np.isfinite(value) else "nan"


def confusion_matrix_from_arrays(pred, target, num_classes, ignore_index=None):
    pred = np.asarray(pred).reshape(-1).astype(np.int64, copy=False)
    target = np.asarray(target).reshape(-1).astype(np.int64, copy=False)
    valid = (target >= 0) & (target < num_classes)
    valid &= (pred >= 0) & (pred < num_classes)
    if ignore_index is not None:
        valid &= target != ignore_index
    if not np.any(valid):
        return np.zeros((num_classes, num_classes), dtype=np.int64)
    bins = target[valid] * num_classes + pred[valid]
    return np.bincount(bins, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


def compute_metrics_from_confusion(confusion):
    confusion = confusion.astype(np.float64, copy=False)
    intersection = np.diag(confusion)
    target_pixels = confusion.sum(axis=1)
    predicted_pixels = confusion.sum(axis=0)
    union = target_pixels + predicted_pixels - intersection
    iou = np.full(confusion.shape[0], np.nan, dtype=np.float64)
    np.divide(intersection, union, out=iou, where=union > 0)
    total_target = target_pixels.sum()
    fwiou = float(np.nansum(target_pixels * iou) / total_target) if total_target > 0 else float("nan")
    return {
        "intersection": intersection,
        "union": union,
        "target_pixels": target_pixels,
        "predicted_pixels": predicted_pixels,
        "iou": iou,
        "FWIoU": fwiou,
    }


def get_rescuenet_palette():
    return [
        0, 0, 0,         # unlabeled
        61, 230, 250,    # water
        180, 120, 120,   # building-no-damage
        235, 255, 7,     # building-medium-damage
        255, 184, 6,     # building-major-damage
        255, 0, 0,       # building-total-destruction
        255, 0, 245,     # vehicle
        140, 140, 140,   # road-clear
        160, 150, 20,    # road-blocked
        4, 250, 7,       # tree
        255, 235, 0      # pool
    ]


def save_prediction(mask, name, color_folder, palette):
    os.makedirs(color_folder, exist_ok=True)

    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape: {mask.shape}")

    color_mask = Image.fromarray(mask.astype(np.uint8), mode="P")
    color_mask.putpalette(palette)
    color_mask.save(os.path.join(color_folder, name + ".png"))


@torch.no_grad()
def evaluate(model, loader, cfg, save_dir=None):
    model.eval()
    num_classes = cfg["net"]["num_classes"]
    ignore_label = cfg["dataset"].get("ignore_label")

    color_dir = os.path.join(save_dir, "color") if save_dir else None
    palette = get_rescuenet_palette() if save_dir else None

    total_confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    per_image_ious = []
    per_image_ious_no_background = []

    for _, (images, labels, names) in enumerate(tqdm(loader)):
        images = images.cuda()
        labels = labels.cuda()

        preds = model(images)["pred"]
        preds = F.interpolate(preds, size=labels.shape[1:], mode="bilinear", align_corners=True)
        preds = preds.argmax(dim=1).cpu().numpy()
        labels = labels.cpu().numpy()

        for i in range(preds.shape[0]):
            pred_i = preds[i]
            label_i = labels[i]

            if save_dir:
                save_prediction(pred_i, names[i], color_dir, palette)

            per_confusion = confusion_matrix_from_arrays(pred_i, label_i, num_classes, ignore_label)
            total_confusion += per_confusion

            per_metrics = compute_metrics_from_confusion(per_confusion)
            present = per_metrics["target_pixels"] > 0
            present_no_background = present.copy()
            if present_no_background.size:
                present_no_background[0] = False
            per_image_ious.append(nanmean_or_nan(per_metrics["iou"][present]))
            per_image_ious_no_background.append(nanmean_or_nan(per_metrics["iou"][present_no_background]))

    metrics = compute_metrics_from_confusion(total_confusion)
    iou = metrics["iou"]
    acc = np.full(num_classes, np.nan, dtype=np.float64)
    np.divide(metrics["intersection"], metrics["target_pixels"], out=acc, where=metrics["target_pixels"] > 0)
    no_background = np.ones(num_classes, dtype=bool)
    if num_classes:
        no_background[0] = False
    miou = nanmean_or_nan(iou)
    miou_no_background = nanmean_or_nan(iou[no_background])
    acc_avg = nanmean_or_nan(acc)

    print("\n--- Evaluation Results ---")
    for i in range(num_classes):
        print(f"Class {i:02d} | IoU: {metric_percent(iou[i])} | Acc: {metric_percent(acc[i])}")
    print(
        f"\nmIoU: {metric_percent(miou)}, "
        f"mIoU_no_background: {metric_percent(miou_no_background)}, "
        f"Acc: {metric_percent(acc_avg)}, "
        f"fwIoU: {metric_percent(metrics['FWIoU'])}"
    )

    print(
        f"\nPer-Image mIoU: mean = {metric_percent(nanmean_or_nan(per_image_ious))}, "
        f"std = {metric_percent(nanstd_or_nan(per_image_ious))}"
    )
    print(
        f"Per-Image mIoU_no_background: mean = {metric_percent(nanmean_or_nan(per_image_ious_no_background))}, "
        f"std = {metric_percent(nanstd_or_nan(per_image_ious_no_background))}"
    )

    return miou, acc_avg, metrics["FWIoU"]


def main():
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--save_dir", type=str, default=None, help="Optional output dir for visual results")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.load(f, Loader=yaml.Loader)

    model = ModelBuilder(cfg["net"])
    ckpt = torch.load(args.ckpt, map_location="cpu")
    key = "teacher_state" if "teacher_state" in ckpt else "model_state"
    model.load_state_dict(convert_state_dict(ckpt[key]), strict=False)
    model.cuda()

    _, _, val_loader = get_loader(cfg, seed=0)
    evaluate(model, val_loader, cfg, save_dir=args.save_dir)


if __name__ == "__main__":
    main()
