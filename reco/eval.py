import argparse
import os

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision
from tqdm import tqdm

from build_data import transform, create_floodnet_label_colormap, create_rescuenet_label_colormap
from network.deeplabv2 import DeepLabv2
from network.deeplabv3.deeplabv3 import DeepLabv3Plus


DATASETS = {
    "floodnet": {
        "num_classes": 10,
        "colormap": create_floodnet_label_colormap,
    },
    "rescuenet": {
        "num_classes": 11,
        "colormap": create_rescuenet_label_colormap,
    },
}


def nanmean_or_nan(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else float("nan")


def nanstd_or_nan(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(values.std(ddof=0)) if values.size else float("nan")


def metric_string(value):
    value = float(value)
    return f"{value:.4f}" if np.isfinite(value) else "nan"


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


def build_model(backbone, num_classes, output_dim):
    encoder = torchvision.models.resnet101(pretrained=False)
    if backbone == "deeplabv2":
        return DeepLabv2(encoder, num_classes=num_classes, output_dim=output_dim)
    return DeepLabv3Plus(encoder, num_classes=num_classes, output_dim=output_dim)


def load_weights(model, model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    model.load_state_dict(checkpoint)


def color_map(label, colormap):
    return colormap[label]


def evaluate_dataset(dataset_name, model_path, backbone, output_dim, output_dir):
    info = DATASETS[dataset_name]
    num_classes = info["num_classes"]
    root_dir = f"dataset/{dataset_name}"
    val_list_file = os.path.join(root_dir, "val.txt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = build_model(backbone, num_classes, output_dim).to(device)
    load_weights(model, model_path, device)
    model.eval()

    colormap = info["colormap"]()
    im_size = [3000, 4000]

    with open(val_list_file) as f:
        idx_list = [line.strip() for line in f if line.strip()]

    save_root = os.path.join(output_dir, dataset_name)
    os.makedirs(save_root, exist_ok=True)

    total_confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    per_image_ious = []
    per_image_ious_no_background = []

    for idx in tqdm(idx_list):
        image = Image.open(f"{root_dir}/validationset/val-org-img/{idx}.jpg")
        label = Image.open(f"{root_dir}/validationset/val-label-img/{idx}_lab.png")
        image_tensor, label_tensor = transform(image, label, None, crop_size=im_size, scale_size=(1.0, 1.0), augmentation=False)

        image_tensor = image_tensor.to(device)
        label_tensor = label_tensor.to(device)
        image_w, image_h = image.size

        with torch.no_grad():
            logits, _ = model(image_tensor.unsqueeze(0))
            logits = F.interpolate(logits, size=label_tensor.shape[1:], mode="bilinear", align_corners=True)
            pred = torch.argmax(torch.softmax(logits, dim=1), dim=1)

        pred_np = pred.squeeze().cpu().numpy()
        label_np = label_tensor.squeeze().cpu().numpy()
        per_confusion = confusion_matrix_from_arrays(pred_np, label_np, num_classes)
        total_confusion += per_confusion
        per_metrics = compute_metrics_from_confusion(per_confusion)
        present = per_metrics["target_pixels"] > 0
        present_no_background = present.copy()
        if present_no_background.size:
            present_no_background[0] = False
        per_image_ious.append(nanmean_or_nan(per_metrics["iou"][present]))
        per_image_ious_no_background.append(nanmean_or_nan(per_metrics["iou"][present_no_background]))

        colored_mask = Image.fromarray(color_map(pred_np, colormap)[:image_h, :image_w])
        colored_mask.save(f"{save_root}/{idx}_mask_colored.png")

    metrics = compute_metrics_from_confusion(total_confusion)
    class_iou = metrics["iou"]
    no_background = np.ones(num_classes, dtype=bool)
    if num_classes:
        no_background[0] = False
    miou = nanmean_or_nan(class_iou)
    miou_no_background = nanmean_or_nan(class_iou[no_background])

    print(f"\n==== {dataset_name.upper()} RESULTS ====")
    for i, val in enumerate(class_iou):
        print(f"Class {i}: IoU = {metric_string(val)}")
    print(f"Mean IoU: {metric_string(miou)}")
    print(f"Mean IoU no background: {metric_string(miou_no_background)}")
    print(f"FWIoU: {metric_string(metrics['FWIoU'])}")
    print(
        f"Per-image mIoU: Mean = {metric_string(nanmean_or_nan(per_image_ious))}, "
        f"Std = {metric_string(nanstd_or_nan(per_image_ious))}"
    )
    print(
        f"Per-image mIoU no background: Mean = {metric_string(nanmean_or_nan(per_image_ious_no_background))}, "
        f"Std = {metric_string(nanstd_or_nan(per_image_ious_no_background))}"
    )


def default_model_path(dataset, split, num_labels, apply_aug, seed, reco, output_root=".", run_id=None):
    tag = f"{dataset}_split{split}_label{num_labels}_semi_{apply_aug}_{seed}"
    if reco:
        tag = f"{tag}_reco"
    if run_id:
        tag = f"{tag}_{run_id}"
    return os.path.join(output_root, "model_weights", f"{tag}.pth")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ReCo checkpoints on RescueNet/FloodNet")
    parser.add_argument("--dataset", choices=["rescuenet", "floodnet"], required=True)
    parser.add_argument("--split", default="25", choices=["12_5", "25", "50"])
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--num-labels", default=15, type=int)
    parser.add_argument("--apply-aug", default="classmix")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--no-reco", action="store_true", help="look for a non-ReCo checkpoint name")
    parser.add_argument("--backbone", choices=["deeplabv3p", "deeplabv2"], default="deeplabv3p")
    parser.add_argument("--output-dim", default=256, type=int)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--output-root", default=".")
    parser.add_argument("--run-id", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model_path = args.model_path or default_model_path(
        args.dataset, args.split, args.num_labels, args.apply_aug, args.seed, not args.no_reco,
        args.output_root, args.run_id)
    evaluate_dataset(args.dataset, model_path, args.backbone, args.output_dim, args.output_dir)
