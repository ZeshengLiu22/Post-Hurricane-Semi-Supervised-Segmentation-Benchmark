import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torchvision import transforms
from seg_core.model import MiT_SegFormer


def create_floodnet_label_colormap():
    colormap = np.zeros((256, 3), dtype=np.uint8)
    colormap[0] = [0, 0, 0]               # Background
    colormap[1] = [255, 0, 0]             # Building-flooded
    colormap[2] = [180, 120, 120]         # Building-non-flooded
    colormap[3] = [160, 150, 20]          # Road-flooded
    colormap[4] = [140, 140, 140]         # Road-non-flooded
    colormap[5] = [61, 230, 250]          # Water
    colormap[6] = [0, 82, 255]            # Tree
    colormap[7] = [255, 0, 245]           # Vehicle
    colormap[8] = [255, 235, 0]           # Pool
    colormap[9] = [4, 250, 7]             # Grass
    return colormap


def load_image(path):
    img = Image.open(path).convert('RGB')
    orig_size = img.size  # (W, H)
    w, h = orig_size
    img_small = img.resize((w // 2, h // 2), Image.BILINEAR)
    return transforms.ToTensor()(img_small).unsqueeze(0), (h, w)  # return original H, W


def load_mask(path):
    return np.array(Image.open(path), dtype=np.uint8)


def colorize_mask(mask, colormap):
    color_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for label_id, color in enumerate(colormap):
        color_mask[mask == label_id] = color
    return color_mask


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--val-img-dir', type=str, required=True)
    parser.add_argument('--val-mask-dir', type=str, required=True)
    parser.add_argument('--save-dir', type=str, default='./predictions')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--backbone', type=str, default='b1')
    parser.add_argument('--num-classes', type=int, default=10)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    colormap = create_floodnet_label_colormap()

    # Load model on GPU
    model = MiT_SegFormer(backbone='mit_' + args.backbone,
                          num_classes=args.num_classes,
                          embedding_dim=256,
                          pretrained=False)

    # Strip "module." prefix if needed
    raw_state = torch.load(args.checkpoint, map_location='cpu')
    new_state = {k.replace('module.', ''): v for k, v in raw_state.items()}
    model.load_state_dict(new_state)
    model.cuda().eval()

    image_names = sorted(os.listdir(args.val_img_dir))
    total_confusion = np.zeros((args.num_classes, args.num_classes), dtype=np.int64)
    per_image_ious = []
    per_image_ious_no_background = []

    for name in tqdm(image_names, desc='Evaluating'):
        img_path = os.path.join(args.val_img_dir, name)

        base_name = os.path.splitext(name)[0]
        mask_name = base_name + '_lab.png'
        mask_path = os.path.join(args.val_mask_dir, mask_name)

        if not os.path.exists(mask_path):
            print(f"Warning: GT mask not found for {name} → {mask_name}, skipping.")
            continue

        img, orig_size = load_image(img_path)  # shape (1, 3, H/2, W/2), original (H, W)
        img = img.cuda()
        gt_mask = load_mask(mask_path)

        with torch.no_grad():
            pred_logits = model(img)
            pred_logits = F.interpolate(pred_logits, size=orig_size, mode='bilinear', align_corners=False)
            pred_mask = pred_logits.argmax(dim=1).squeeze().cpu().numpy().astype(np.uint8)

        color_mask = colorize_mask(pred_mask, colormap)
        save_path = os.path.join(args.save_dir, base_name + '.png')
        Image.fromarray(color_mask).save(save_path)

        per_confusion = confusion_matrix_from_arrays(pred_mask, gt_mask, args.num_classes, ignore_index=255)
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
    no_background = np.ones(args.num_classes, dtype=bool)
    if args.num_classes:
        no_background[0] = False
    miou = nanmean_or_nan(iou)
    miou_no_background = nanmean_or_nan(iou[no_background])

    print("\n==== FloodNet Evaluation (GPU, 2× Downscaled) ====")
    for i, class_iou in enumerate(iou):
        print(f"Class {i}: IoU = {metric_string(class_iou)}")
    print(f"Mean IoU: {metric_string(miou)}")
    print(f"Mean IoU no background: {metric_string(miou_no_background)}")
    print(f"FWIoU:    {metric_string(metrics['FWIoU'])}")
    print(
        f"Per-image mIoU: mean={metric_string(nanmean_or_nan(per_image_ious))}, "
        f"std={metric_string(nanstd_or_nan(per_image_ious))}"
    )
    print(
        f"Per-image mIoU no background: mean={metric_string(nanmean_or_nan(per_image_ious_no_background))}, "
        f"std={metric_string(nanstd_or_nan(per_image_ious_no_background))}"
    )


if __name__ == '__main__':
    main()
