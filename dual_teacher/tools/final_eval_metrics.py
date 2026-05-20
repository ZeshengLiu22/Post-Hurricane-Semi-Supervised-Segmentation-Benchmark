import numpy as np


def _nanmean_or_nan(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else float("nan")


def _nanstd_or_nan(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(values.std(ddof=0)) if values.size else float("nan")


def _fast_hist(label_true, label_pred, num_classes, ignore_index=None):
    label_true = np.asarray(label_true).reshape(-1).astype(np.int64, copy=False)
    label_pred = np.asarray(label_pred).reshape(-1).astype(np.int64, copy=False)
    mask = (label_true >= 0) & (label_true < num_classes)
    mask &= (label_pred >= 0) & (label_pred < num_classes)
    if ignore_index is not None:
        mask &= label_true != ignore_index
    if not np.any(mask):
        return np.zeros((num_classes, num_classes), dtype=np.int64)
    return np.bincount(
        num_classes * label_true[mask] + label_pred[mask],
        minlength=num_classes ** 2,
    ).reshape(num_classes, num_classes)


def _metrics_from_hist(hist):
    hist = hist.astype(np.float64, copy=False)
    intersection = np.diag(hist)
    target_pixels = hist.sum(axis=1)
    predicted_pixels = hist.sum(axis=0)
    union = target_pixels + predicted_pixels - intersection
    iou = np.full(hist.shape[0], np.nan, dtype=np.float64)
    np.divide(intersection, union, out=iou, where=union > 0)
    acc_cls = np.full(hist.shape[0], np.nan, dtype=np.float64)
    np.divide(intersection, target_pixels, out=acc_cls, where=target_pixels > 0)
    total_target = target_pixels.sum()
    return {
        "target_pixels": target_pixels,
        "iou": iou,
        "acc": float(intersection.sum() / total_target) if total_target > 0 else float("nan"),
        "acc_cls": acc_cls,
        "FWIoU": float(np.nansum(target_pixels * iou) / total_target) if total_target > 0 else float("nan"),
    }


def scores(label_trues, label_preds, num_classes, ignore_index=None):
    hist = np.zeros((num_classes, num_classes), dtype=np.int64)
    per_image_mious = []
    per_image_mious_no_background = []
    for lt, lp in zip(label_trues, label_preds):
        per_hist = _fast_hist(lt, lp, num_classes, ignore_index=ignore_index)
        hist += per_hist
        per_metrics = _metrics_from_hist(per_hist)
        present = per_metrics["target_pixels"] > 0
        present_no_background = present.copy()
        if present_no_background.size:
            present_no_background[0] = False
        per_image_mious.append(_nanmean_or_nan(per_metrics["iou"][present]))
        per_image_mious_no_background.append(_nanmean_or_nan(per_metrics["iou"][present_no_background]))

    metrics = _metrics_from_hist(hist)
    no_background = np.ones(num_classes, dtype=bool)
    if num_classes:
        no_background[0] = False

    return {
        "Pixel Accuracy": metrics["acc"],
        "Mean Accuracy": _nanmean_or_nan(metrics["acc_cls"]),
        "Mean IoU": _nanmean_or_nan(metrics["iou"]),
        "Mean IoU no background": _nanmean_or_nan(metrics["iou"][no_background]),
        "FWIoU": metrics["FWIoU"],
        "Per-image Mean IoU": _nanmean_or_nan(per_image_mious),
        "Per-image Mean IoU Std": _nanstd_or_nan(per_image_mious),
        "Per-image Mean IoU no background": _nanmean_or_nan(per_image_mious_no_background),
        "Per-image Mean IoU no background Std": _nanstd_or_nan(per_image_mious_no_background),
        "Class IoU": dict(zip(range(num_classes), metrics["iou"])),
    }
