import argparse
import logging
import os
import pprint

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml

from dataset.semi import SemiDataset
from model.semseg.deeplabv3plus import DeepLabV3Plus
from util.classes import CLASSES
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import count_params, init_log, AverageMeter
from util.dist_helper import setup_distributed


parser = argparse.ArgumentParser(description='Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--labeled-id-path', type=str, required=True)
parser.add_argument('--unlabeled-id-path', type=str, required=True)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)


def nanmean_or_nan(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else float("nan")


def nanstd_or_nan(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(values.std(ddof=0)) if values.size else float("nan")


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
    return {"target_pixels": target_pixels, "iou": iou, "FWIoU": fwiou}


def evaluate_final_metrics(model, loader, mode, cfg):
    model.eval()
    assert mode in ['original', 'center_crop', 'sliding_window']
    distributed = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if distributed else 0
    device = torch.device('cuda', torch.cuda.current_device())
    amp_enabled = bool(cfg.get('amp', False))
    amp_dtype_name = str(cfg.get('amp_dtype', 'bfloat16')).lower()
    amp_dtype = torch.bfloat16 if amp_dtype_name in ('bf16', 'bfloat16') else torch.float16
    ignore_index = cfg.get('criterion', {}).get('kwargs', {}).get('ignore_index', 255)

    confusion = np.zeros((cfg['nclass'], cfg['nclass']), dtype=np.int64)
    per_image_mious = []
    per_image_mious_no_background = []

    with torch.no_grad():
        for img, mask, _ in loader:
            img = img.cuda(non_blocking=True)

            if mode == 'sliding_window':
                grid = cfg['crop_size']
                b, _, h, w = img.shape
                final = torch.zeros(b, cfg['nclass'], h, w, device=img.device)
                row = 0
                while row < h:
                    col = 0
                    while col < w:
                        with torch.cuda.amp.autocast(enabled=amp_enabled, dtype=amp_dtype):
                            pred = model(img[:, :, row: min(h, row + grid), col: min(w, col + grid)])
                        final[:, :, row: min(h, row + grid), col: min(w, col + grid)] += pred.softmax(dim=1)
                        col += int(grid * 2 / 3)
                    row += int(grid * 2 / 3)
                pred = final.argmax(dim=1)
            else:
                if mode == 'center_crop':
                    h, w = img.shape[-2:]
                    start_h, start_w = (h - cfg['crop_size']) // 2, (w - cfg['crop_size']) // 2
                    img = img[:, :, start_h:start_h + cfg['crop_size'], start_w:start_w + cfg['crop_size']]
                    mask = mask[:, start_h:start_h + cfg['crop_size'], start_w:start_w + cfg['crop_size']]
                with torch.cuda.amp.autocast(enabled=amp_enabled, dtype=amp_dtype):
                    pred = model(img).argmax(dim=1)

            pred_np = pred.cpu().numpy()
            mask_np = mask.numpy()
            confusion += confusion_matrix_from_arrays(pred_np, mask_np, cfg['nclass'], ignore_index)

            for i in range(pred_np.shape[0]):
                per_confusion = confusion_matrix_from_arrays(pred_np[i], mask_np[i], cfg['nclass'], ignore_index)
                per_metrics = compute_metrics_from_confusion(per_confusion)
                present = per_metrics["target_pixels"] > 0
                present_no_background = present.copy()
                if present_no_background.size:
                    present_no_background[0] = False
                per_image_mious.append(nanmean_or_nan(per_metrics["iou"][present]))
                per_image_mious_no_background.append(nanmean_or_nan(per_metrics["iou"][present_no_background]))

    reduced_confusion = torch.from_numpy(confusion).to(device=device)
    if distributed:
        dist.all_reduce(reduced_confusion)

    metrics = compute_metrics_from_confusion(reduced_confusion.cpu().numpy())
    no_background = np.ones(cfg['nclass'], dtype=bool)
    if cfg['nclass']:
        no_background[0] = False
    iou_class = metrics["iou"] * 100.0
    mIoU = nanmean_or_nan(iou_class)
    fwIoU = metrics["FWIoU"] * 100.0
    fwiou_class = metrics["target_pixels"] * metrics["iou"] / max(metrics["target_pixels"].sum(), 1.0) * 100.0

    if rank == 0:
        print(f"Mean IoU no background: {nanmean_or_nan(iou_class[no_background]):.2f}")
        print(
            "Per-image mIoU: "
            f"mean={nanmean_or_nan(per_image_mious) * 100.0:.2f}, "
            f"std={nanstd_or_nan(per_image_mious) * 100.0:.2f}"
        )
        print(
            "Per-image mIoU no background: "
            f"mean={nanmean_or_nan(per_image_mious_no_background) * 100.0:.2f}, "
            f"std={nanstd_or_nan(per_image_mious_no_background) * 100.0:.2f}"
        )

    return mIoU, iou_class, fwIoU, fwiou_class


def main():
    args = parser.parse_args()

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    rank, world_size = setup_distributed(port=args.port)

    if rank == 0:
        all_args = {**cfg, **vars(args), 'ngpus': world_size}
        logger.info('{}\n'.format(pprint.pformat(all_args)))
        
        writer = SummaryWriter(args.save_path)
        
        os.makedirs(args.save_path, exist_ok=True)
    
    cudnn.enabled = True
    cudnn.benchmark = True

    model = DeepLabV3Plus(cfg)
    optimizer = SGD([{'params': model.backbone.parameters(), 'lr': cfg['lr']},
                    {'params': [param for name, param in model.named_parameters() if 'backbone' not in name],
                    'lr': cfg['lr'] * cfg['lr_multi']}], lr=cfg['lr'], momentum=0.9, weight_decay=1e-4)
    
    if rank == 0:
        logger.info('Total params: {:.1f}M\n'.format(count_params(model)))

    local_rank = int(os.environ["LOCAL_RANK"])
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False,
                                                      output_device=local_rank, find_unused_parameters=False)

    if cfg['criterion']['name'] == 'CELoss':
        criterion_l = nn.CrossEntropyLoss(**cfg['criterion']['kwargs']).cuda(local_rank)
    elif cfg['criterion']['name'] == 'OHEM':
        criterion_l = ProbOhemCrossEntropy2d(**cfg['criterion']['kwargs']).cuda(local_rank)
    else:
        raise NotImplementedError('%s criterion is not implemented' % cfg['criterion']['name'])

    criterion_u = nn.CrossEntropyLoss(reduction='none').cuda(local_rank)

    trainset_u = SemiDataset(cfg['dataset'], cfg['data_root'], 'train_u',
                             cfg['crop_size'], args.unlabeled_id_path)
    trainset_l = SemiDataset(cfg['dataset'], cfg['data_root'], 'train_l',
                             cfg['crop_size'], args.labeled_id_path, nsample=len(trainset_u.ids))
    valset = SemiDataset(cfg['dataset'], cfg['data_root'], 'val')

    trainsampler_l = torch.utils.data.distributed.DistributedSampler(trainset_l)
    trainloader_l = DataLoader(trainset_l, batch_size=cfg['batch_size'],
                               pin_memory=True, num_workers=1, drop_last=True, sampler=trainsampler_l)
    trainsampler_u = torch.utils.data.distributed.DistributedSampler(trainset_u)
    trainloader_u = DataLoader(trainset_u, batch_size=cfg['batch_size'],
                               pin_memory=True, num_workers=1, drop_last=True, sampler=trainsampler_u)
    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=1,
                           drop_last=False, sampler=valsampler)

    total_iters = len(trainloader_u) * cfg['epochs']
    previous_best = 0.0
    epoch = -1
    
    if os.path.exists(os.path.join(args.save_path, 'best.pth')):
        logger.info('************ Evaluating Model and best model FOUND *********************')
        checkpoint = torch.load(os.path.join(args.save_path, 'best.pth'))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        previous_best = checkpoint['previous_best']
        
        if rank == 0:
            logger.info('************ Load from checkpoint at epoch %i\n' % epoch)

        eval_mode = 'sliding_window' if cfg['dataset'] == 'cityscapes' else 'original'
        mIoU, iou_class, fwIoU, fwiou_class = evaluate_final_metrics(model, valloader, eval_mode, cfg)

        if rank == 0:
            for (cls_idx, iou) in enumerate(iou_class):
                logger.info('***** Evaluation ***** >>>> Class [{:} {:}] '
                            'IoU: {:.2f}'.format(cls_idx, CLASSES[0][cfg['dataset']][cls_idx], iou))
            logger.info('***** Evaluation {} ***** >>>> MeanIoU: {:.2f}\n'.format(eval_mode, mIoU))
            
            writer.add_scalar('eval/mIoU', mIoU, epoch)
            for i, iou in enumerate(iou_class):
                writer.add_scalar('eval/%s_IoU' % (CLASSES[0][cfg['dataset']][i]), iou, epoch)


            for (cls_idx, iou) in enumerate(fwiou_class):
                logger.info('***** Evaluation ***** >>>> Class [{:} {:}] '
                            'FWIoU contribution: {:.2f}'.format(cls_idx, CLASSES[0][cfg['dataset']][cls_idx], iou))
            logger.info('***** Evaluation {} ***** >>>> FWIoU: {:.2f}\n'.format(eval_mode, fwIoU))
            
            writer.add_scalar('eval/FWIoU', fwIoU, epoch)
            for i, iou in enumerate(fwiou_class):
                writer.add_scalar('eval/%s_FWIoU_contribution' % (CLASSES[0][cfg['dataset']][i]), iou, epoch)

if __name__ == '__main__':
    main()
