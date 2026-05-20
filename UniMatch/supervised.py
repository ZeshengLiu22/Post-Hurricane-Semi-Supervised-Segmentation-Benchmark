import argparse
import logging
import os
import pprint

import torch
import numpy as np
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml

from dataset.semi import SemiDataset
from model.semseg.deeplabv3plus import DeepLabV3Plus
from util.classes import CLASSES
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import count_params, AverageMeter, intersectionAndUnion, init_log
from util.dist_helper import setup_distributed
from PIL import Image
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--labeled-id-path', type=str, required=True)
parser.add_argument('--unlabeled-id-path', type=str, default=None)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--local_rank', '--local-rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)


FLOODNET_COLORMAP = [
    (0, 0, 0),         # 0: background
    (255, 0, 0),       # 1: building-flooded
    (180, 120, 120),   # 2: building-non-flooded
    (160, 150, 20),    # 3: road-flooded
    (140, 140, 140),   # 4: road-non-flooded
    (61, 230, 250),    # 5: water
    (0, 82, 255),      # 6: tree
    (255, 0, 245),     # 7: vehicle
    (255, 235, 0),     # 8: pool
    (4, 250, 7),       # 9: grass
]

RESCUENET_COLORMAP = [
    (0, 0, 0),         # 0: unlabeled
    (61, 230, 250),    # 1: water
    (180, 120, 120),   # 2: building-no-damage
    (235, 255, 7),     # 3: building-medium-damage
    (255, 184, 6),     # 4: building-major-damage
    (255, 0, 0),       # 5: building-total-destruction
    (255, 0, 245),     # 6: vehicle
    (140, 140, 140),   # 7: road-clear
    (160, 150, 20),    # 8: road-blocked
    (4, 250, 7),       # 9: tree
    (255, 235, 0),     # 10: pool
]

def colorize_mask_fixed(pred_mask, dataset):
    """Convert a class-indexed mask to an RGB image using a fixed 11-class colormap."""
    h, w = pred_mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    if dataset == 'FloodNet':
        COLORMAP = FLOODNET_COLORMAP
    else:
        COLORMAP = RESCUENET_COLORMAP
    for class_idx, color in enumerate(COLORMAP):
        color_mask[pred_mask == class_idx] = color
    return Image.fromarray(color_mask)


def evaluate(model, loader, mode, cfg, eval=False):
    model.eval()
    assert mode in ['original', 'center_crop', 'sliding_window']
    distributed = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if distributed else 0
    device = torch.device('cuda', torch.cuda.current_device())
    amp_enabled = bool(cfg.get('amp', False))
    amp_dtype_name = str(cfg.get('amp_dtype', 'bfloat16')).lower()
    amp_dtype = torch.bfloat16 if amp_dtype_name in ('bf16', 'bfloat16') else torch.float16
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()  # New: to accumulate ground truth pixel counts per class
    intersection_sum = np.zeros(cfg['nclass'], dtype=np.float64)
    union_sum = np.zeros(cfg['nclass'], dtype=np.float64)
    target_sum = np.zeros(cfg['nclass'], dtype=np.float64)
    
    per_image_ious = []
    image_ids = []

    best_iou = -1
    worst_iou = 101
    best_id = None
    worst_id = None

    with torch.no_grad():
        for img, mask, id in tqdm(loader, disable=(rank != 0)):
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

            intersection, union, target = \
                intersectionAndUnion(pred.cpu().numpy(), mask.numpy(), cfg['nclass'], 255)
            intersection_sum += intersection
            union_sum += union
            target_sum += target

            if eval:
                for i in range(pred.shape[0]):
                    pred_mask = pred[i].cpu().byte().numpy()
                    img_out = Image.fromarray(pred_mask)
                    color_mask = colorize_mask_fixed(pred_mask, 'RescueNet')
                    save_name = id[i].split('/')[2].split(' ')[0].replace('.jpg', '_pred.png')
                    color_mask.save(os.path.join("./predictions/RescueNet", save_name))

            for i in range(pred.shape[0]):
                inter = ((pred[i].cpu().numpy() == mask[i].numpy()) & (mask[i].numpy() != 255)).sum()
                union = ((pred[i].cpu().numpy() != 255) | (mask[i].numpy() != 255)).sum()
                iou = inter / (union + 1e-10) * 100
                per_image_ious.append(iou)
                image_ids.append(id[i])

                if iou > best_iou:
                    best_iou = iou
                    best_id = id[i]
                if iou < worst_iou:
                    worst_iou = iou
                    worst_id = id[i]

    reduced_intersection = torch.from_numpy(intersection_sum).to(device=device)
    reduced_union = torch.from_numpy(union_sum).to(device=device)
    reduced_target = torch.from_numpy(target_sum).to(device=device)

    if distributed:
        dist.all_reduce(reduced_intersection)
        dist.all_reduce(reduced_union)
        dist.all_reduce(reduced_target)

    intersection_meter.update(reduced_intersection.cpu().numpy())
    union_meter.update(reduced_union.cpu().numpy())
    target_meter.update(reduced_target.cpu().numpy())  # New: accumulate class pixel frequencies

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10) * 100.0
    mIOU = np.mean(iou_class)

    avg_iou = np.mean(per_image_ious)
    std_iou = np.std(per_image_ious)

    if rank == 0:
        print(f"Mean IoU: {mIOU:.2f}")
        print(f"Image-level IoU - Average: {avg_iou:.2f}, Std Dev: {std_iou:.2f}")
        print(f"Best Image: ID = {best_id}, IoU = {best_iou:.2f}")
        print(f"Worst Image: ID = {worst_id}, IoU = {worst_iou:.2f}")

    # New: compute FWIoU
    freq = target_meter.sum / (np.sum(target_meter.sum) + 1e-10)
    fwiou = np.sum(freq * (intersection_meter.sum / (union_meter.sum + 1e-10))) * 100
    fwiou_class = freq * iou_class * 100

    return mIOU, iou_class, fwiou, fwiou_class


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
    if rank == 0:
        logger.info('Total params: {:.1f}M\n'.format(count_params(model)))

    optimizer = SGD([{'params': model.backbone.parameters(), 'lr': cfg['lr']},
                     {'params': [param for name, param in model.named_parameters() if 'backbone' not in name],
                      'lr': cfg['lr'] * cfg['lr_multi']}], lr=cfg['lr'], momentum=0.9, weight_decay=1e-4)

    local_rank = int(os.environ["LOCAL_RANK"])
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False,
                                                      output_device=local_rank, find_unused_parameters=False)

    if cfg['criterion']['name'] == 'CELoss':
        criterion = nn.CrossEntropyLoss(**cfg['criterion']['kwargs']).cuda(local_rank)
    elif cfg['criterion']['name'] == 'OHEM':
        criterion = ProbOhemCrossEntropy2d(**cfg['criterion']['kwargs']).cuda(local_rank)
    else:
        raise NotImplementedError('%s criterion is not implemented' % cfg['criterion']['name'])

    trainset = SemiDataset(cfg['dataset'], cfg['data_root'], 'train_l', cfg['crop_size'], args.labeled_id_path)
    valset = SemiDataset(cfg['dataset'], cfg['data_root'], 'val')

    trainsampler = torch.utils.data.distributed.DistributedSampler(trainset)
    trainloader = DataLoader(trainset, batch_size=cfg['batch_size'],
                             pin_memory=True, num_workers=1, drop_last=True, sampler=trainsampler)
    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=1,
                           drop_last=False, sampler=valsampler)

    iters = 0
    total_iters = len(trainloader) * cfg['epochs']
    previous_best = 0.0
    epoch = -1

    if os.path.exists(os.path.join(args.save_path, 'latest.pth')):
        checkpoint = torch.load(os.path.join(args.save_path, 'latest.pth'))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        previous_best = checkpoint['previous_best']
        
        if rank == 0:
            logger.info('************ Load from checkpoint at epoch %i\n' % epoch)
    
    for epoch in range(epoch + 1, cfg['epochs']):
        if rank == 0:
            logger.info('===========> Epoch: {:}, LR: {:.5f}, Previous best: {:.2f}'.format(
                epoch, optimizer.param_groups[0]['lr'], previous_best))

        model.train()
        total_loss = AverageMeter()

        trainsampler.set_epoch(epoch)

        for i, (img, mask) in enumerate(trainloader):

            img, mask = img.cuda(), mask.cuda()

            pred = model(img)

            loss = criterion(pred, mask)
            
            torch.distributed.barrier()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss.update(loss.item())

            iters = epoch * len(trainloader) + i
            lr = cfg['lr'] * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * cfg['lr_multi']

            if rank == 0:
                writer.add_scalar('train/loss_all', loss.item(), iters)
                writer.add_scalar('train/loss_x', loss.item(), iters)
            
            if (i % (max(2, len(trainloader) // 8)) == 0) and (rank == 0):
                logger.info('Iters: {:}, Total loss: {:.3f}'.format(i, total_loss.avg))

        eval_mode = 'sliding_window' if cfg['dataset'] == 'cityscapes' else 'original'
        mIoU, iou_class, *_ = evaluate(model, valloader, eval_mode, cfg)

        if rank == 0:
            for (cls_idx, iou) in enumerate(iou_class):
                logger.info('***** Evaluation ***** >>>> Class [{:} {:}] '
                            'IoU: {:.2f}'.format(cls_idx, CLASSES[cfg['dataset']][cls_idx], iou))
            logger.info('***** Evaluation {} ***** >>>> MeanIoU: {:.2f}\n'.format(eval_mode, mIoU))
            
            writer.add_scalar('eval/mIoU', mIoU, epoch)
            for i, iou in enumerate(iou_class):
                writer.add_scalar('eval/%s_IoU' % (CLASSES[cfg['dataset']][i]), iou, epoch)

        is_best = mIoU > previous_best
        previous_best = max(mIoU, previous_best)
        if rank == 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'previous_best': previous_best,
            }
            torch.save(checkpoint, os.path.join(args.save_path, 'latest.pth'))
            if is_best:
                torch.save(checkpoint, os.path.join(args.save_path, 'best.pth'))


if __name__ == '__main__':
    main()
