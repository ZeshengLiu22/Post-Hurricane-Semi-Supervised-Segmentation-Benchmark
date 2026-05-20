'''
Training code for Reco on semi-supervised semantic segmentations
Modified based on official github implementations
****
Siginificant Changes by Zesheng: Use Accelerate to parallel the code and make it able to run on multiple GPUs.
****
'''

import torch
import torchvision.models as models
import torch.optim as optim
import argparse
import os

from network.deeplabv3.deeplabv3 import *
from network.deeplabv2 import *
from build_data import *
from module_list import *
from tqdm.auto import tqdm

from accelerate import Accelerator, DistributedDataParallelKwargs
import warnings
warnings.filterwarnings("ignore")


ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(
    device_placement=True,
    step_scheduler_with_optimizer=False,
    mixed_precision='bf16',
    kwargs_handlers=[ddp_kwargs],
)
torch.autograd.set_detect_anomaly(False)


parser = argparse.ArgumentParser(description='Semi-supervised Segmentation with Perfect Labels')
parser.add_argument('--mode', default=None, type=str)
parser.add_argument('--port', default=None, type=int)

parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--num_labels', default=15, type=int, help='number of labelled training data, set 0 to use all training data')
parser.add_argument('--split', default='25', choices=['12_5', '25', '50'], help='post-disaster labelled/unlabelled split percentage')
parser.add_argument('--lr', default=2.5e-3, type=float)
parser.add_argument('--weight_decay', default=5e-4, type=float)
parser.add_argument('--dataset', default='cityscapes', type=str, help='pascal, cityscapes, sun, rescuenet, floodnet')
parser.add_argument('--apply_aug', default='cutout', type=str, help='apply semi-supervised method: cutout cutmix classmix')
parser.add_argument('--id', default=1, type=int, help='number of repeated samples')
parser.add_argument('--weak_threshold', default=0.7, type=float)
parser.add_argument('--strong_threshold', default=0.97, type=float)
parser.add_argument('--apply_reco', action='store_true')
parser.add_argument('--num_negatives', default=512, type=int, help='number of negative keys')
parser.add_argument('--num_queries', default=256, type=int, help='number of queries per segment per image')
parser.add_argument('--temp', default=0.5, type=float)
parser.add_argument('--output_dim', default=256, type=int, help='output dimension from representation head')
parser.add_argument('--backbone', default='deeplabv3p', type=str, help='choose backbone: deeplabv3p, deeplabv2')
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--num_gpu', default=8, type=int)
parser.add_argument('--epochs', default=150, type=int)
parser.add_argument('--num-workers', default=4, type=int)
parser.add_argument('--val-num-workers', default=4, type=int)
parser.add_argument('--pin-memory', default=False, type=lambda x: str(x).lower() in ('1', 'true', 'yes', 'y'))
parser.add_argument('--prefetch-factor', default=4, type=int)
parser.add_argument('--persistent-workers', default=True, type=lambda x: str(x).lower() in ('1', 'true', 'yes', 'y'))
parser.add_argument('--output-root', default='.', type=str, help='root directory for model_weights/ and logging/')
parser.add_argument('--run-id', default=None, type=str, help='optional suffix to avoid overwriting repeated runs')

args = parser.parse_args()


def run_tag(args):
    tag = '{}_split{}_label{}_semi_{}_{}'.format(args.dataset, args.split, args.num_labels, args.apply_aug, args.seed)
    if args.apply_reco:
        tag = '{}_reco'.format(tag)
    if args.run_id:
        tag = '{}_{}'.format(tag, args.run_id)
    return tag

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

model_weights_dir = os.path.join(args.output_root, 'model_weights')
logging_dir = os.path.join(args.output_root, 'logging')
if accelerator.is_main_process:
    os.makedirs(model_weights_dir, exist_ok=True)
    os.makedirs(logging_dir, exist_ok=True)

data_loader = BuildDataLoader(
    args.dataset,
    args.num_labels,
    split=args.split,
    num_workers=args.num_workers,
    val_num_workers=args.val_num_workers,
    pin_memory=args.pin_memory,
    prefetch_factor=args.prefetch_factor,
    persistent_workers=args.persistent_workers,
)
train_l_loader, train_u_loader, test_loader = data_loader.build(supervised=False)

# Load Semantic Network
#device = torch.device("cuda:{:d}".format(args.gpu) if torch.cuda.is_available() else "cpu")
device = accelerator.device

if args.backbone == 'deeplabv3p':
    model = DeepLabv3Plus(models.resnet101(pretrained=True), num_classes=data_loader.num_segments, output_dim=args.output_dim).to(accelerator.device)
elif args.backbone == 'deeplabv2':
    model = DeepLabv2(models.resnet101(pretrained=True), num_classes=data_loader.num_segments, output_dim=args.output_dim).to(accelerator.device)

total_epoch = args.epochs
ema = EMA(model, 0.99)  # Mean teacher model

optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9, nesterov=True)
scheduler = PolyLR(optimizer, total_epoch, power=0.9)

model, ema.model, optimizer, scheduler, train_l_loader, train_u_loader, test_loader = accelerator.prepare(
    model, ema.model, optimizer, scheduler, train_l_loader, train_u_loader, test_loader)

train_epoch = len(train_l_loader)
test_epoch = len(test_loader)
avg_cost = np.zeros((total_epoch, 10))
iteration = 0

for index in tqdm(range(total_epoch), disable=not accelerator.is_local_main_process):
    cost = np.zeros(3)
    train_l_dataset = iter(train_l_loader)
    train_u_dataset = iter(train_u_loader)

    model.train()
    ema.model.train()
    l_conf_mat = ConfMatrix(data_loader.num_segments)
    u_conf_mat = ConfMatrix(data_loader.num_segments)
    for i in range(train_epoch): # Train epoch is length of train loader
        train_l_data, train_l_label = next(train_l_dataset) #Changed by Zesheng for debug
        #train_l_data, train_l_label = train_l_data.to(device), train_l_label.to(device)

        train_u_data, train_u_label = next(train_u_dataset) #Changed by Zesheng for debug
        #train_u_data, train_u_label = train_u_data.to(device), train_u_label.to(device)

        optimizer.zero_grad()

        # generate pseudo-labels
        with torch.no_grad():
            with accelerator.autocast():
                pred_u, _ = ema.model(train_u_data)
                pred_u_large_raw = F.interpolate(pred_u, size=train_u_label.shape[1:], mode='bilinear', align_corners=True)
                pseudo_logits, pseudo_labels = torch.max(torch.softmax(pred_u_large_raw, dim=1), dim=1)

            # random scale images first
            train_u_aug_data, train_u_aug_label, train_u_aug_logits = \
                batch_transform(train_u_data, pseudo_labels, pseudo_logits,
                                data_loader.crop_size, data_loader.scale_size, apply_augmentation=False)

            # apply mixing strategy: cutout, cutmix or classmix
            train_u_aug_data, train_u_aug_label, train_u_aug_logits = \
                generate_unsup_data(train_u_aug_data, train_u_aug_label, train_u_aug_logits, mode=args.apply_aug)

            # apply augmentation: color jitter + flip + gaussian blur
            train_u_aug_data, train_u_aug_label, train_u_aug_logits = \
                batch_transform(train_u_aug_data, train_u_aug_label, train_u_aug_logits,
                                data_loader.crop_size, (1.0, 1.0), apply_augmentation=True)

        # generate labelled and unlabelled data loss
        with accelerator.autocast():
            labeled_batch_size = train_l_data.shape[0]
            pred_all, rep_all = model(torch.cat((train_l_data, train_u_aug_data), dim=0))
            pred_l, pred_u = pred_all[:labeled_batch_size], pred_all[labeled_batch_size:]
            rep_l, rep_u = rep_all[:labeled_batch_size], rep_all[labeled_batch_size:]

            pred_l_large = F.interpolate(pred_l, size=train_l_label.shape[1:], mode='bilinear', align_corners=True)
            pred_u_large = F.interpolate(pred_u, size=train_l_label.shape[1:], mode='bilinear', align_corners=True)

            # supervised-learning loss
            sup_loss = compute_supervised_loss(pred_l_large, train_l_label)

            # unsupervised-learning loss
            unsup_loss = compute_unsupervised_loss(pred_u_large, train_u_aug_label, train_u_aug_logits, args.strong_threshold)

            # apply regional contrastive loss
            if args.apply_reco:
                with torch.no_grad():
                    train_u_aug_mask = train_u_aug_logits.ge(args.weak_threshold).float()
                    mask_all = torch.cat(((train_l_label.unsqueeze(1) >= 0).float(), train_u_aug_mask.unsqueeze(1)))
                    mask_all = F.interpolate(mask_all, size=pred_all.shape[2:], mode='nearest')

                    label_l = F.interpolate(label_onehot(train_l_label, data_loader.num_segments), size=pred_all.shape[2:], mode='nearest')
                    label_u = F.interpolate(label_onehot(train_u_aug_label, data_loader.num_segments), size=pred_all.shape[2:], mode='nearest')
                    label_all = torch.cat((label_l, label_u))

                    prob_l = torch.softmax(pred_l, dim=1)
                    prob_u = torch.softmax(pred_u, dim=1)
                    prob_all = torch.cat((prob_l, prob_u))

                reco_loss = compute_reco_loss(rep_all, label_all, mask_all, prob_all, args.strong_threshold,
                                            args.temp, args.num_queries, args.num_negatives)
            else:
                reco_loss = pred_l.new_tensor(0.0)
        
            loss = sup_loss + unsup_loss + reco_loss
        
        #loss.backward()
        accelerator.backward(loss)
        optimizer.step()
        ema.update(model)

        l_conf_mat.update(pred_l_large.argmax(1).flatten(), train_l_label.flatten())
        u_conf_mat.update(pred_u_large_raw.argmax(1).flatten(), train_u_label.flatten())

        cost[0] = sup_loss.item()
        cost[1] = unsup_loss.item()
        cost[2] = reco_loss.item()
        avg_cost[index, :3] += cost / train_epoch
        iteration += 1

    scheduler.step()

    avg_cost[index, 3:5] = l_conf_mat.get_metrics()
    avg_cost[index, 5:7] = u_conf_mat.get_metrics()

    with torch.no_grad():
        ema.model.eval()
        conf_mat = ConfMatrix(data_loader.num_segments)
        test_loss_sum = torch.tensor(0.0, device=device)
        test_sample_count = torch.tensor(0.0, device=device)
        for test_data, test_label in test_loader:

            with accelerator.autocast():
                pred, rep = ema.model(test_data)
                pred = F.interpolate(pred, size=test_label.shape[1:], mode='bilinear', align_corners=True)
                # Add Post-processing here

                loss = compute_supervised_loss(pred, test_label)

            conf_mat.update(pred.argmax(1).flatten(), test_label.flatten())
            test_loss_sum += loss.detach() * test_data.shape[0]
            test_sample_count += test_data.shape[0]

        if conf_mat.mat is None:
            conf_mat.mat = torch.zeros((data_loader.num_segments, data_loader.num_segments), dtype=torch.int64, device=device)

        conf_mat.mat = accelerator.reduce(conf_mat.mat, reduction="sum")
        test_loss_sum = accelerator.reduce(test_loss_sum, reduction="sum")
        test_sample_count = accelerator.reduce(test_sample_count, reduction="sum")

        miou, acc, iou_per_class = conf_mat.get_metrics(test=True)
        avg_cost[index, 7] = (test_loss_sum / test_sample_count.clamp_min(1)).item()
        avg_cost[index, 8] = miou
        avg_cost[index, 9] = acc
        
    accelerator.print('EPOCH: {:04d} ITER: {:04d} | TRAIN [Loss | mIoU | Acc.]: {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} || Test [Loss | mIoU | Acc.]: {:.4f} {:.4f} {:.4f}'
        .format(index, iteration, avg_cost[index][0], avg_cost[index][1], avg_cost[index][2],
                avg_cost[index][3], avg_cost[index][4], avg_cost[index][5], avg_cost[index][6], avg_cost[index][7], avg_cost[index][8],
                avg_cost[index][9]))
    accelerator.print('Top: mIoU {:.4f} Acc {:.4f}'.format(avg_cost[:, 8].max(), avg_cost[:, 9].max()))

    if avg_cost[index][8] >= avg_cost[:, 8].max():
        best_iou_per_class = iou_per_class
        accelerator.print('Top IOU per class:', best_iou_per_class)
        accelerator.print('Current Confusion Matrix:', conf_mat.mat)

        if accelerator.is_main_process:
            tag = run_tag(args)
            torch.save(accelerator.unwrap_model(ema.model).state_dict(), os.path.join(model_weights_dir, '{}.pth'.format(tag)))

    if accelerator.is_main_process:
        tag = run_tag(args)
        np.save(os.path.join(logging_dir, '{}.npy'.format(tag)), avg_cost)
