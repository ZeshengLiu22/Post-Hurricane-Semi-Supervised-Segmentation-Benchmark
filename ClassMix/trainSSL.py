import argparse
import os
import sys
import random
import timeit
import datetime

import numpy as np
import pickle
import scipy.misc

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils import data, model_zoo
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch.autograd import Variable
import torchvision.transforms as transform

from model.deeplabv2 import Res_Deeplab

from utils.loss import CrossEntropy2d
from utils.loss import CrossEntropyLoss2dPixelWiseWeighted
from utils.loss import MSELoss2d

from utils import transformmasks
from utils import transformsgpu
from utils.helpers import colorize_mask
import utils.palette as palette

from data.voc_dataset import VOCDataSet

from data import get_loader, get_data_path
from data.augmentations import *
from tqdm import tqdm

import PIL
from torchvision import transforms
import json
from pathlib import Path
from torch.utils import tensorboard
from evaluateSSL import evaluate

import time
import torch.backends.cudnn as cudnn

start = timeit.default_timer()
start_writeable = datetime.datetime.now().strftime('%m-%d_%H-%M')
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLASSMIX_COCO_PRETRAIN = os.environ.get(
    'CLASSMIX_COCO_PRETRAIN',
    str(PROJECT_ROOT / 'ClassMix' / 'pretrained' / 'resnet101COCO-41f33a49.pth'),
)


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ('yes', 'true', 't', '1', 'y'):
        return True
    if value in ('no', 'false', 'f', '0', 'n'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


def dataloader_kwargs(num_workers, pin_memory, prefetch_factor, persistent_workers):
    kwargs = {
        'num_workers': num_workers,
        'pin_memory': pin_memory,
    }
    if num_workers > 0:
        kwargs['prefetch_factor'] = prefetch_factor
        kwargs['persistent_workers'] = persistent_workers
    return kwargs


def setup_distributed():
    if 'RANK' not in os.environ or 'WORLD_SIZE' not in os.environ:
        return False, 0, 0, 1

    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')
    return True, local_rank, dist.get_rank(), dist.get_world_size()


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def unwrap_model(model):
    return model.module if hasattr(model, 'module') else model


def is_main_process():
    return not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0


def amp_dtype_from_args():
    return torch.bfloat16 if args.amp_dtype == 'bf16' else torch.float16


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--gpus", type=int, default=1,
                        help="choose number of gpu devices to use (default: 1)")
    parser.add_argument("-c", "--config", type=str, default='config.json',
                        help='Path to the config file (default: config.json)')
    parser.add_argument("-r", "--resume", type=str, default=None,
                        help='Path to the .pth file to resume from (default: None)')
    parser.add_argument("-n", "--name", type=str, default=None, required=True,
                        help='Name of the run (default: None)')
    parser.add_argument("--save-images", type=str2bool, default=False,
                        help='Save debug unlabeled images under the run checkpoint dir (default: false)')
    parser.add_argument("--amp", type=str2bool, default=True,
                        help='Use CUDA AMP for forward/backward (default: true)')
    parser.add_argument("--amp-dtype", type=str, default='bf16', choices=['bf16', 'fp16'],
                        help='AMP dtype. bf16 is intended for H100/Ampere+ GPUs (default: bf16)')
    return parser.parse_args()



def loss_calc(pred, label, criterion):
    label = Variable(label.long()).cuda(non_blocking=True)
    return criterion(pred, label)

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))

def adjust_learning_rate(optimizer, i_iter):
    lr = lr_poly(learning_rate, i_iter, num_iterations, lr_power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1 :
        optimizer.param_groups[1]['lr'] = lr * 10

def sigmoid_ramp_up(iter, max_iter):
    if iter >= max_iter:
        return 1
    else:
        return np.exp(- 5 * (1 - iter / max_iter) ** 2)

def create_ema_model(model):
    ema_model = Res_Deeplab(num_classes=num_classes)

    for param in ema_model.parameters():
        param.detach_()
    mp = list(unwrap_model(model).parameters())
    mcp = list(ema_model.parameters())
    n = len(mp)
    for i in range(0, n):
        mcp[i].data[:] = mp[i].data[:].clone()
    if distributed and use_sync_batchnorm:
        ema_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(ema_model)
    return ema_model

def update_ema_variables(ema_model, model, alpha_teacher, iteration):
    # Use the "true" average until the exponential average is more correct
    alpha_teacher = min(1 - 1 / (iteration + 1), alpha_teacher)
    ema_model = unwrap_model(ema_model)
    model = unwrap_model(model)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    return ema_model

def strongTransform(parameters, data=None, target=None):
    assert ((data is not None) or (target is not None))
    data, target = transformsgpu.mix(mask = parameters["Mix"], data = data, target = target)
    data, target = transformsgpu.colorJitter(colorJitter = parameters["ColorJitter"], img_mean = torch.from_numpy(IMG_MEAN.copy()).cuda(), data = data, target = target)
    data, target = transformsgpu.gaussian_blur(blur = parameters["GaussianBlur"], data = data, target = None)
    data, target = transformsgpu.flip(flip = parameters["flip"], data = data, target = target)
    return data, target

def weakTransform(parameters, data=None, target=None):
    data, target = transformsgpu.flip(flip = parameters["flip"], data = data, target = target)
    return data, target

def getWeakInverseTransformParameters(parameters):
    return parameters

def getStrongInverseTransformParameters(parameters):
    return parameters

class DeNormalize(object):
    def __init__(self, mean):
        self.mean = mean

    def __call__(self, tensor):
        IMG_MEAN = torch.from_numpy(self.mean.copy())
        IMG_MEAN, _ = torch.broadcast_tensors(IMG_MEAN.unsqueeze(1).unsqueeze(2), tensor)
        tensor = tensor+IMG_MEAN
        tensor = (tensor/255).float()
        tensor = torch.flip(tensor,(0,))
        return tensor

class Learning_Rate_Object(object):
    def __init__(self,learning_rate):
        self.learning_rate = learning_rate

def save_image(image, epoch, id, palette):
    debug_dir = os.path.join(checkpoint_dir, 'debug_images')
    os.makedirs(debug_dir, exist_ok=True)
    with torch.no_grad():
        if image.shape[0] == 3:
            restore_transform = transforms.Compose([
            DeNormalize(IMG_MEAN),
            transforms.ToPILImage()])

            image = restore_transform(image)
            #image = PIL.Image.fromarray(np.array(image)[:, :, ::-1])  # BGR->RGB
            image.save(os.path.join(debug_dir, str(epoch)+ id + '.png'))
        else:
            mask = image.numpy()
            colorized_mask = colorize_mask(mask, palette)
            colorized_mask.save(os.path.join(debug_dir, str(epoch)+ id + '.png'))

def _save_checkpoint(iteration, model, optimizer, config, ema_model, save_best=False, overwrite=True):
    if not is_main_process():
        return

    checkpoint = {
        'iteration': iteration,
        'optimizer': optimizer.state_dict(),
        'config': config,
    }
    checkpoint['model'] = unwrap_model(model).state_dict()
    if train_unlabeled:
        checkpoint['ema_model'] = unwrap_model(ema_model).state_dict()

    if save_best:
        filename = os.path.join(checkpoint_dir, f'best_model.pth')
        torch.save(checkpoint, filename)
        print("Saving current best model: best_model.pth")
    else:
        filename = os.path.join(checkpoint_dir, f'checkpoint-iter{iteration}.pth')
        print(f'\nSaving a checkpoint: {filename} ...')
        torch.save(checkpoint, filename)
        if overwrite:
            try:
                os.remove(os.path.join(checkpoint_dir, f'checkpoint-iter{iteration - save_checkpoint_every}.pth'))
            except:
                pass

def _resume_checkpoint(resume_path, model, optimizer, ema_model):
    print(f'Loading checkpoint : {resume_path}')
    checkpoint = torch.load(resume_path)

    # Load last run info, the model params, the optimizer and the loggers
    iteration = checkpoint['iteration'] + 1
    print('Starting at iteration: ' + str(iteration))

    unwrap_model(model).load_state_dict(checkpoint['model'])

    optimizer.load_state_dict(checkpoint['optimizer'])

    if train_unlabeled:
        unwrap_model(ema_model).load_state_dict(checkpoint['ema_model'])

    return iteration, model, optimizer, ema_model

def main():
    global num_iterations
    if is_main_process():
        print(config)

    best_mIoU = 0
    amp_enabled = args.amp and torch.cuda.is_available()
    amp_dtype = amp_dtype_from_args()
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled and amp_dtype == torch.float16)
    supervised_loss = CrossEntropy2d(ignore_label=ignore_label).cuda()

    if consistency_loss == 'CE':
        unlabeled_loss = CrossEntropyLoss2dPixelWiseWeighted(ignore_index=ignore_label).cuda()
    elif consistency_loss == 'MSE':
        unlabeled_loss = MSELoss2d().cuda()

    # cudnn.enabled = True

    # create network
    # cudnn.enabled = False 
    model = Res_Deeplab(num_classes=num_classes)

    if restore_from:
        # load pretrained parameters
        if restore_from[:4] == 'http' :
            saved_state_dict = model_zoo.load_url(restore_from)
        else:
            saved_state_dict = torch.load(restore_from)

        # Copy loaded parameters to model
        new_params = model.state_dict().copy()
        for name, param in new_params.items():
            if name in saved_state_dict and param.size() == saved_state_dict[name].size():
                new_params[name].copy_(saved_state_dict[name])
        model.load_state_dict(new_params)

    # Initiate ema-model
    if train_unlabeled:
        ema_model = create_ema_model(model)
        ema_model.train()
        ema_model = ema_model.cuda()
    else:
        ema_model = None

    if distributed and use_sync_batchnorm:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.train()
    model.cuda()
    if distributed:
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )

    cudnn.benchmark = True

    if dataset == 'pascal_voc':
        data_loader = get_loader(dataset)
        data_path = get_data_path(dataset)
        train_dataset = data_loader(data_path, crop_size=input_size, scale=random_scale, mirror=random_flip)
    
    if dataset == 'rescuenet':
        data_loader = get_loader(dataset)
        data_path = get_data_path(dataset)
        train_dataset = data_loader(data_path, crop_size=input_size, scale=random_scale, mirror=random_flip, split_percent=split_percent)
        train_unlabeled_dataset = data_loader(data_path, crop_size=input_size, scale=random_scale, mirror=random_flip, unlabeled=True, split_percent=split_percent)
    
    if dataset == 'floodnet':
        data_loader = get_loader(dataset)
        data_path = get_data_path(dataset)
        train_dataset = data_loader(data_path, crop_size=input_size, scale=random_scale, mirror=random_flip, split_percent=split_percent)
        train_unlabeled_dataset = data_loader(data_path, crop_size=input_size, scale=random_scale, mirror=random_flip, unlabeled=True, split_percent=split_percent)

    elif dataset == 'cityscapes':
        data_loader = get_loader('cityscapes')
        data_path = get_data_path('cityscapes')
        if random_crop:
            data_aug = Compose([RandomCrop_city(input_size)])
        else:
            data_aug = None

        train_dataset = data_loader(data_path, is_transform=True, augmentations=data_aug, img_size=input_size)

    if training_epochs is not None:
        steps_per_epoch = max(1, len(train_dataset) // batch_size)
        num_iterations = int(steps_per_epoch * training_epochs)
        config['training']['num_iterations'] = num_iterations

    train_dataset_size = len(train_dataset) + len(train_unlabeled_dataset)
    if is_main_process():
        print ('dataset size: ', train_dataset_size)

    partial_size = labeled_samples
    if is_main_process():
        print('Training with labeled samples:', partial_size)
    if split_id is not None:
        train_ids = pickle.load(open(split_id, 'rb'))
        if is_main_process():
            print('loading train ids from {}'.format(split_id))
    else:
        np.random.seed(random_seed)
        train_ids = np.arange(train_dataset_size)
        np.random.shuffle(train_ids)

    train_sampler = DistributedSampler(train_dataset, shuffle=True) if distributed else None
    trainloader = data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        **dataloader_kwargs(num_workers, pin_memory, prefetch_factor, persistent_workers),
    )
    trainloader_iter = iter(trainloader)

    if train_unlabeled:
        train_remain_sampler = DistributedSampler(train_unlabeled_dataset, shuffle=True) if distributed else None
        trainloader_remain = data.DataLoader(
            train_unlabeled_dataset,
            batch_size=batch_size,
            sampler=train_remain_sampler,
            shuffle=(train_remain_sampler is None),
            **dataloader_kwargs(unlabeled_num_workers, pin_memory, prefetch_factor, persistent_workers),
        )
        trainloader_remain_iter = iter(trainloader_remain)

    # Optimizer for segmentation network
    learning_rate_object = Learning_Rate_Object(config['training']['learning_rate'])

    if optimizer_type == 'SGD':
        optimizer = optim.SGD(unwrap_model(model).optim_parameters(learning_rate_object),
                    lr=learning_rate, momentum=momentum,weight_decay=weight_decay)

    optimizer.zero_grad()

    interp = nn.Upsample(size=(input_size[0], input_size[1]), mode='bilinear', align_corners=True)

    start_iteration = 0

    if args.resume:
        start_iteration, model, optimizer, ema_model = _resume_checkpoint(args.resume, model, optimizer, ema_model)

    accumulated_loss_l = []
    if train_unlabeled:
        accumulated_loss_u = []

    if is_main_process():
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        with open(checkpoint_dir + '/config.json', 'w') as handle:
            json.dump(config, handle, indent=4, sort_keys=False)
        pickle.dump(train_ids, open(os.path.join(checkpoint_dir, 'train_split.pkl'), 'wb'))

    epochs_since_start = 0
    for i_iter in range(start_iteration, num_iterations):
        model.train()

        loss_l_value = 0
        if train_unlabeled:
            loss_u_value = 0

        optimizer.zero_grad()

        if lr_schedule:
            adjust_learning_rate(optimizer, i_iter)

        # Training loss for labeled data only
        try:
            batch = next(trainloader_iter)
            if batch[0].shape[0] != batch_size:
                batch = next(trainloader_iter)
        except:
            epochs_since_start = epochs_since_start + 1
            if distributed and train_sampler is not None:
                train_sampler.set_epoch(epochs_since_start)
            if is_main_process():
                print('Epochs since start: ',epochs_since_start)
            trainloader_iter = iter(trainloader)
            batch = next(trainloader_iter)

        weak_parameters={"flip": 0}

        images, labels, _, _, _ = batch
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)

        images, labels = weakTransform(weak_parameters, data = images, target = labels)

        with torch.cuda.amp.autocast(enabled=amp_enabled, dtype=amp_dtype):
            pred = interp(model(images))
            L_l = loss_calc(pred, labels, supervised_loss)

        if train_unlabeled:
            try:
                batch_remain = next(trainloader_remain_iter)
                if batch_remain[0].shape[0] != batch_size:
                    batch_remain = next(trainloader_remain_iter)
            except:
                trainloader_remain_iter = iter(trainloader_remain)
                batch_remain = next(trainloader_remain_iter)

            images_remain, _, _, _, _ = batch_remain
            images_remain = images_remain.cuda(non_blocking=True)
            inputs_u_w, _ = weakTransform(weak_parameters, data = images_remain)
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=amp_enabled, dtype=amp_dtype):
                    logits_u_w = interp(ema_model(inputs_u_w))
            logits_u_w, _ = weakTransform(getWeakInverseTransformParameters(weak_parameters), data = logits_u_w.detach())

            softmax_u_w = torch.softmax(logits_u_w.detach(), dim=1)
            max_probs, argmax_u_w = torch.max(softmax_u_w, dim=1)

            if mix_mask == "class":

                for image_i in range(batch_size):
                    classes = torch.unique(argmax_u_w[image_i])
                    classes = classes[classes != ignore_label]
                    nclasses = classes.shape[0]
                    classes = classes[torch.randperm(nclasses, device=classes.device)[:int((nclasses - nclasses % 2) / 2)]]
                    if image_i == 0:
                        MixMask = transformmasks.generate_class_mask(argmax_u_w[image_i], classes).unsqueeze(0).cuda()
                    else:
                        MixMask = torch.cat((MixMask,transformmasks.generate_class_mask(argmax_u_w[image_i], classes).unsqueeze(0).cuda()))

            elif mix_mask == 'cut':
                img_size = inputs_u_w.shape[2:4]
                for image_i in range(batch_size):
                    if image_i == 0:
                        MixMask = torch.from_numpy(transformmasks.generate_cutout_mask(img_size)).unsqueeze(0).cuda().float()
                    else:
                        MixMask = torch.cat((MixMask, torch.from_numpy(transformmasks.generate_cutout_mask(img_size)).unsqueeze(0).cuda().float()))

            elif mix_mask == "cow":
                img_size = inputs_u_w.shape[2:4]
                sigma_min = 8
                sigma_max = 32
                p_min = 0.5
                p_max = 0.5
                for image_i in range(batch_size):
                    sigma = np.exp(np.random.uniform(np.log(sigma_min), np.log(sigma_max)))     # Random sigma
                    p = np.random.uniform(p_min, p_max)     # Random p
                    if image_i == 0:
                        MixMask = torch.from_numpy(transformmasks.generate_cow_mask(img_size, sigma, p, seed=None)).unsqueeze(0).cuda().float()
                    else:
                        MixMask = torch.cat((MixMask,torch.from_numpy(transformmasks.generate_cow_mask(img_size, sigma, p, seed=None)).unsqueeze(0).cuda().float()))

            elif mix_mask == None:
                MixMask = torch.ones_like(inputs_u_w)

            strong_parameters = {"Mix": MixMask}
            if random_flip:
                strong_parameters["flip"] = random.randint(0, 1)
            else:
                strong_parameters["flip"] = 0
            if color_jitter:
                strong_parameters["ColorJitter"] = random.uniform(0, 1)
            else:
                strong_parameters["ColorJitter"] = 0
            if gaussian_blur:
                strong_parameters["GaussianBlur"] = random.uniform(0, 1)
            else:
                strong_parameters["GaussianBlur"] = 0

            inputs_u_s, _ = strongTransform(strong_parameters, data = images_remain)
            with torch.cuda.amp.autocast(enabled=amp_enabled, dtype=amp_dtype):
                logits_u_s = interp(model(inputs_u_s))

            softmax_u_w_mixed, _ = strongTransform(strong_parameters, data = softmax_u_w)
            max_probs, pseudo_label = torch.max(softmax_u_w_mixed, dim=1)

            if pixel_weight == "threshold_uniform":
                unlabeled_weight = max_probs.ge(0.968).float().sum() / pseudo_label.numel()
                pixelWiseWeight = torch.ones_like(max_probs) * unlabeled_weight
            elif pixel_weight == "threshold":
                pixelWiseWeight = max_probs.ge(0.968).float()
            elif pixel_weight == 'sigmoid':
                max_iter = 10000
                pixelWiseWeight = torch.ones_like(max_probs) * sigmoid_ramp_up(i_iter, max_iter)
            elif pixel_weight == False:
                pixelWiseWeight = torch.ones_like(max_probs)

            with torch.cuda.amp.autocast(enabled=amp_enabled, dtype=amp_dtype):
                if consistency_loss == 'CE':
                    L_u = consistency_weight * unlabeled_loss(logits_u_s, pseudo_label, pixelWiseWeight)
                elif consistency_loss == 'MSE':
                    unlabeled_weight = max_probs.ge(0.968).float().sum() / pseudo_label.numel()
                    #softmax_u_w_mixed = torch.cat((softmax_u_w_mixed[1].unsqueeze(0),softmax_u_w_mixed[0].unsqueeze(0)))
                    L_u = consistency_weight * unlabeled_weight * unlabeled_loss(logits_u_s, softmax_u_w_mixed)

                loss = L_l + L_u

        else:
            loss = L_l

        loss = loss.mean()
        loss_l_value += L_l.detach().float().mean().item()
        if train_unlabeled:
            loss_u_value += L_u.detach().float().mean().item()

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # update Mean teacher network
        if ema_model is not None:
            alpha_teacher = 0.99
            ema_model = update_ema_variables(ema_model = ema_model, model = model, alpha_teacher=alpha_teacher, iteration=i_iter)

        if is_main_process():
            if train_unlabeled:
                print('iter = {0:6d}/{1:6d}, loss_l = {2:.3f}, loss_u = {3:.3f}'.format(i_iter, num_iterations, loss_l_value, loss_u_value))
            else:
                print('iter = {0:6d}/{1:6d}, loss_l = {2:.3f}'.format(i_iter, num_iterations, loss_l_value))

        if i_iter % save_checkpoint_every == 0 and i_iter != 0:
            _save_checkpoint(i_iter, model, optimizer, config, ema_model)

        if use_tensorboard and is_main_process():
            if 'tensorboard_writer' not in locals():
                tensorboard_writer = tensorboard.SummaryWriter(log_dir, flush_secs=30)

            accumulated_loss_l.append(loss_l_value)
            if train_unlabeled:
                accumulated_loss_u.append(loss_u_value)
            if i_iter % log_per_iter == 0 and i_iter != 0:

                tensorboard_writer.add_scalar('Training/Supervised loss', np.mean(accumulated_loss_l), i_iter)
                accumulated_loss_l = []

                if train_unlabeled:
                    tensorboard_writer.add_scalar('Training/Unsupervised loss', np.mean(accumulated_loss_u), i_iter)
                    accumulated_loss_u = []


        if i_iter % val_per_iter == 0 and i_iter != 0:
            if is_main_process():
                model.eval()
                mIoU, eval_loss = evaluate(
                    unwrap_model(model),
                    dataset,
                    ignore_label=ignore_label,
                    input_size=input_size,
                    save_dir=checkpoint_dir,
                    val_num_workers=val_num_workers,
                    pin_memory=pin_memory,
                    prefetch_factor=prefetch_factor,
                    persistent_workers=persistent_workers,
                    amp_enabled=amp_enabled,
                    amp_dtype=amp_dtype,
                )

                model.train()

                if mIoU > best_mIoU and save_best_model:
                    best_mIoU = mIoU
                    _save_checkpoint(i_iter, model, optimizer, config, ema_model, save_best=True)

                if use_tensorboard:
                    tensorboard_writer.add_scalar('Validation/mIoU', mIoU, i_iter)
                    tensorboard_writer.add_scalar('Validation/Loss', eval_loss, i_iter)
            if distributed:
                dist.barrier()

        if save_unlabeled_images and train_unlabeled and is_main_process() and i_iter % save_checkpoint_every == 0:
            # Saves two mixed images and the corresponding prediction
            save_image(inputs_u_s[0].cpu(),i_iter,'input1',palette.CityScpates_palette)
            save_image(inputs_u_s[1].cpu(),i_iter,'input2',palette.CityScpates_palette)
            _, pred_u_s = torch.max(logits_u_s, dim=1)
            save_image(pred_u_s[0].cpu(),i_iter,'pred1',palette.CityScpates_palette)
            save_image(pred_u_s[1].cpu(),i_iter,'pred2',palette.CityScpates_palette)

    _save_checkpoint(num_iterations, model, optimizer, config, ema_model)

    if is_main_process():
        model.eval()
        mIoU, val_loss = evaluate(
            unwrap_model(model),
            dataset,
            ignore_label=ignore_label,
            input_size=input_size,
            save_dir=checkpoint_dir,
            val_num_workers=val_num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
        )

        model.train()
        if mIoU > best_mIoU and save_best_model:
            best_mIoU = mIoU
            _save_checkpoint(i_iter, model, optimizer, config, ema_model, save_best=True)

        if use_tensorboard:
            tensorboard_writer.add_scalar('Validation/mIoU', mIoU, i_iter)
            tensorboard_writer.add_scalar('Validation/Loss', val_loss, i_iter)

    end = timeit.default_timer()
    if is_main_process():
        print('Total time: ' + str(end-start) + ' seconds')
    cleanup_distributed()

if __name__ == '__main__':

    print('---------------------------------Starting---------------------------------')

    args = get_arguments()

    if args.resume:
        config = torch.load(args.resume)['config']
    else:
        config = json.load(open(args.config))

    model = config['model']
    dataset = config['dataset']

    if dataset == 'cityscapes':
        IMG_MEAN = np.flip(np.array([73.15835921, 82.90891754, 72.39239876]))
        num_classes = 19
        if config['training']['data']['split_id_list'] == 0:
            split_id = './splits/city/split_0.pkl'
        elif config['training']['data']['split_id_list'] == 1:
            split_id = './splits/city/split_1.pkl'
        elif config['training']['data']['split_id_list'] == 2:
            split_id = './splits/city/split_2.pkl'
        else:
            split_id = None

    elif dataset == 'pascal_voc':
        IMG_MEAN = np.flip(np.array([104.00698793,116.66876762,122.67891434]))
        num_classes = 21
        data_dir = './data/voc_dataset/'
        data_list_path = './data/voc_list/train_aug.txt'
        if config['training']['data']['split_id_list'] == 0:
            split_id = './splits/voc/split_0.pkl'
        else:
            split_id = None
    
    elif dataset == 'rescuenet':
        IMG_MEAN = np.flip(np.array([104.00698793,116.66876762,122.67891434]))
        num_classes = 11
        split_id = None
    
    elif dataset == 'floodnet':
        IMG_MEAN = np.flip(np.array([104.00698793,116.66876762,122.67891434]))
        num_classes = 10
        split_id = None

    if config['pretrained'] == 'coco':
        if os.path.exists(CLASSMIX_COCO_PRETRAIN):
            restore_from = CLASSMIX_COCO_PRETRAIN
        else:
            restore_from = 'http://vllab1.ucmerced.edu/~whung/adv-semi-seg/resnet101COCO-41f33a49.pth'
    elif config['pretrained'] in (None, '', 'none', 'None'):
        restore_from = None
    else:
        restore_from = config['pretrained']

    batch_size = config['training']['batch_size']
    num_iterations = config['training']['num_iterations']
    training_epochs = config['training'].get('epochs')

    input_size_string = config['training']['data']['input_size']
    h, w = map(int, input_size_string.split(','))
    input_size = (h, w)

    ignore_label = config['ignore_label'] # 255 for PASCAL-VOC / 250 for Cityscapes

    learning_rate = config['training']['learning_rate']

    optimizer_type = config['training']['optimizer']
    lr_schedule = config['training']['lr_schedule']
    lr_power = config['training']['lr_schedule_power']
    weight_decay = config['training']['weight_decay']
    momentum = config['training']['momentum']
    num_workers = config['training']['num_workers']
    unlabeled_num_workers = config['training'].get('unlabeled_num_workers', num_workers)
    val_num_workers = config['training'].get('val_num_workers', num_workers)
    pin_memory = config['training'].get('pin_memory', True)
    prefetch_factor = config['training'].get('prefetch_factor', 2)
    persistent_workers = config['training'].get('persistent_workers', True)
    use_sync_batchnorm = config['training']['use_sync_batchnorm']
    random_seed = config['seed']

    labeled_samples = config['training']['data']['labeled_samples']
    split_percent = config['training']['data'].get('split_percent')

    #unlabeled CONFIGURATIONS
    train_unlabeled = config['training']['unlabeled']['train_unlabeled']
    mix_mask = config['training']['unlabeled']['mix_mask']
    pixel_weight = config['training']['unlabeled']['pixel_weight']
    consistency_loss = config['training']['unlabeled']['consistency_loss']
    consistency_weight = config['training']['unlabeled']['consistency_weight']
    random_flip = config['training']['unlabeled']['flip']
    color_jitter = config['training']['unlabeled']['color_jitter']
    gaussian_blur = config['training']['unlabeled']['blur']

    random_scale = config['training']['data']['scale']
    random_crop = config['training']['data']['crop']

    save_checkpoint_every = config['utils']['save_checkpoint_every']
    if args.resume:
        checkpoint_dir = os.path.join(*args.resume.split('/')[:-1]) + '_resume-' + start_writeable
    else:
        checkpoint_dir = os.path.join(config['utils']['checkpoint_dir'], start_writeable + '-' + args.name)
    log_dir = checkpoint_dir

    val_per_iter = config['utils']['val_per_iter']
    use_tensorboard = config['utils']['tensorboard']
    log_per_iter = config['utils']['log_per_iter']

    save_best_model = config['utils']['save_best_model']
    distributed, local_rank, rank, world_size = setup_distributed()
    if args.gpus > 1 and not distributed:
        raise RuntimeError(
            'ClassMix now uses DistributedDataParallel. Launch multi-GPU runs with '
            '`torchrun --nproc_per_node=<num_gpus> ClassMix/trainSSL.py ...`.'
        )

    if args.save_images and is_main_process():
        print('Saving unlabeled images')
        save_unlabeled_images = True
    else:
        save_unlabeled_images = False

    gpus = (local_rank,)

    main()
