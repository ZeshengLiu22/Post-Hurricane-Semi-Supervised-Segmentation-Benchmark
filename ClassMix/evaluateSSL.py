import argparse
import scipy
from scipy import ndimage
import cv2
import numpy as np
import sys
from collections import OrderedDict
import os

import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
from torch.utils import data, model_zoo

from model.deeplabv2 import Res_Deeplab
from data.voc_dataset import VOCDataSet
from data import get_data_path, get_loader
import torchvision.transforms as transform
from torchvision import transforms

from PIL import Image
import scipy.misc
from utils.loss import CrossEntropy2d

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

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

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="SSL evaluation script")
    parser.add_argument("-m","--model-path", type=str, default=None, required=True,
                        help="Model to evaluate")
    parser.add_argument("--gpu", type=int, default=(0,),
                        help="choose gpu device.")
    parser.add_argument("--save-output-images", action="store_true",
                        help="save output images")
    return parser.parse_args()


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

class VOCColorize(object):
    def __init__(self, n=22):
        self.cmap = color_map(22)
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.shape
        color_image = np.zeros((3, size[0], size[1]), dtype=np.uint8)

        for label in range(0, len(self.cmap)):
            mask = (label == gray_image)
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        # handle void
        mask = (255 == gray_image)
        color_image[0][mask] = color_image[1][mask] = color_image[2][mask] = 255

        return color_image

def color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap

def get_label_vector(target, nclass):
    # target is a 3D Variable BxHxW, output is 2D BxnClass
    hist, _ = np.histogram(target, bins=nclass, range=(0, nclass-1))
    vect = hist>0
    vect_out = np.zeros((21,1))
    for i in range(len(vect)):
        if vect[i] == True:
            vect_out[i] = 1
        else:
            vect_out[i] = 0

    return vect_out

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
    return f"{value:.6f}" if np.isfinite(value) else "nan"


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


def get_iou(data_list, class_num, dataset, save_path=None):
    from multiprocessing import Pool
    from utils.metric import ConfusionMatrix

    ConfM = ConfusionMatrix(class_num)
    f = ConfM.generateM
    pool = Pool()
    m_list = pool.map(f, data_list)
    pool.close()
    pool.join()

    for m in m_list:
        ConfM.addM(m)


    aveJ, j_list, M = ConfM.jaccard()

        # Frequency Weighted IoU calculation
    total_pixels = np.sum(ConfM.M)
    class_pixel_counts = np.sum(ConfM.M, axis=1)  # ground truth total per class (row sum)
    IoU = j_list
    fwiou = np.sum((class_pixel_counts / total_pixels) * IoU)

    if dataset == 'pascal_voc':
        classes = np.array(('background',  # always index 0
            'aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair',
            'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train', 'tvmonitor'))
    elif dataset == 'cityscapes':
        classes = np.array(("road", "sidewalk",
            "building", "wall", "fence", "pole",
            "traffic_light", "traffic_sign", "vegetation",
            "terrain", "sky", "person", "rider",
            "car", "truck", "bus",
            "train", "motorcycle", "bicycle"))

    elif dataset== 'rescuenet':
        classes = np.array(('background', 'Water', 'Building_No_Damage', 'Building_Minor_Damage',
            'Building_Major_Damage', 'Building_Total_Destruction', 'Vehicle', 'Road-Clear', 'Road-Blocked', 'Tree', 'Pool'))

    elif dataset== 'floodnet':
        classes = np.array(('Background', 'Building-flooded', 'Building-non-flooded', 'Road-flooded',
            'Road-non-flooded', 'Water', 'Tree', 'Vehicle', 'Pool', 'Grass'))


    for i, iou in enumerate(j_list):
        print('class {:2d} {:12} IU {:.5}'.format(i, classes[i], j_list[i]))

    print('meanIOU: ' + str(aveJ) + '\n')
    print('FWIoU: ' + str(fwiou) + '\n')
    if save_path:
        with open(save_path, 'w') as f:
            for i, iou in enumerate(j_list):
                f.write('class {:2d} {:12} IU {:.25}'.format(i, classes[i], j_list[i]) + '\n')
            f.write('meanIOU: ' + str(aveJ) + '\n')
            f.write('FWIoU: ' + str(fwiou) + '\n')
    return aveJ


def get_iou_final(data_list, class_num, dataset, save_path=None, ignore_index=None):
    confusion = np.zeros((class_num, class_num), dtype=np.int64)
    per_image_mious = []
    per_image_mious_no_background = []

    for gt, pred in data_list:
        per_confusion = confusion_matrix_from_arrays(pred, gt, class_num, ignore_index)
        confusion += per_confusion

        per_metrics = compute_metrics_from_confusion(per_confusion)
        present = per_metrics["target_pixels"] > 0
        present_no_background = present.copy()
        if present_no_background.size:
            present_no_background[0] = False
        per_image_mious.append(nanmean_or_nan(per_metrics["iou"][present]))
        per_image_mious_no_background.append(nanmean_or_nan(per_metrics["iou"][present_no_background]))

    metrics = compute_metrics_from_confusion(confusion)
    j_list = metrics["iou"]
    aveJ = nanmean_or_nan(j_list)
    no_background = np.ones(class_num, dtype=bool)
    if class_num:
        no_background[0] = False
    mean_no_background = nanmean_or_nan(j_list[no_background])

    if dataset == 'pascal_voc':
        classes = np.array(('background',  # always index 0
            'aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair',
            'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train', 'tvmonitor'))
    elif dataset == 'cityscapes':
        classes = np.array(("road", "sidewalk",
            "building", "wall", "fence", "pole",
            "traffic_light", "traffic_sign", "vegetation",
            "terrain", "sky", "person", "rider",
            "car", "truck", "bus",
            "train", "motorcycle", "bicycle"))

    elif dataset== 'rescuenet':
        classes = np.array(('background', 'Water', 'Building_No_Damage', 'Building_Minor_Damage',
            'Building_Major_Damage', 'Building_Total_Destruction', 'Vehicle', 'Road-Clear', 'Road-Blocked', 'Tree', 'Pool'))

    elif dataset== 'floodnet':
        classes = np.array(('Background', 'Building-flooded', 'Building-non-flooded', 'Road-flooded',
            'Road-non-flooded', 'Water', 'Tree', 'Vehicle', 'Pool', 'Grass'))


    for i, iou in enumerate(j_list):
        print('class {:2d} {:12} IU {}'.format(i, classes[i], metric_string(iou)))

    print('meanIOU: ' + metric_string(aveJ))
    print('meanIOU_no_background: ' + metric_string(mean_no_background))
    print('FWIoU: ' + metric_string(metrics["FWIoU"]))
    print(
        'per-image meanIOU: mean={}, std={}'.format(
            metric_string(nanmean_or_nan(per_image_mious)),
            metric_string(nanstd_or_nan(per_image_mious)),
        )
    )
    print(
        'per-image meanIOU_no_background: mean={}, std={}\n'.format(
            metric_string(nanmean_or_nan(per_image_mious_no_background)),
            metric_string(nanstd_or_nan(per_image_mious_no_background)),
        )
    )
    if save_path:
        with open(save_path, 'w') as f:
            for i, iou in enumerate(j_list):
                f.write('class {:2d} {:12} IU {}'.format(i, classes[i], metric_string(iou)) + '\n')
            f.write('meanIOU: ' + metric_string(aveJ) + '\n')
            f.write('meanIOU_no_background: ' + metric_string(mean_no_background) + '\n')
            f.write('FWIoU: ' + metric_string(metrics["FWIoU"]) + '\n')
            f.write(
                'per-image meanIOU: mean={}, std={}\n'.format(
                    metric_string(nanmean_or_nan(per_image_mious)),
                    metric_string(nanstd_or_nan(per_image_mious)),
                )
            )
            f.write(
                'per-image meanIOU_no_background: mean={}, std={}\n'.format(
                    metric_string(nanmean_or_nan(per_image_mious_no_background)),
                    metric_string(nanstd_or_nan(per_image_mious_no_background)),
                )
            )
    return aveJ

def dataloader_kwargs(num_workers=0, pin_memory=True, prefetch_factor=2, persistent_workers=False):
    kwargs = {
        'num_workers': num_workers,
        'pin_memory': pin_memory,
    }
    if num_workers > 0:
        kwargs['prefetch_factor'] = prefetch_factor
        kwargs['persistent_workers'] = persistent_workers
    return kwargs


def evaluate(model, dataset, ignore_label=250, save_output_images=False, save_dir=None, input_size=(512,1024),
             val_num_workers=0, pin_memory=True, prefetch_factor=2, persistent_workers=False,
             amp_enabled=False, amp_dtype=torch.bfloat16, final_metrics=False):

    if dataset == 'pascal_voc':
        num_classes = 21
        input_size = (505, 505)
        data_loader = get_loader(dataset)
        data_path = get_data_path(dataset)
        test_dataset = data_loader(data_path, split="val", crop_size=input_size, scale=False, mirror=False)
        testloader = data.DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            **dataloader_kwargs(val_num_workers, pin_memory, prefetch_factor, persistent_workers),
        )
        interp = nn.Upsample(size=input_size, mode='bilinear', align_corners=True)

    elif dataset == 'cityscapes':
        num_classes = 19
        data_loader = get_loader('cityscapes')
        data_path = get_data_path('cityscapes')
        test_dataset = data_loader( data_path, img_size=input_size, is_transform=True, split='val')
        testloader = data.DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            **dataloader_kwargs(val_num_workers, pin_memory, prefetch_factor, persistent_workers),
        )
        interp = nn.Upsample(size=input_size, mode='bilinear', align_corners=True)
    
    elif dataset == 'rescuenet':
        num_classes = 11
        input_size = (750, 750)
        data_loader = get_loader('rescuenet')
        data_path = get_data_path('rescuenet')
        test_dataset = data_loader( data_path, crop_size=input_size, split='val')
        testloader = data.DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            **dataloader_kwargs(val_num_workers, pin_memory, prefetch_factor, persistent_workers),
        )
        interp = nn.Upsample(size=input_size, mode='bilinear', align_corners=True)
    
    elif dataset == 'floodnet':
        num_classes = 10
        input_size = (750, 750)
        data_loader = get_loader('floodnet')
        data_path = get_data_path('floodnet')
        test_dataset = data_loader(data_path, crop_size=input_size, split='val')
        testloader = data.DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            **dataloader_kwargs(val_num_workers, pin_memory, prefetch_factor, persistent_workers),
        )
        interp = nn.Upsample(size=input_size, mode='bilinear', align_corners=True)

    print('Evaluating, found ' + str(len(testloader)) + ' images.')

    data_list = []
    colorize = VOCColorize()
    criterion = CrossEntropy2d(ignore_label=ignore_label).cuda()

    total_loss = []

    for index, batch in enumerate(testloader):
        image, label, size, name, _ = batch
        size = size[0]

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=amp_enabled and torch.cuda.is_available(), dtype=amp_dtype):
                output  = model(Variable(image).cuda(non_blocking=True))
                output = interp(output)

                label_cuda = Variable(label.long()).cuda(non_blocking=True)
                loss = criterion(output, label_cuda)
            total_loss.append(loss.item())

            output = output.float().cpu().data[0].numpy()


            if dataset == 'pascal_voc':
                output = output[:,:size[0],:size[1]]
                gt = np.asarray(label[0].numpy()[:size[0],:size[1]], dtype=np.int)
            elif dataset == 'cityscapes':
                gt = np.asarray(label[0].numpy(), dtype=np.int)
            elif dataset == 'rescuenet':
                gt = np.asarray(label[0].numpy(), dtype=np.int)
            elif dataset == 'floodnet':
                gt = np.asarray(label[0].numpy(), dtype=np.int)

            output = output.transpose(1,2,0)
            output = np.asarray(np.argmax(output, axis=2), dtype=np.int)
            data_list.append([gt.reshape(-1), output.reshape(-1)])
            if save_output_images:
                if dataset == 'pascal_voc':
                    filename = os.path.join(save_dir, '{}.png'.format(name[0]))
                    color_file = Image.fromarray(colorize(output).transpose(1, 2, 0), 'RGB')
                    color_file.save(filename)
                if dataset == 'rescuenet':
                    filename = os.path.join(save_dir, '{}'.format(name[0]))
                    # print(filename)
                    # color_file = Image.fromarray(colorize(output).transpose(1, 2, 0), 'RGB')
                    color_mask = colorize_mask_fixed(output, 'RescueNet')
                    save_name = filename.replace('.jpg', '_pred.png')
                    color_mask.save(save_name)

                if dataset == 'floodnet':
                    filename = os.path.join(save_dir, '{}'.format(name[0]))
                    # color_file = Image.fromarray(output.transpose(1, 2, 0), 'RGB')
                    color_mask = colorize_mask_fixed(output, 'FloodNet')
                    save_name = filename.replace('.jpg', '_pred.png')
                    color_mask.save(save_name)
        if (index+1) % 10 == 0:
            print('%d processed'%(index+1))

    if save_dir:
        filename = os.path.join(save_dir, 'result.txt')
    else:
        filename = None
    if final_metrics:
        mIoU = get_iou_final(data_list, num_classes, dataset, filename, ignore_index=ignore_label)
    else:
        mIoU = get_iou(data_list, num_classes, dataset, filename)
    loss = np.mean(total_loss)
    return mIoU, loss

def main():
    """Create the model and start the evaluation process."""

    gpu0 = args.gpu

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model = Res_Deeplab(num_classes=num_classes)

    checkpoint = torch.load(args.model_path)
    state_dict = checkpoint['model']
    if any(key.startswith('module.') for key in state_dict.keys()):
        state_dict = {key.replace('module.', '', 1): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict)

    model.cuda()
    model.eval()
    
    loader_cfg = config.get('training', {})
    evaluate(
        model,
        dataset,
        ignore_label=ignore_label,
        save_output_images=args.save_output_images,
        save_dir=save_dir,
        input_size=input_size,
        val_num_workers=loader_cfg.get('val_num_workers', loader_cfg.get('num_workers', 0)),
        pin_memory=loader_cfg.get('pin_memory', True),
        prefetch_factor=loader_cfg.get('prefetch_factor', 2),
        persistent_workers=loader_cfg.get('persistent_workers', False),
        final_metrics=True,
    )


if __name__ == '__main__':
    args = get_arguments()

    config = torch.load(args.model_path)['config']

    dataset = config['dataset']

    if dataset == 'cityscapes':
        num_classes = 19
        input_size = (512,1024)
    elif dataset == 'pascal_voc':
        num_classes = 21
    elif dataset == 'rescuenet':
        num_classes = 11
        input_size = (750,750)
    elif dataset == 'floodnet':
        num_classes = 10
        input_size = (750,750)


    ignore_label = config['ignore_label']
    save_dir = os.path.join(*args.model_path.split('/')[:-1])

    main()
