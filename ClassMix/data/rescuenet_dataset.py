import os
import os.path as osp
import numpy as np
import random
#import matplotlib.pyplot as plt
import collections
import torch
import torchvision
import cv2
from torch.utils import data
from PIL import Image

class RescueNetDataset(data.Dataset):
    def __init__(self, root, split="train", max_iters=None, crop_size=(750, 750), scale=True, mirror=True, ignore_label=255, unlabeled=False, split_percent=None):
        self.root = root
        self.crop_h, self.crop_w = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.is_mirror = mirror
        if split == "train":
            if split_percent is not None:
                list_path = f'./data/rescuenet/splits/{split_percent}/labeled.txt'
            else:
                list_path = './data/rescuenet/train-set/labeled.txt'
            if unlabeled:
                if split_percent is not None:
                    list_path = f'./data/rescuenet/splits/{split_percent}/unlabeled.txt'
                else:
                    list_path = './data/rescuenet/train-set/unlabeled.txt'
        elif split == "val":
            list_path =  './data/rescuenet/val.txt'
        self.samples = []
        with open(list_path) as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                if len(parts) == 1:
                    image_rel = f'train-set/train-org-img/{parts[0]}.jpg'
                    label_rel = f'train-set/train-label-img/{parts[0]}_lab.png'
                else:
                    image_rel, label_rel = parts[0], parts[1]
                self.samples.append((image_rel, label_rel))

        if not max_iters==None:
	        self.samples = self.samples * int(np.ceil(float(max_iters) / len(self.samples)))


        self.files = []
        # for split in ["train", "trainval", "val"]:
        for image_rel, label_rel in self.samples:
            self.files.append({
                "img": osp.join(self.root, image_rel),
                "label": osp.join(self.root, label_rel),
                "name": image_rel
            })

        IMG_MEAN = (104.00698793,116.66876762,122.67891434)
        self.mean = IMG_MEAN

        self.class_names = ['background', 'Water', 'Building_No_Damage', 'Building_Minor_Damage', 
            'Building_Major_Damage', 'Building_Total_Destruction', 'Vehicle', 'Road-Clear', 'Road-Blocked', 'Tree', 'Pool']

    def __len__(self):
        return len(self.files)

    def generate_scale_label(self, image, label):
        f_scale = 0.5 + random.randint(0, 11) / 10.0
        image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_LINEAR)
        label = cv2.resize(label, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_NEAREST)
        return image, label

    def __getitem__(self, index):
        datafiles = self.files[index]

        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        label = cv2.imread(datafiles["label"], cv2.IMREAD_GRAYSCALE)
        size = image.shape
        name = datafiles["name"]
        if self.scale:
            image, label = self.generate_scale_label(image, label)
        image = np.asarray(image, np.float32)
        image -= self.mean
        img_h, img_w = label.shape
        pad_h = max(self.crop_h - img_h, 0)
        pad_w = max(self.crop_w - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            img_pad = cv2.copyMakeBorder(image, 0, pad_h, 0,
                pad_w, cv2.BORDER_CONSTANT,
                value=(0.0, 0.0, 0.0))
            label_pad = cv2.copyMakeBorder(label, 0, pad_h, 0,
                pad_w, cv2.BORDER_CONSTANT,
                value=(self.ignore_label,))
        else:
            img_pad, label_pad = image, label

        img_h, img_w = label_pad.shape
        h_off = random.randint(0, img_h - self.crop_h)
        w_off = random.randint(0, img_w - self.crop_w)
        image = np.asarray(img_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        label = np.asarray(label_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.int64)
        image = image[:, :, ::-1]  # change to BGR
        image = image.transpose((2, 0, 1))
        if self.is_mirror:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            label = label[:, ::flip]

        return image.copy(), label.copy(), np.array(size), name, index

if __name__ == '__main__':
    dst = RescueNetDataset("../../../data/jpk322/RescueNet/")

    trainloader = data.DataLoader(dst, batch_size=4)
    # for i, data in enumerate(trainloader):
    #     imgs, labels, size, name, index = data
    #     if i == 0:
    #         img = torchvision.utils.make_grid(imgs).numpy()
    #         img = np.transpose(img, (1, 2, 0))
    #         img = img[:, :, ::-1]
    #         #plt.imshow(img)
    #         #plt.show()
