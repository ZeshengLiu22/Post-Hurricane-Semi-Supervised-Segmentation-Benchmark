import copy
import math
import os
import os.path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

from . import augmentation as psp_trsform
from .base import BaseDataset


class flood_dset(BaseDataset):
    def __init__(self, data_root, data_list, trs_form):
        super(flood_dset, self).__init__(data_list)
        self.data_root = data_root
        self.transform = trs_form
        self.list_sample_new = self.list_sample # Always use full list

    def __getitem__(self, index):
        image_rel, label_rel = self.list_sample_new[index]
        image_name = os.path.splitext(os.path.basename(image_rel))[0]

        image_rel = os.path.join("org-img", image_rel)
        label_rel = os.path.join("label-img", label_rel)
        image_path = os.path.join(self.data_root, image_rel)
        label_path = os.path.join(self.data_root, label_rel)

        image = self.img_loader(image_path, "RGB")
        label = self.img_loader(label_path, "L")
        image, label = self.transform(image, label)

        if getattr(self, "return_name", False):
            return image[0], label[0, 0].long(), image_name
        else:
            return image[0], label[0, 0].long()

        def __len__(self):
            return len(self.list_sample_new)
    

def build_transfrom(cfg):
    trs_form = []
    mean, std, ignore_label = cfg["mean"], cfg["std"], cfg["ignore_label"]
    
    trs_form.append(psp_trsform.ToTensor())
    trs_form.append(psp_trsform.Normalize(mean=mean, std=std))
    
    if cfg.get("resize", False):
        trs_form.append(psp_trsform.Resize(cfg["resize"]))
        
    if cfg.get("rand_resize", False):
        trs_form.append(psp_trsform.RandResize(cfg["rand_resize"]))
        
    if cfg.get("rand_rotation", False):
        rand_rotation = cfg["rand_rotation"]
        trs_form.append(
            psp_trsform.RandRotate(rand_rotation, ignore_label=ignore_label)
        )
        
    if cfg.get("GaussianBlur", False) and cfg["GaussianBlur"]:
        trs_form.append(psp_trsform.RandomGaussianBlur())
        
    if cfg.get("flip", False) and cfg.get("flip"):
        trs_form.append(psp_trsform.RandomHorizontalFlip())
        
    if cfg.get("crop", False):
        crop_size, crop_type = cfg["crop"]["size"], cfg["crop"]["type"]
        trs_form.append(
            psp_trsform.Crop(crop_size, crop_type=crop_type, ignore_label=ignore_label)
        )
        
    if cfg.get("cutout", False):
        n_holes, length = cfg["cutout"]["n_holes"], cfg["cutout"]["length"]
        trs_form.append(psp_trsform.Cutout(n_holes=n_holes, length=length))
        
    if cfg.get("cutmix", False):
        n_holes, prop_range = cfg["cutmix"]["n_holes"], cfg["cutmix"]["prop_range"]
        trs_form.append(psp_trsform.Cutmix(prop_range=prop_range, n_holes=n_holes))

    return psp_trsform.Compose(trs_form)

def build_floodloader(split, all_cfg, seed=0):
    cfg_dset = all_cfg["dataset"]
    cfg = copy.deepcopy(cfg_dset)
    cfg.update(cfg.get(split, {}))

    batch_size = cfg.get("batch_size", 1)
    
    # build transform
    trs_form = build_transfrom(cfg)
    dset = flood_dset(cfg["data_root"], cfg["data_list"], trs_form)

    if split == "val":
        dset.return_name = True

    # build sampler
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        sample = DistributedSampler(dset)
    else:
        sample = None
    # sample = DistributedSampler(dset)

    loader = DataLoader(
        dset,
        batch_size=batch_size,
        sampler=sample,
        shuffle=False,
        **dataloader_kwargs(cfg, split),
    )
    return loader


def build_flood_semi_loader(split, all_cfg, seed=0):
    cfg_dset = all_cfg["dataset"]
    cfg = copy.deepcopy(cfg_dset)
    cfg.update(cfg.get(split, {}))

    batch_size = cfg.get("batch_size", 1)

    # build transform
    trs_form = build_transfrom(cfg)
    trs_form_unsup = build_transfrom(cfg)
    

    if split == "val":
        dset = flood_dset(cfg["data_root"], cfg["data_list"], trs_form)
        # build sampler
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            sample = DistributedSampler(dset)
        else:
            sample = None

        loader = DataLoader(
            dset,
            batch_size=batch_size,
            sampler=sample,
            shuffle=False,
            **dataloader_kwargs(cfg, split),
        )
        return loader

    # Semi-supervised training
    dset_sup = flood_dset(cfg["data_root"], cfg["data_list"], trs_form)
    
    # build sampler for unlabeled set
    if "labeled-" in cfg["data_list"]:
        data_list_unsup = cfg["data_list"].replace("labeled-", "unlabeled-")
    else:
        data_list_unsup = cfg["data_list"].replace("labeled.txt", "unlabeled.txt")
    dset_unsup = flood_dset(
            cfg["data_root"], data_list_unsup, trs_form_unsup)

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        sample_sup = DistributedSampler(dset_sup)
    else:
        sample_sup = None

    loader_sup = DataLoader(
            dset_sup,
            batch_size=batch_size,
            sampler=sample_sup,
            shuffle=False,
            drop_last=True,
            **dataloader_kwargs(cfg, split),
    )

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        sample_unsup = DistributedSampler(dset_unsup)
    else:
        sample_unsup = None
    loader_unsup = DataLoader(
            dset_unsup,
            batch_size=batch_size,
            sampler=sample_unsup,
            shuffle=False,
            drop_last=True,
            **dataloader_kwargs(cfg, split),
    )
    return loader_sup, loader_unsup


def dataloader_kwargs(cfg, split):
    if split == "val":
        workers = cfg.get("val_num_workers", cfg.get("val_workers", cfg.get("num_workers", cfg.get("workers", 2))))
    else:
        workers = cfg.get("num_workers", cfg.get("workers", 2))
    kwargs = {
        "num_workers": workers,
        "pin_memory": cfg.get("pin_memory", True),
    }
    if workers > 0:
        kwargs["prefetch_factor"] = cfg.get("prefetch_factor", 2)
        kwargs["persistent_workers"] = cfg.get("persistent_workers", True)
    return kwargs
