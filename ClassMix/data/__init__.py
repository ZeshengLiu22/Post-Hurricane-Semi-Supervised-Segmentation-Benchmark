import json

from data.base import *
from data.cityscapes_loader import cityscapesLoader
from data.voc_dataset import VOCDataSet
from data.rescuenet_dataset import RescueNetDataset
from data.floodnet_dataset import FloodNetDataset

def get_loader(name):
    """get_loader
    :param name:
    """
    return {
        "cityscapes": cityscapesLoader,
        "pascal_voc": VOCDataSet,
        "rescuenet": RescueNetDataset,
        "floodnet": FloodNetDataset
    }[name]

def get_data_path(name):
    """get_data_path
    :param name:
    :param config_file:
    """
    if name == 'cityscapes':
        return '../data/CityScapes/'
    if name == 'pascal_voc':
        return '../data/VOC2012/'
    if name == 'rescuenet':
        return './dataset/rescuenet/'
    if name == 'floodnet':
        return './dataset/floodnet/'
