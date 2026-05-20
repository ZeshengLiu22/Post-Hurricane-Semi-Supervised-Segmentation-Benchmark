"""
Dual-Teacher
Copyright (c) 2023-present NAVER Cloud Corp.
distributed under NVIDIA Source Code License for SegFormer
--------------------------------------------------------
References:
SegFormer: https://github.com/NVlabs/SegFormer
--------------------------------------------------------
"""
# dataset settings
dataset_type = 'FloodNetDataset'
data_root = '/data/users/zel220/dual_teacher_zesheng/data/floodnet'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (750, 750)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='Resize', img_scale=(2048, 512), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='MultiScaleFlipAug',
        img_scale=(750, 750),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            # Change keep_ratio from True to False
            dict(type='Resize', keep_ratio=False),
            dict(type='RandomFlip'),
             dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img', 'gt_semantic_seg']),
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=8,
    val_workers_per_gpu=8,
    pin_memory=True,
    prefetch_factor=2,
    persistent_workers=True,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/train',
        ann_dir='annotations/train',
        pipeline=train_pipeline),
    train_semi_l=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/train-org-img-l',
        ann_dir='annotations/train-label-img-l',
        pipeline=train_pipeline),
    train_semi_u=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/train-org-img-u',  # Assuming unlabeled images have the same directory
        ann_dir='annotations/train-label-img-u',  # Or use an empty annotation directory if there are no labels
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/val-org-img',
        ann_dir='annotations/val-label-img',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/val-org-img',
        ann_dir='annotations/val-label-img',
        pipeline=test_pipeline))
