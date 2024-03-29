# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
# dataset settings
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (1024, 1024)
classes = ('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
            'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
            'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
            'bicycle')
palette = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
            [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
            [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
            [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
            [0, 80, 100], [0, 0, 230], [119, 11, 32]]
cityscapes_acdcref_train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True, keys=['source', 'target']),
    dict(type='LoadAnnotations', keys=['source']),
    dict(type='Resize', img_scale=(2048, 1024), keys=['source']),
    dict(type='Resize', img_scale=(1920, 1080), keys=['target']),
    dict(type='CentralCrop', crop_size=(1024, 1920), keys=['source', 'target']),
    dict(type='FDA', bandwidth=0.01, keys=[('source', 'target'), ('target', 'source')]),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75, keys=['source']),
    dict(type='RandomCrop', crop_size=crop_size, keys=['target']),
    dict(type='RandomFlip', prob=0.5, keys=['source']),
    dict(type='RandomFlip', prob=0.5, keys=['target']),
    dict(type='Normalize', **img_norm_cfg, keys=['source', 'target']),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255, keys=['source', 'target']),
    dict(type='DefaultFormatBundle', keys=['source', 'target']),
    dict(type='Collect', keys=[['img', 'img_stylized', 'gt_semantic_seg'], ['img', 'img_stylized']], parts=['source', 'target']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1920, 1080),
        # MultiScaleFlipAug is disabled by not providing img_ratios and
        # setting flip=False
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=[['img']]),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=32,
    train=dict(
        type='UDADatasetDual',
        source='Cityscapes',
        target='ACDCref',
        img_dir_source='leftImg8bit/train',
        img_dir_target='rgb_anon/train',
        img_suffix_source='_leftImg8bit.png',
        img_suffix_target='_rgb_ref_anon.png',
        ann_dir_source='gtFine/train',
        ann_dir_target='gt/train',
        seg_map_suffix_source='_gtFine_labelTrainIds.png',
        seg_map_suffix_target='_gtFine_labelTrainIds.png',
        data_root_source='data/cityscapes/',
        data_root_target='data/acdcref/',
        classes=classes,
        palette=palette,
        pipeline=cityscapes_acdcref_train_pipeline),
    val=dict(
        type='ACDCrefDataset',
        data_root='data/acdcref/',
        img_dir='rgb_anon/val',
        ann_dir='gt/val',
        pipeline=test_pipeline),
    test=dict(
        type='ACDCrefDataset',
        data_root='data/acdcref/',
        img_dir='rgb_anon/val',
        ann_dir='gt/val',
        pipeline=test_pipeline))
