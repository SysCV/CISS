# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
# Obtained from: https://github.com/lhoyer/DAFormer
# UDA with Thing-Class ImageNet Feature Distance + Increased Alpha
_base_ = ['dacs.py']
uda = dict(
    type='DACSCISS',
    alpha=0.999,
    imnet_feature_dist_lambda=0.005,
    imnet_feature_dist_classes=[6, 7, 11, 12, 13, 14, 15, 16, 17, 18],
    imnet_feature_dist_scale_min_ratio=0.75,
    stylize = dict(
        source = dict(
            ce_original=False,
            ce_stylized=True,
            inv=True,
        ),
        target = dict(
            ce=[('stylized', 'stylized')],
            inv=[(('original', 'original'), ('stylized', 'stylized'))],
            pseudolabels='stylized',
        ),
        inv_loss = dict(
            weight = 10.0,
        )
    )
)
