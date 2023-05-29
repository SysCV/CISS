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
            ce_original=True,
            ce_stylized=False,
            inv=True,
        ),
        target = dict(
            ce=[],
            inv=[(('original', 'original'), ('stylized', 'stylized'))],
        )
    )
)
