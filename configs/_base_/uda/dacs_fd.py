# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
# Obtained from: https://github.com/lhoyer/DAFormer
# UDA with ImageNet Feature Distance
_base_ = ['dacs.py']
uda = dict(imnet_feature_dist_lambda=0.005, )
