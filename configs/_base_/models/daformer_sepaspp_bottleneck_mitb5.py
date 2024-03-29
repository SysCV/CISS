# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
# Obtained from: https://github.com/lhoyer/DAFormer
# Context only at F4 in Tab. 7

_base_ = ['daformer_conv1_mitb5.py']

norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    neck=dict(type='SegFormerAdapter', scales=[8]),
    decode_head=dict(
        decoder_params=dict(
            embed_neck_cfg=dict(
                _delete_=True,
                type='rawconv_and_aspp',
                kernel_size=1,
                sep=True,
                dilations=(1, 6, 12, 18),
                pool=False,
                act_cfg=dict(type='ReLU'),
                norm_cfg=norm_cfg))))
