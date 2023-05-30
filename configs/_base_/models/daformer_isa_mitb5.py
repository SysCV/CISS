# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
# Obtained from: https://github.com/lhoyer/DAFormer
# ISA Fusion in Tab. 7

_base_ = ['daformer_conv1_mitb5.py']

norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    decode_head=dict(
        decoder_params=dict(
            fusion_cfg=dict(
                _delete_=True,
                type='isa',
                isa_channels=256,
                key_query_num_convs=1,
                down_factor=(8, 8),
                act_cfg=dict(type='ReLU'),
                norm_cfg=norm_cfg))))
