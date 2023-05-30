# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
# uda settings
uda = dict(
    type='AdvSeg',
    discriminator_type='LS',
    lr_D=1e-4,
    lr_D_power=0.9,
    lr_D_min=0,
    lambda_adv_target=dict(main=0.001, aux=0.0002),
    debug_img_interval=1000)
