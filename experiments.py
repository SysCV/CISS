# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
# Adapted from: https://github.com/lhoyer/DAFormer

import itertools
import os

from mmcv import Config

# flake8: noqa


def get_model_base(architecture, backbone):
    architecture = architecture.replace('sfa_', '')
    for j in range(1, 100):
        hrda_name = [e for e in architecture.split('_') if f'hrda{j}' in e]
        for n in hrda_name:
            architecture = architecture.replace(f'{n}_', '')
    architecture = architecture.replace('_nodbn', '')
    if 'segformer' in architecture:
        return {
            'mitb5': f'_base_/models/{architecture}_b5.py',
            # It's intended that <=b4 refers to b5 config
            'mitb4': f'_base_/models/{architecture}_b5.py',
            'mitb3': f'_base_/models/{architecture}_b5.py',
            'r101v1c': f'_base_/models/{architecture}_r101.py',
        }[backbone]
    if 'daformer_' in architecture and 'mitb5' in backbone:
        return f'_base_/models/{architecture}_mitb5.py'
    if 'upernet' in architecture and 'mit' in backbone:
        return f'_base_/models/{architecture}_mit.py'
    assert 'mit' not in backbone or '-del' in backbone
    return {
        'dlv2': '_base_/models/deeplabv2_r50-d8.py',
        'dlv2red': '_base_/models/deeplabv2red_r50-d8.py',
        'dlv3p': '_base_/models/deeplabv3plus_r50-d8.py',
        'da': '_base_/models/danet_r50-d8.py',
        'isa': '_base_/models/isanet_r50-d8.py',
        'uper': '_base_/models/upernet_r50.py',
    }[architecture]


def get_pretraining_file(backbone):
    if 'mitb5' in backbone:
        return 'pretrained/mit_b5.pth'
    if 'mitb4' in backbone:
        return 'pretrained/mit_b4.pth'
    if 'mitb3' in backbone:
        return 'pretrained/mit_b3.pth'
    if 'r101v1c' in backbone:
        return 'open-mmlab://resnet101_v1c'
    return {
        'r50v1c': 'open-mmlab://resnet50_v1c',
        'x50-32': 'open-mmlab://resnext50_32x4d',
        'x101-32': 'open-mmlab://resnext101_32x4d',
        's50': 'open-mmlab://resnest50',
        's101': 'open-mmlab://resnest101',
        's200': 'open-mmlab://resnest200',
    }[backbone]


def get_backbone_cfg(backbone):
    for i in [1, 2, 3, 4, 5]:
        if backbone == f'mitb{i}':
            return dict(type=f'mit_b{i}')
        if backbone == f'mitb{i}-del':
            return dict(_delete_=True, type=f'mit_b{i}')
    return {
        'r50v1c': {
            'depth': 50
        },
        'r101v1c': {
            'depth': 101
        },
        'x50-32': {
            'type': 'ResNeXt',
            'depth': 50,
            'groups': 32,
            'base_width': 4,
        },
        'x101-32': {
            'type': 'ResNeXt',
            'depth': 101,
            'groups': 32,
            'base_width': 4,
        },
        's50': {
            'type': 'ResNeSt',
            'depth': 50,
            'stem_channels': 64,
            'radix': 2,
            'reduction_factor': 4,
            'avg_down_stride': True
        },
        's101': {
            'type': 'ResNeSt',
            'depth': 101,
            'stem_channels': 128,
            'radix': 2,
            'reduction_factor': 4,
            'avg_down_stride': True
        },
        's200': {
            'type': 'ResNeSt',
            'depth': 200,
            'stem_channels': 128,
            'radix': 2,
            'reduction_factor': 4,
            'avg_down_stride': True,
        },
    }[backbone]


def update_decoder_in_channels(cfg, architecture, backbone):
    cfg.setdefault('model', {}).setdefault('decode_head', {})
    if 'dlv3p' in architecture and 'mit' in backbone:
        cfg['model']['decode_head']['c1_in_channels'] = 64
    if 'sfa' in architecture:
        cfg['model']['decode_head']['in_channels'] = 512
    return cfg


def setup_rcs(cfg, temperature, min_crop_ratio):
    cfg.setdefault('data', {}).setdefault('train', {})
    cfg['data']['train']['rare_class_sampling'] = dict(
        min_pixels=3000, class_temp=temperature, min_crop_ratio=min_crop_ratio)
    return cfg


def generate_experiment_cfgs(id):

    def config_from_vars():
        cfg = {
            '_base_': ['_base_/default_runtime.py'],
            'gpu_model': gpu_model,
            'n_gpus': n_gpus
        }
        if seed is not None:
            cfg['seed'] = seed
        if launcher is not None:
            cfg['launcher'] = launcher

        # Setup model config
        architecture_mod = architecture
        sync_crop_size_mod = sync_crop_size
        inference_mod = inference
        model_base = get_model_base(architecture_mod, backbone)
        model_base_cfg = Config.fromfile(os.path.join('configs', model_base))
        cfg['_base_'].append(model_base)
        cfg['model'] = {
            'pretrained': get_pretraining_file(backbone),
            'backbone': get_backbone_cfg(backbone),
        }
        if 'sfa_' in architecture_mod:
            cfg['model']['neck'] = dict(type='SegFormerAdapter')
        if '_nodbn' in architecture_mod:
            cfg['model'].setdefault('decode_head', {})
            cfg['model']['decode_head']['norm_cfg'] = None
        cfg = update_decoder_in_channels(cfg, architecture_mod, backbone)

        hrda_ablation_opts = None
        outer_crop_size = sync_crop_size_mod \
            if sync_crop_size_mod is not None \
            else (int(crop.split('x')[0]), int(crop.split('x')[1]))
        if 'hrda1' in architecture_mod:
            o = [e for e in architecture_mod.split('_') if 'hrda' in e][0]
            hr_crop_size = (int((o.split('-')[1])), int((o.split('-')[1])))
            hr_loss_w = float(o.split('-')[2])
            hrda_ablation_opts = o.split('-')[3:]
            cfg['model']['type'] = 'HRDAEncoderDecoder'
            cfg['model']['scales'] = [1, 0.5]
            cfg['model'].setdefault('decode_head', {})
            cfg['model']['decode_head']['single_scale_head'] = model_base_cfg[
                'model']['decode_head']['type']
            cfg['model']['decode_head']['type'] = 'HRDAHead'
            cfg['model']['hr_crop_size'] = hr_crop_size
            cfg['model']['feature_scale'] = 0.5
            cfg['model']['crop_coord_divisible'] = 8
            cfg['model']['hr_slide_inference'] = True
            cfg['model']['decode_head']['attention_classwise'] = True
            cfg['model']['decode_head']['hr_loss_weight'] = hr_loss_w
            if outer_crop_size == hr_crop_size:
                # If the hr crop is smaller than the lr crop (hr_crop_size <
                # outer_crop_size), there is direct supervision for the lr
                # prediction as it is not fused in the region without hr
                # prediction. Therefore, there is no need for a separate
                # lr_loss.
                cfg['model']['decode_head']['lr_loss_weight'] = hr_loss_w
                # If the hr crop covers the full lr crop region, calculating
                # the FD loss on both scales stabilizes the training for
                # difficult classes.
                cfg['model']['feature_scale'] = 'all' if '_fd' in uda else 0.5

        # HRDA Ablations
        if hrda_ablation_opts is not None:
            for o in hrda_ablation_opts:
                if o == 'fixedatt':
                    # Average the predictions from both scales instead of
                    # learning a scale attention.
                    cfg['model']['decode_head']['fixed_attention'] = 0.5
                elif o == 'nooverlap':
                    # Don't use overlapping slide inference for the hr
                    # prediction.
                    cfg['model']['hr_slide_overlapping'] = False
                elif o == 'singleatt':
                    # Use the same scale attention for all class channels.
                    cfg['model']['decode_head']['attention_classwise'] = False
                elif o == 'blurhr':
                    # Use an upsampled lr crop (blurred) for the hr crop
                    cfg['model']['blur_hr_crop'] = True
                elif o == 'samescale':
                    # Use the same scale/resolution for both crops.
                    cfg['model']['scales'] = [1, 1]
                    cfg['model']['feature_scale'] = 1
                elif o[:2] == 'sc':
                    cfg['model']['scales'] = [1, float(o[2:])]
                    if not isinstance(cfg['model']['feature_scale'], str):
                        cfg['model']['feature_scale'] = float(o[2:])
                else:
                    raise NotImplementedError(o)

        # Setup inference mode
        if inference_mod == 'whole' or crop == '2048x1024':
            assert model_base_cfg['model']['test_cfg']['mode'] == 'whole'
        elif inference_mod == 'slide':
            cfg['model'].setdefault('test_cfg', {})
            cfg['model']['test_cfg']['mode'] = 'slide'
            cfg['model']['test_cfg']['batched_slide'] = True
            crsize = sync_crop_size_mod if sync_crop_size_mod is not None \
                else [int(e) for e in crop.split('x')]
            cfg['model']['test_cfg']['stride'] = [e // 2 for e in crsize]
            cfg['model']['test_cfg']['crop_size'] = crsize
            architecture_mod += '_sl'
        else:
            raise NotImplementedError(inference_mod)

        # Setup UDA config
        if uda == 'target-only':
            cfg['_base_'].append(f'_base_/datasets/{target}_{crop}.py')
        elif uda == 'source-only':
            cfg['_base_'].append(
                f'_base_/datasets/{source}_to_{target}_{crop}.py')
        else:
            if stylization is not None:
                cfg['_base_'].append(
                    f'_base_/datasets/uda_{source}_to_{target}_{crop}_{stylization}.py')
            else:
                cfg['_base_'].append(
                    f'_base_/datasets/uda_{source}_to_{target}_{crop}.py')
            cfg['_base_'].append(f'_base_/uda/{uda}.py')
        cfg['data'] = dict(
            samples_per_gpu=batch_size,
            workers_per_gpu=workers_per_gpu,
            train={})
        # DAFormer legacy cropping that only works properly if the training
        # crop has the height of the (resized) target image.
        if 'dacs' in uda and plcrop in [True, 'v1']:
            cfg.setdefault('uda', {})
            cfg['uda']['pseudo_weight_ignore_top'] = 15
            cfg['uda']['pseudo_weight_ignore_bottom'] = 120
        # Generate mask of the pseudo-label margins in the data loader before
        # the image itself is cropped to ensure that the pseudo-label margins
        # are only masked out if the training crop is at the periphery of the
        # image.
        if 'dacs' in uda and plcrop == 'v2':
            if stylization is not None:
                cfg['data']['train']['crop_pseudo_margins_target'] = [30, 240, 30, 30]
            else:
                cfg['data']['train'].setdefault('target', {})
                cfg['data']['train']['target']['crop_pseudo_margins'] = \
                    [30, 240, 30, 30]
        if 'dacs' in uda and rcs_T is not None:
            cfg = setup_rcs(cfg, rcs_T, rcs_min_crop)
        if 'dacs' in uda and sync_crop_size_mod is not None:
            cfg.setdefault('data', {}).setdefault('train', {})
            cfg['data']['train']['sync_crop_size'] = sync_crop_size_mod
        if stylization is not None:
            cfg.setdefault('uda', {})
            cfg['uda']['stylization'] = stylization
        if inv_loss_weight is not None:
            cfg.setdefault('uda', {}).setdefault('stylize', {}).setdefault('inv_loss', {})
            cfg['uda']['stylize']['inv_loss']['weight'] = inv_loss_weight
        if inv_loss_weight_target is not None:
            cfg.setdefault('uda', {}).setdefault('stylize', {}).setdefault('inv_loss', {})
            cfg['uda']['stylize']['inv_loss']['weight_target'] = inv_loss_weight_target
        if inv_loss_norm is not None:
            cfg.setdefault('uda', {}).setdefault('stylize', {}).setdefault('inv_loss', {})
            cfg['uda']['stylize']['inv_loss']['norm'] = inv_loss_norm

        # Setup data root directories.
        if os.environ.get('DIR_SOURCE_DATASET') is not None:
            data_root_source = os.environ['DIR_SOURCE_DATASET'] + os.sep
            cfg.setdefault('data', {}).setdefault('train', {})
            if stylization is not None:
                cfg['data']['train']['data_root_source'] = data_root_source
            else:
                cfg['data']['train'].setdefault('source', {})
                cfg['data']['train']['source']['data_root'] = data_root_source
        if os.environ.get('DIR_TARGET_DATASET') is not None:
            data_root_target = os.environ['DIR_TARGET_DATASET'] + os.sep
            cfg.setdefault('data', {}).setdefault('train', {})
            if stylization is not None:
                cfg['data']['train']['data_root_target'] = data_root_target
            else:
                cfg['data']['train'].setdefault('target', {})
                cfg['data']['train']['target']['data_root'] = data_root_target
            cfg.setdefault('data', {}).setdefault('val', {})
            cfg['data']['val']['data_root'] = data_root_target
            cfg.setdefault('data', {}).setdefault('test', {})
            cfg['data']['test']['data_root'] = data_root_target
        
        # Setup optimizer and schedule
        if 'dacs' in uda or 'minent' in uda or 'advseg' in uda:
            cfg['optimizer_config'] = None  # Don't use outer optimizer

        cfg['_base_'].extend(
            [f'_base_/schedules/{opt}.py', f'_base_/schedules/{schedule}.py'])
        cfg['optimizer'] = {'lr': lr}
        cfg['optimizer'].setdefault('paramwise_cfg', {})
        cfg['optimizer']['paramwise_cfg'].setdefault('custom_keys', {})
        opt_param_cfg = cfg['optimizer']['paramwise_cfg']['custom_keys']
        if pmult:
            opt_param_cfg['head'] = dict(lr_mult=10.)
        if 'mit' in backbone:
            opt_param_cfg['pos_block'] = dict(decay_mult=0.)
            opt_param_cfg['norm'] = dict(decay_mult=0.)

        # Setup runner
        cfg['runner'] = dict(type='IterBasedRunner', max_iters=iters)
        cfg['checkpoint_config'] = dict(
            by_epoch=False, interval=iters // 10, max_keep_ckpts=1)
        cfg['evaluation'] = dict(interval=iters // 10, metric='mIoU', distributed_eval=distributed_eval, pre_eval=pre_eval)

        # Construct config name
        uda_mod = uda
        if 'dacs' in uda and rcs_T is not None:
            uda_mod += f'_rcs{rcs_T}'
            if rcs_min_crop != 0.5:
                uda_mod += f'-{rcs_min_crop}'
        if 'dacs' in uda and sync_crop_size_mod is not None:
            uda_mod += f'_sf{sync_crop_size_mod[0]}x{sync_crop_size_mod[1]}'
        if 'dacs' in uda:
            if not plcrop:
                pass
            elif plcrop in [True, 'v1']:
                uda_mod += '_cpl'
            elif plcrop[0] == 'v':
                uda_mod += f'_cpl{plcrop[1:]}'
            else:
                raise NotImplementedError(plcrop)
        crop_name = f'_{crop}' if crop != '512x512' else ''
        cfg['name'] = f'{source}2{target}{crop_name}_{uda_mod}_' \
                      f'{architecture_mod}_{backbone}_{schedule}'
        if opt != 'adamw':
            cfg['name'] += f'_{opt}'
        if lr != 0.00006:
            cfg['name'] += f'_{lr}'
        if not pmult:
            cfg['name'] += f'_pm{pmult}'
        cfg['exp'] = id
        cfg['name_dataset'] = f'{source}2{target}{crop_name}'
        cfg['name_architecture'] = f'{architecture_mod}_{backbone}'
        cfg['name_encoder'] = backbone
        cfg['name_decoder'] = architecture_mod
        cfg['name_uda'] = uda_mod
        cfg['name_opt'] = f'{opt}_{lr}_pm{pmult}_{schedule}' \
                          f'_{n_gpus}x{batch_size}_{iters // 1000}k'
        if seed is not None:
            cfg['name'] += f'_s{seed}'
        cfg['name'] = cfg['name'].replace('.', '').replace('True', 'T') \
            .replace('False', 'F').replace('cityscapes', 'cs') \
            .replace('synthia', 'syn') \
            .replace('darkzurich', 'dzur')
        return cfg

    # -------------------------------------------------------------------------
    # Set some defaults
    # -------------------------------------------------------------------------
    cfgs = []
    n_gpus = 1
    launcher = None
    distributed_eval = False
    pre_eval = False
    batch_size = 2
    iters = 40000
    opt, lr, schedule, pmult = 'adamw', 0.00006, 'poly10warm', True
    crop = '512x512'
    gpu_model = 'NVIDIAGeForceRTX2080Ti'
    datasets = [
        ('gta', 'cityscapes'),
    ]
    stylization = None
    architecture = None
    ciss_config = None
    inv_loss_weight = None
    inv_loss_weight_target = None
    inv_loss_norm = None
    workers_per_gpu = 1
    rcs_T = None
    rcs_min_crop = 0.5
    plcrop = False
    inference = 'whole'
    sync_crop_size = None
    # ----------------------------------------
    # Table 1: Final CISS on Cityscapes->ACDC.
    # ----------------------------------------
    if id == 1:
        seeds = [0, 1, 2]
        #         source,          target,         crop,        rcs_min_crop
        cs2acdc = ('cityscapesHR', 'acdcHR',       '1024x1024', 0.5 * (2 ** 2))
        stylization = 'fda'
        dec, backbone = 'daformer_sepaspp', 'mitb5'
        uda, rcs_T, plcrop = 'dacs_a999_fdthings_ciss_src_ceorig_inv_trg_ceorigorig_invorigorigstylizedstylized', 0.01, False
        inv_loss_weight = 50.0
        inv_loss_weight_target = 20.0
        inference = 'slide'
        workers_per_gpu = 16
        for dataset, architecture, sync_crop_size in [
            (cs2acdc, f'hrda1-512-0.1_{dec}', None),
        ]:
            for seed in seeds:
                source, target, crop, rcs_min_crop = dataset
                gpu_model = 'NVIDIATITANRTX'
                cfg = config_from_vars()
                cfgs.append(cfg)
    # --------------------------------------------------
    # Table 5, row 1: HRDA baseline on Cityscapes->ACDC.
    # --------------------------------------------------
    elif id == 50:
        seeds = [0, 1, 2]
        #         source,          target,         crop,        rcs_min_crop
        cs2acdc = ('cityscapesHR', 'acdcHR',       '1024x1024', 0.5 * (2 ** 2))
        dec, backbone = 'daformer_sepaspp', 'mitb5'
        # Use plcrop=False as ACDC has no rectification
        # artifacts in contrast to Cityscapes.
        uda, rcs_T, plcrop = 'dacs_a999_fdthings', 0.01, False
        inference = 'slide'
        for dataset, architecture, sync_crop_size in [
            (cs2acdc, f'hrda1-512-0.1_{dec}', None),
        ]:
            for seed in seeds:
                source, target, crop, rcs_min_crop = dataset
                gpu_model = 'NVIDIATITANRTX'
                cfg = config_from_vars()
                cfgs.append(cfg)
    # -------------------------------------------------------------------------
    # Table 5, row 2: FDA baseline on Cityscapes -> ACDC.
    # -------------------------------------------------------------------------
    elif id == 51:
        seeds = [0, 1, 2]
        #         source,          target,         crop,        rcs_min_crop
        cs2acdc = ('cityscapesHR', 'acdcHR',       '1024x1024', 0.5 * (2 ** 2))
        stylization = 'fda'
        dec, backbone = 'daformer_sepaspp', 'mitb5'
        uda, rcs_T, plcrop = 'dacs_a999_fdthings_ciss_src_cestylized', 0.01, False
        inference = 'slide'
        workers_per_gpu = 32
        for dataset, architecture, sync_crop_size in [
            (cs2acdc, f'hrda1-512-0.1_{dec}', None),
        ]:
            for seed in seeds:
                source, target, crop, rcs_min_crop = dataset
                gpu_model = 'NVIDIATITANRTX'
                cfg = config_from_vars()
                cfgs.append(cfg)
    # -----------------------------------------------------------------------------------------------------------------------
    # Table 6: CISS ablation study on invariance loss weights in source domain: CE stylized -> N, CE original -> Y, Inv -> Y.
    # -----------------------------------------------------------------------------------------------------------------------
    elif id == 129:
        seeds = [0, 1, 2]
        #         source,          target,         crop,        rcs_min_crop
        cs2acdc = ('cityscapesHR', 'acdcHR',       '1024x1024', 0.5 * (2 ** 2))
        stylization = 'fda'
        dec, backbone = 'daformer_sepaspp', 'mitb5'
        uda, rcs_T, plcrop = 'dacs_a999_fdthings_ciss_src_ceorig_inv', 0.01, False
        inference = 'slide'
        workers_per_gpu = 16
        for dataset, architecture, sync_crop_size in [
            (cs2acdc, f'hrda1-512-0.1_{dec}', None),
        ]:
            for seed in seeds:
                for inv_loss_weight in [
                    50.0,
                    100.0,
                    200.0,
                    500.0,
                    1000.0,
                ]:
                    source, target, crop, rcs_min_crop = dataset
                    gpu_model = 'NVIDIATITANRTX'
                    cfg = config_from_vars()
                    cfgs.append(cfg)
    # -----------------------------------------------------------------------------------------------------------------------
    # Table 6: CISS ablation study on invariance loss weights in target domain: CE stylized -> N, CE original -> Y, Inv -> Y.
    # -----------------------------------------------------------------------------------------------------------------------
    elif id == 130:
        seeds = [0, 1, 2]
        #         source,          target,         crop,        rcs_min_crop
        cs2acdc = ('cityscapesHR', 'acdcHR',       '1024x1024', 0.5 * (2 ** 2))
        stylization = 'fda'
        dec, backbone = 'daformer_sepaspp', 'mitb5'
        uda, rcs_T, plcrop = 'dacs_a999_fdthings_ciss_src_ceorig_trg_ceorigorig_invorigorigstylizedstylized', 0.01, False
        inference = 'slide'
        workers_per_gpu = 16
        for dataset, architecture, sync_crop_size in [
            (cs2acdc, f'hrda1-512-0.1_{dec}', None),
        ]:
            for seed in seeds:
                for inv_loss_weight in [
                    20.0,
                    50.0,
                    100.0,
                    200.0,
                    500.0,
                ]:
                    source, target, crop, rcs_min_crop = dataset
                    gpu_model = 'NVIDIATITANRTX'
                    cfg = config_from_vars()
                    cfgs.append(cfg)
    # ---------------------------------------------------------------------------------------------------------------------------------
    # Table 5, row 6: CISS ablation study on target domain with CE orig + Inv on source.
    # ---------------------------------------------------------------------------------------------------------------------------------
    elif id == 133:
        seeds = [0, 1, 2]
        #         source,          target,         crop,        rcs_min_crop
        cs2acdc = ('cityscapesHR', 'acdcHR',       '1024x1024', 0.5 * (2 ** 2))
        stylization = 'fda'
        dec, backbone = 'daformer_sepaspp', 'mitb5'
        uda, rcs_T, plcrop = 'dacs_a999_fdthings_ciss_src_ceorig_inv_trg_ceorigorig_cestylizedstylized', 0.01, False
        inference = 'slide'
        workers_per_gpu = 16
        for dataset, architecture, sync_crop_size in [
            (cs2acdc, f'hrda1-512-0.1_{dec}', None),
        ]:
            for seed in seeds:
                for inv_loss_weight in [
                    200.0,
                ]:
                    source, target, crop, rcs_min_crop = dataset
                    gpu_model = 'NVIDIATITANRTX'
                    cfg = config_from_vars()
                    cfgs.append(cfg)
    # ----------------------------------------------------------------------------------------
    # Table 5, row 3: CISS ablation study on source domain.
    # ----------------------------------------------------------------------------------------
    elif id == 134:
        seeds = [0, 1, 2]
        #         source,          target,         crop,        rcs_min_crop
        cs2acdc = ('cityscapesHR', 'acdcHR',       '1024x1024', 0.5 * (2 ** 2))
        stylization = 'fda'
        dec, backbone = 'daformer_sepaspp', 'mitb5'
        uda, rcs_T, plcrop = 'dacs_a999_fdthings_ciss_src_cestylized_ceorig', 0.01, False
        inference = 'slide'
        workers_per_gpu = 16
        for dataset, architecture, sync_crop_size in [
            (cs2acdc, f'hrda1-512-0.1_{dec}', None),
        ]:
            for seed in seeds:
                source, target, crop, rcs_min_crop = dataset
                gpu_model = 'NVIDIATITANRTX'
                cfg = config_from_vars()
                cfgs.append(cfg)
    # ---------------------------------------------------------------------------------
    # Table 3: Final CISS on Cityscapes -> Dark Zurich - lambda_s = 100, lambda_t = 50.
    # ---------------------------------------------------------------------------------
    elif id == 135:
        seeds = [0, 1, 2]
        #         source,          target,         crop,        rcs_min_crop
        cs2acdc = ('cityscapesHR', 'darkzurichHR', '1024x1024', 0.5 * (2 ** 2))
        stylization = 'fda'
        dec, backbone = 'daformer_sepaspp', 'mitb5'
        uda, rcs_T, plcrop = 'dacs_a999_fdthings_ciss_src_ceorig_inv_trg_ceorigorig_invorigorigstylizedstylized', 0.01, False
        inference = 'slide'
        workers_per_gpu = 16
        for dataset, architecture, sync_crop_size in [
            (cs2acdc, f'hrda1-512-0.1_{dec}', None),
        ]:
            for (inv_loss_weight, inv_loss_weight_target) in [
                (100.0, 50.0),
            ]:
                for seed in seeds:
                    source, target, crop, rcs_min_crop = dataset
                    gpu_model = 'NVIDIATITANRTX'
                    cfg = config_from_vars()
                    cfgs.append(cfg)
    # --------------------------------------------------------------------------------------------------------
    # Table 7: CISS with Reinhard stylization in target domain.
    # --------------------------------------------------------------------------------------------------------
    elif id == 136:
        seeds = [0, 1, 2]
        #         source,          target,         crop,        rcs_min_crop
        cs2acdc = ('cityscapesHR', 'acdcHR',       '1024x1024', 0.5 * (2 ** 2))
        stylization = 'reinhard'
        dec, backbone = 'daformer_sepaspp', 'mitb5'
        uda, rcs_T, plcrop = 'dacs_a999_fdthings_ciss_src_ceorig_trg_ceorigorig_invorigorigstylizedstylized', 0.01, False
        inference = 'slide'
        workers_per_gpu = 16
        for dataset, architecture, sync_crop_size in [
            (cs2acdc, f'hrda1-512-0.1_{dec}', None),
        ]:
            for inv_loss_weight in [
                2.0,
            ]:
                for seed in seeds:
                    source, target, crop, rcs_min_crop = dataset
                    gpu_model = 'NVIDIATITANRTX'
                    cfg = config_from_vars()
                    cfgs.append(cfg)
    # ------------------------------------------------------------------------------------------------------------------------------------------
    # Figure 4: CISS ablation study on invariance point in source domain. Here, invariance is applied to the outputs of the network (red curve).
    # ------------------------------------------------------------------------------------------------------------------------------------------
    elif id == 137:
        seeds = [0, 1, 2]
        #         source,          target,         crop,        rcs_min_crop
        cs2acdc = ('cityscapesHR', 'acdcHR',       '1024x1024', 0.5 * (2 ** 2))
        stylization = 'fda'
        dec, backbone = 'daformer_sepaspp', 'mitb5'
        uda, rcs_T, plcrop = 'dacs_a999_fdthings_ciss_src_ceorig_inv', 0.01, False
        inference = 'slide'
        workers_per_gpu = 16
        for dataset, architecture, sync_crop_size in [
            (cs2acdc, f'hrda1-512-0.1_{dec}', None),
        ]:
            for inv_loss_weight in [
                [0.0, 0.0, 0.0, 0.0, 0.001],
                [0.0, 0.0, 0.0, 0.0, 0.01],
                [0.0, 0.0, 0.0, 0.0, 0.1],
                [0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 10.0],
            ]:
                for seed in seeds:
                    source, target, crop, rcs_min_crop = dataset
                    gpu_model = 'NVIDIATITANRTX'
                    cfg = config_from_vars()
                    cfgs.append(cfg)
    else:
        raise NotImplementedError('Unknown id {}'.format(id))

    return cfgs
