# The ema model and the domain-mixing are based on:
# https://github.com/vikolss/DACS

import math
import os
import random
from copy import deepcopy

import mmcv
import numpy as np
import torch
from matplotlib import pyplot as plt
from timm.models.layers import DropPath
from torch.nn.modules.dropout import _DropoutNd

from mmseg.core import add_prefix
from mmseg.models import UDA, HRDAEncoderDecoder, build_segmentor
from mmseg.models.segmentors.hrda_encoder_decoder import crop
from mmseg.models.uda.uda_decorator import UDADecorator, get_module
from mmseg.models.uda.dacs import DACS, calc_grad_magnitude
from mmseg.models.utils.dacs_transforms import (denorm, get_class_masks,
                                                get_mean_std, strong_transform)
from mmseg.models.utils.visualization import subplotimg
from mmseg.utils.utils import downscale_label_ratio


@UDA.register_module()
class DACSDISS(DACS):

    def __init__(self, **cfg):
        super(DACSDISS, self).__init__(**cfg)
        # self.local_iter = 0
        # self.max_iters = cfg['max_iters']
        # self.alpha = cfg['alpha']
        # self.pseudo_threshold = cfg['pseudo_threshold']
        # self.psweight_ignore_top = cfg['pseudo_weight_ignore_top']
        # self.psweight_ignore_bottom = cfg['pseudo_weight_ignore_bottom']
        # self.fdist_lambda = cfg['imnet_feature_dist_lambda']
        # self.fdist_classes = cfg['imnet_feature_dist_classes']
        # self.fdist_scale_min_ratio = cfg['imnet_feature_dist_scale_min_ratio']
        # self.enable_fdist = self.fdist_lambda > 0
        # self.mix = cfg['mix']
        # self.blur = cfg['blur']
        # self.color_jitter_s = cfg['color_jitter_strength']
        # self.color_jitter_p = cfg['color_jitter_probability']
        # self.debug_img_interval = cfg['debug_img_interval']
        # self.print_grad_magnitude = cfg['print_grad_magnitude']
        # assert self.mix == 'class'

        # self.debug_fdist_mask = None
        # self.debug_gt_rescale = None

        # self.class_probs = {}
        # ema_cfg = deepcopy(cfg['model'])
        # self.ema_model = build_segmentor(ema_cfg)

        # if self.enable_fdist:
        #     self.imnet_model = build_segmentor(deepcopy(cfg['model']))
        # else:
        #     self.imnet_model = None
        
        self.stylization = cfg['stylization']
        assert self.stylization['source']['ce_original'] or self.stylization['source']['ce_stylized']
        self.stylization['target'] = cfg['stylization'].get('target', {})
        self.stylization['target']['pseudolabels'] = self.stylization['target'].get('pseudolabels', 'original')
        self.stylization['target']['ce'] = self.stylization['target'].get('ce', [('original', 'original')])
        assert len(self.stylization['target']['ce']) > 0

    def forward_train(self,
                      img,
                      img_stylized,
                      img_metas,
                      gt_semantic_seg,
                      target_img,
                      target_img_stylized,
                      target_img_metas,
                      rare_class=None,
                      valid_pseudo_mask=None):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        log_vars = {}
        batch_size = img.shape[0]
        dev = img.device

        # Init/update ema model
        if self.local_iter == 0:
            self._init_ema_weights()
            # assert _params_equal(self.get_ema_model(), self.get_model())

        if self.local_iter > 0:
            self._update_ema(self.local_iter)
            # assert not _params_equal(self.get_ema_model(), self.get_model())
            # assert self.get_ema_model().training
        self.update_debug_state()
        seg_debug = {}

        means, stds = get_mean_std(img_metas, dev)
        strong_parameters = {
            'mix': None,
            'color_jitter': random.uniform(0, 1),
            'color_jitter_s': self.color_jitter_s,
            'color_jitter_p': self.color_jitter_p,
            'blur': random.uniform(0, 1) if self.blur else 0,
            'mean': means[0].unsqueeze(0),  # assume same normalization
            'std': stds[0].unsqueeze(0)
        }

        # Train on source images.
        # 1) Train on original source images.
        if self.stylization['source']['ce_original']:
            clean_losses = self.get_model().forward_train(
                img, img_metas, gt_semantic_seg, return_feat=True)
            src_feat = clean_losses.pop('features')
            seg_debug['Source'] = self.get_model().decode_head.debug_output
            clean_loss, clean_log_vars = self._parse_losses(clean_losses)
            log_vars.update(clean_log_vars)
            clean_loss.backward(retain_graph=self.enable_fdist)
            if self.print_grad_magnitude:
                params = self.get_model().backbone.parameters()
                seg_grads = [
                    p.grad.detach().clone() for p in params if p.grad is not None
                ]
                grad_mag = calc_grad_magnitude(seg_grads)
                mmcv.print_log(f'Seg. Grad.: {grad_mag}', 'mmseg')
        
        # 2) Train on stylized source images.
        if self.stylization['source']['ce_stylized']:
            clean_stylized_losses = self.get_model().forward_train(
                img_stylized, img_metas, gt_semantic_seg, return_feat=True)
            src_feat_stylized = clean_stylized_losses.pop('features')
            seg_debug['Source Stylized'] = self.get_model().decode_head.debug_output
            clean_stylized_loss, clean_stylized_log_vars = self._parse_losses(clean_stylized_losses)
            log_vars.update(clean_stylized_log_vars)
            clean_stylized_loss.backward(retain_graph=self.enable_fdist)
            if self.print_grad_magnitude:
                params = self.get_model().backbone.parameters()
                seg_stylized_grads = [
                    p.grad.detach().clone() for p in params if p.grad is not None
                ]
                if self.stylization['source']['ce_original']:
                    seg_stylized_grads = [g2 - g1 for g1, g2 in zip(seg_grads, seg_stylized_grads)]
                grad_mag = calc_grad_magnitude(seg_stylized_grads)
                mmcv.print_log(f'Seg. Grad. Stylized: {grad_mag}', 'mmseg')

        # ImageNet feature distance
        if self.enable_fdist:
            # on original source images
            if self.stylization['source']['ce_original']:
                feat_loss, feat_log = self.calc_feat_dist(img, gt_semantic_seg,
                                                        src_feat)
                log_vars.update(add_prefix(feat_log, 'src'))
                feat_loss.backward()
                if self.print_grad_magnitude:
                    params = self.get_model().backbone.parameters()
                    fd_grads = [
                        p.grad.detach() for p in params if p.grad is not None
                    ]
                    if self.stylization['source']['ce_stylized']:
                        fd_grads = [g3 - (g1 + g2) for g1, g2, g3 in zip(seg_grads, seg_stylized_grads, fd_grads)]
                    else:
                        fd_grads = [g2 - g1 for g1, g2 in zip(seg_grads, fd_grads)]
                    grad_mag = calc_grad_magnitude(fd_grads)
                    mmcv.print_log(f'Fdist Grad.: {grad_mag}', 'mmseg')
            # on stylized source images
            if self.stylization['source']['ce_stylized']:
                feat_stylized_loss, feat_stylized_log = self.calc_feat_dist(img_stylized, gt_semantic_seg, src_feat_stylized)
                log_vars.update(add_prefix(feat_stylized_log, 'src_stylized'))
                feat_stylized_loss.backward()
                if self.print_grad_magnitude:
                    params = self.get_model().backbone.parameters()
                    fd_stylized_grads = [
                        p.grad.detach() for p in params if p.grad is not None
                    ]
                    if self.stylization['source']['ce_original']:
                        fd_stylized_grads = [g4 - (g1 + g2 + g3) for g1, g2, g3, g4 in zip(seg_grads, seg_stylized_grads, fd_grads, fd_stylized_grads)]
                    else:
                        fd_stylized_grads = [g2 - g1 for g1, g2 in zip(seg_stylized_grads, fd_stylized_grads)]
                    grad_mag = calc_grad_magnitude(fd_stylized_grads)
                    mmcv.print_log(f'Fdist Grad.: {grad_mag}', 'mmseg')
        if self.stylization['source']['ce_original']:
            del src_feat, clean_loss
        if self.stylization['source']['ce_stylized']:
            del src_feat_stylized, clean_stylized_loss
        if self.enable_fdist:
            if self.stylization['source']['ce_original']:
                del feat_loss
            if self.stylization['source']['ce_stylized']:
                del feat_stylized_loss

        # Generate pseudo-label
        for m in self.get_ema_model().modules():
            if isinstance(m, _DropoutNd):
                m.training = False
            if isinstance(m, DropPath):
                m.training = False
        if self.stylization['target']['pseudolabels'] == 'original':
            ema_logits = self.get_ema_model().generate_pseudo_label(
                target_img, target_img_metas)
        elif self.stylization['target']['pseudolabels'] == 'stylized':
            ema_logits = self.get_ema_model().generate_pseudo_label(
                target_img_stylized, target_img_metas)
        seg_debug['Target'] = self.get_ema_model().decode_head.debug_output

        ema_softmax = torch.softmax(ema_logits.detach(), dim=1)
        del ema_logits
        pseudo_prob, pseudo_label = torch.max(ema_softmax, dim=1)
        ps_large_p = pseudo_prob.ge(self.pseudo_threshold).long() == 1
        ps_size = np.size(np.array(pseudo_label.cpu()))
        pseudo_weight = torch.sum(ps_large_p).item() / ps_size
        pseudo_weight = pseudo_weight * torch.ones(
            pseudo_prob.shape, device=dev)
        del pseudo_prob, ps_large_p, ps_size

        if self.psweight_ignore_top > 0:
            # Don't trust pseudo-labels in regions with potential
            # rectification artifacts. This can lead to a pseudo-label
            # drift from sky towards building or traffic light.
            assert valid_pseudo_mask is None
            pseudo_weight[:, :self.psweight_ignore_top, :] = 0
        if self.psweight_ignore_bottom > 0:
            assert valid_pseudo_mask is None
            pseudo_weight[:, -self.psweight_ignore_bottom:, :] = 0
        if valid_pseudo_mask is not None:
            pseudo_weight *= valid_pseudo_mask.squeeze(1)
        gt_pixel_weight = torch.ones((pseudo_weight.shape), device=dev)

        # Apply mixing
        mixed_img, mixed_lbl = [None] * len(self.stylization['target']['ce']), [None] * len(self.stylization['target']['ce'])
        mix_masks = [None] * len(self.stylization['target']['ce'])
        mix_losses = [None] * len(self.stylization['target']['ce'])
        for j, s in enumerate(self.stylization['target']['ce']):
            mixed_img[j], mixed_lbl[j] = [None] * batch_size, [None] * batch_size
            mix_masks[j] = get_class_masks(gt_semantic_seg)
            style_source = s[0]
            style_target = s[1]
            if style_source == 'stylized':
                source_img_input = img_stylized
            elif style_source == 'original':
                source_img_input = img
            if style_target == 'stylized':
                target_img_input = target_img_stylized
            elif style_target == 'original':
                target_img_input = target_img

            for i in range(batch_size):                
                strong_parameters['mix'] = mix_masks[j][i]
                mixed_img[j][i], mixed_lbl[j][i] = strong_transform(
                    strong_parameters,
                    data=torch.stack((source_img_input[i], target_img_input[i])),
                    target=torch.stack((gt_semantic_seg[i][0], pseudo_label[i])))
                _, pseudo_weight[i] = strong_transform(
                    strong_parameters,
                    target=torch.stack((gt_pixel_weight[i], pseudo_weight[i])))
            mixed_img[j] = torch.cat(mixed_img[j])
            mixed_lbl[j] = torch.cat(mixed_lbl[j])

            # Train on mixed images
            mix_losses[j] = self.get_model().forward_train(
                mixed_img[j], img_metas, mixed_lbl[j], pseudo_weight, return_feat=False)
            seg_debug[' '.join(['Mix', style_source, style_target])] = self.get_model().decode_head.debug_output
            mix_losses[j] = add_prefix(mix_losses[j], '_'.join(['mix', style_source, style_target]))
            mix_loss, mix_log_vars = self._parse_losses(mix_losses[j])
            log_vars.update(mix_log_vars)
            mix_loss.backward()
        del gt_pixel_weight

        if self.local_iter % self.debug_img_interval == 0:
            out_dir = os.path.join(self.train_cfg['work_dir'],
                                   'class_mix_debug')
            os.makedirs(out_dir, exist_ok=True)
            vis_img = torch.clamp(denorm(img, means, stds), 0, 1)
            vis_img_stylized = torch.clamp(denorm(img_stylized, means, stds), 0, 1)
            vis_trg_img = torch.clamp(denorm(target_img, means, stds), 0, 1)
            vis_trg_img_stylized = torch.clamp(denorm(target_img_stylized, means, stds), 0, 1)
            vis_mixed_img = torch.clamp(denorm(mixed_img[0], means, stds), 0, 1)
            for j in range(batch_size):
                rows, cols = 2, 5
                fig, axs = plt.subplots(
                    rows,
                    cols,
                    figsize=(3 * cols, 3 * rows),
                    gridspec_kw={
                        'hspace': 0.1,
                        'wspace': 0,
                        'top': 0.95,
                        'bottom': 0,
                        'right': 1,
                        'left': 0
                    },
                )
                subplotimg(axs[0][0], vis_img[j], 'Source Image')
                subplotimg(axs[1][0], vis_trg_img[j], 'Target Image')
                subplotimg(axs[0][1], vis_img_stylized[j], 'Stylized Source Image')
                subplotimg(axs[1][1], vis_trg_img_stylized[j], 'Stylized Target Image')
                subplotimg(
                    axs[0][2],
                    gt_semantic_seg[j],
                    'Source Seg GT',
                    cmap='cityscapes')
                subplotimg(
                    axs[1][2],
                    pseudo_label[j],
                    'Target Seg (Pseudo) GT',
                    cmap='cityscapes')
                # subplotimg(axs[0][2], vis_mixed_img[j], 'Mixed Image')
                subplotimg(
                    axs[0][3], mix_masks[0][j][0], 'Domain Mask', cmap='gray')
                # subplotimg(axs[0][3], pred_u_s[j], "Seg Pred",
                #            cmap="cityscapes")
                subplotimg(
                    axs[1][3], mixed_lbl[j], 'Seg Targ', cmap='cityscapes')
                # subplotimg(
                #     axs[0][3], pseudo_weight[j], 'Pseudo W.', vmin=0, vmax=1)
                if self.debug_fdist_mask is not None:
                    subplotimg(
                        axs[0][4],
                        self.debug_fdist_mask[j][0],
                        'FDist Mask',
                        cmap='gray')
                if self.debug_gt_rescale is not None:
                    subplotimg(
                        axs[1][4],
                        self.debug_gt_rescale[j],
                        'Scaled GT',
                        cmap='cityscapes')
                for ax in axs.flat:
                    ax.axis('off')
                plt.savefig(
                    os.path.join(out_dir,
                                 f'{(self.local_iter + 1):06d}_{j}.png'))
                plt.close()

            if (seg_debug['Source'] is not None or seg_debug['Source Stylized'] is not None) and seg_debug:
                rows =\
                    1 +\
                    int(self.stylization['source']['ce_original']) +\
                    int(self.stylization['source']['ce_stylized']) +\
                    len(self.stylization['target']['ce'])
                cols = len(seg_debug['Source']) if seg_debug.get('Source') is not None else len(seg_debug['Source Stylized'])
                for j in range(batch_size):
                    fig, axs = plt.subplots(
                        rows,
                        cols,
                        figsize=(3 * cols, 3 * rows),
                        gridspec_kw={
                            'hspace': 0.1,
                            'wspace': 0,
                            'top': 0.95,
                            'bottom': 0,
                            'right': 1,
                            'left': 0
                        },
                    )
                    for k1, (n1, outs) in enumerate(seg_debug.items()):
                        for k2, (n2, out) in enumerate(outs.items()):
                            if out.shape[1] == 3:
                                vis = torch.clamp(
                                    denorm(out, means, stds), 0, 1)
                                subplotimg(axs[k1][k2], vis[j], f'{n1} {n2}')
                            else:
                                if out.ndim == 3:
                                    args = dict(cmap='cityscapes')
                                else:
                                    args = dict(cmap='gray', vmin=0, vmax=1)
                                subplotimg(axs[k1][k2], out[j], f'{n1} {n2}',
                                           **args)
                    for ax in axs.flat:
                        ax.axis('off')
                    plt.savefig(
                        os.path.join(out_dir,
                                     f'{(self.local_iter + 1):06d}_{j}_s.png'))
                    plt.close()
        self.local_iter += 1

        return log_vars
