import json
import os.path as osp

import mmcv
import numpy as np
import torch
from mmcv.parallel import DataContainer as DC

from .custom_dual import CustomDatasetDual
from . import CityscapesDataset
from .builder import DATASETS


def get_rcs_class_probs(data_root, temperature):
    with open(osp.join(data_root, 'sample_class_stats.json'), 'r') as of:
        sample_class_stats = json.load(of)
    overall_class_stats = {}
    for s in sample_class_stats:
        s.pop('file')
        for c, n in s.items():
            c = int(c)
            if c not in overall_class_stats:
                overall_class_stats[c] = n
            else:
                overall_class_stats[c] += n
    overall_class_stats = {
        k: v
        for k, v in sorted(
            overall_class_stats.items(), key=lambda item: item[1])
    }
    freq = torch.tensor(list(overall_class_stats.values()))
    freq = freq / torch.sum(freq)
    freq = 1 - freq
    freq = torch.softmax(freq / temperature, dim=-1)

    return list(overall_class_stats.keys()), freq.numpy()


def get_crop_bbox(img_size, crop_size):
    """Randomly get a crop bounding box."""
    assert len(img_size) == len(crop_size)
    assert len(img_size) == 2
    margin_h = max(img_size[0] - crop_size[0], 0)
    margin_w = max(img_size[1] - crop_size[1], 0)
    offset_h = np.random.randint(0, margin_h + 1)
    offset_w = np.random.randint(0, margin_w + 1)
    crop_y1, crop_y2 = offset_h, offset_h + crop_size[0]
    crop_x1, crop_x2 = offset_w, offset_w + crop_size[1]

    return crop_y1, crop_y2, crop_x1, crop_x2


@DATASETS.register_module()
class UDADatasetDual(CustomDatasetDual):

    def __init__(self,
                 pipeline,
                 source,
                 target,
                 img_dir_source,
                 img_dir_target,
                 img_suffix_source='.jpg',
                 img_suffix_target='.jpg',
                 ann_dir_source=None,
                 ann_dir_target=None,
                 seg_map_suffix_source='.png',
                 seg_map_suffix_target='.png',
                 split_source=None,
                 split_target=None,
                 data_root_source=None,
                 data_root_target=None,
                 test_mode=False,
                 ignore_index=255,
                 crop_pseudo_margins_target=None,
                 valid_mask_size_target=None,
                 reduce_zero_label=False,
                 classes=None,
                 palette=None,
                 sync_crop_size=None,
                 rare_class_sampling=None):
                 
        if crop_pseudo_margins_target is not None:
            assert pipeline[-1]['type'] == 'Collect'
            pipeline[-1]['keys'][-1].append('valid_pseudo_mask')

        super(UDADatasetDual, self).__init__(pipeline=pipeline,
                                             img_dir_source=img_dir_source,
                                             img_dir_target=img_dir_target,
                                             img_suffix_source=img_suffix_source,
                                             img_suffix_target=img_suffix_target,
                                             ann_dir_source=ann_dir_source,
                                             ann_dir_target=ann_dir_target,
                                             seg_map_suffix_source=seg_map_suffix_source,
                                             seg_map_suffix_target=seg_map_suffix_target,
                                             split_source=split_source,
                                             split_target=split_target,
                                             data_root_source=data_root_source,
                                             data_root_target=data_root_target,
                                             test_mode=test_mode,
                                             ignore_index=ignore_index,
                                             crop_pseudo_margins_target=crop_pseudo_margins_target,
                                             valid_mask_size_target=valid_mask_size_target,
                                             reduce_zero_label=reduce_zero_label,
                                             classes=classes,
                                             palette=palette)
        self.source = source
        self.target = target
        self.sync_crop_size = sync_crop_size
        rcs_cfg = rare_class_sampling
        self.rcs_enabled = rcs_cfg is not None
        if self.rcs_enabled:
            self.rcs_class_temp = rcs_cfg['class_temp']
            self.rcs_min_crop_ratio = rcs_cfg['min_crop_ratio']
            self.rcs_min_pixels = rcs_cfg['min_pixels']

            self.rcs_classes, self.rcs_classprob = get_rcs_class_probs(
                data_root_source, self.rcs_class_temp)
            mmcv.print_log(f'RCS Classes: {self.rcs_classes}', 'mmseg')
            mmcv.print_log(f'RCS ClassProb: {self.rcs_classprob}', 'mmseg')

            with open(osp.join(data_root_source, 'samples_with_class.json'), 'r') as of:
                samples_with_class_and_n = json.load(of)
            samples_with_class_and_n = {
                int(k): v
                for k, v in samples_with_class_and_n.items()
                if int(k) in self.rcs_classes
            }
            self.samples_with_class = {}
            for c in self.rcs_classes:
                self.samples_with_class[c] = []
                for file, pixels in samples_with_class_and_n[c]:
                    if pixels > self.rcs_min_pixels:
                        self.samples_with_class[c].append(file.split('/')[-1])
                assert len(self.samples_with_class[c]) > 0
            self.file_to_idx = {}
            for i, dic in enumerate(self.img_infos_source):
                file = dic['ann']['seg_map']
                if self.source == 'Cityscapes':
                    file = file.split('/')[-1]
                self.file_to_idx[file] = i

    def pre_pipeline(self, results):
        super(UDADatasetDual, self).pre_pipeline(results)
        if self.crop_pseudo_margins_target is not None:
            results['target']['valid_pseudo_mask'] = np.ones(
                self.valid_mask_size_target, dtype=np.uint8)
            # Don't trust pseudo-labels in regions with potential
            # rectification artifacts. This can lead to a pseudo-label
            # drift from sky towards building or traffic light.
            if self.crop_pseudo_margins_target[0] > 0:
                results['target']['valid_pseudo_mask'][:self.crop_pseudo_margins_target[0], :] = 0
            # Here, the if statement is absolutely necessary
            if self.crop_pseudo_margins_target[1] > 0:
                results['target']['valid_pseudo_mask'][-self.crop_pseudo_margins_target[1]:, :] = 0
            if self.crop_pseudo_margins_target[2] > 0:
                results['target']['valid_pseudo_mask'][:, :self.crop_pseudo_margins_target[2]] = 0
            # Here, the if statement is absolutely necessary
            if self.crop_pseudo_margins_target[3] > 0:
                results['target']['valid_pseudo_mask'][:, -self.crop_pseudo_margins_target[3]:] = 0
            results['target']['seg_fields'].append('valid_pseudo_mask')

    def synchronized_crop(self, s1, s2):
        if self.sync_crop_size is None:
            return s1, s2
        orig_crop_size = s1['img'].data.shape[1:]
        crop_y1, crop_y2, crop_x1, crop_x2 = get_crop_bbox(
            orig_crop_size, self.sync_crop_size)
        for i, s in enumerate([s1, s2]):
            for key in ['img', 'img_stylized', 'gt_semantic_seg', 'valid_pseudo_mask']:
                if key not in s:
                    continue
                s[key] = DC(
                    s[key].data[:, crop_y1:crop_y2, crop_x1:crop_x2],
                    stack=s[key]._stack)
        return s1, s2

    def get_rare_class_sample(self):
        c = np.random.choice(self.rcs_classes, p=self.rcs_classprob)
        f1 = np.random.choice(self.samples_with_class[c])
        i1 = self.file_to_idx[f1]
        i2 = np.random.choice(range(len(self.img_infos_target)))
        idx = i1 * len(self.img_infos_target) + i2
        results = super().__getitem__(idx)
        s1 = results['source']
        if self.rcs_min_crop_ratio > 0:
            for j in range(10):
                n_class = torch.sum(s1['gt_semantic_seg'].data == c)
                # mmcv.print_log(f'{j}: {n_class}', 'mmseg')
                if n_class > self.rcs_min_pixels * self.rcs_min_crop_ratio:
                    break
                # Sample a new random crop from source image i1.
                # Please note, that super().__getitem__(idx) applies the
                # preprocessing pipeline to the loaded image, which includes
                # RandomCrop, and results in a new crop of the image.
                results = super().__getitem__(idx)
                s1 = results['source']
        s2 = results['target']

        # Before synchronized_crop(), s1 and s2 are cropped independently from
        # the entire image when calling results = super().__getitem__(idx).
        # This corresponds to the original implementation in DACS and DAFormer.
        # However, in both papers only large crops were used.
        # In some experiments of the HRDA paper, smaller crop sizes are
        # necessary. We found that independent small crops do not work
        # well with ClassMix (see dacs.py) as the content layout does not
        # match. Therefore, we use synchronized cropping, where the same
        # subcrop region is applied to s1 and s2.
        s1, s2 = self.synchronized_crop(s1, s2)
        out = {
            **s1, 'target_img_metas': s2['img_metas'],
            'target_img': s2['img'], 'target_img_stylized': s2['img_stylized']
        }
        if 'valid_pseudo_mask' in s2:
            out['valid_pseudo_mask'] = s2['valid_pseudo_mask']
        return out

    def __getitem__(self, idx):
        if self.rcs_enabled:
            return self.get_rare_class_sample()
        else:
            results = super().__getitem__(idx)
            s1 = results['source']
            s2 = results['target']
            s1, s2 = self.synchronized_crop(s1, s2)
            out = {
                **s1, 'target_img_metas': s2['img_metas'],
                'target_img': s2['img'], 'target_img_stylized': s2['img_stylized']
            }
            if 'valid_pseudo_mask' in s2:
                out['valid_pseudo_mask'] = s2['valid_pseudo_mask']
            return out