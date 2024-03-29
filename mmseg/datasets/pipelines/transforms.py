# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications:
# 1) Support override_scale in Resize
# 2) Update Resize, RandomCrop, RandomFlip, Normalize, Pad to support indexing of results with keys.
# 3) Include Fourier domain adaptation (FDA) transform.

import mmcv
from mmcv.utils import Timer
import numpy as np
from mmcv.utils import deprecated_api_warning, is_tuple_of
from numpy import random

from ..builder import PIPELINES


@PIPELINES.register_module()
class Resize(object):
    """Resize images & seg.

    This transform resizes the input image to some scale. If the input dict
    contains the key "scale", then the scale in the input dict is used,
    otherwise the specified scale in the init method is used.

    ``img_scale`` can be None, a tuple (single-scale) or a list of tuple
    (multi-scale). There are 4 multiscale modes:

    - ``ratio_range is not None``:
    1. When img_scale is None, img_scale is the shape of image in results
        (img_scale = results['img'].shape[:2]) and the image is resized based
        on the original size. (mode 1)
    2. When img_scale is a tuple (single-scale), randomly sample a ratio from
        the ratio range and multiply it with the image scale. (mode 2)

    - ``ratio_range is None and multiscale_mode == "range"``: randomly sample a
    scale from the a range. (mode 3)

    - ``ratio_range is None and multiscale_mode == "value"``: randomly sample a
    scale from multiple scales. (mode 4)

    Args:
        img_scale (tuple or list[tuple]): Images scales for resizing.
            Default:None.
        multiscale_mode (str): Either "range" or "value".
            Default: 'range'
        ratio_range (tuple[float]): (min_ratio, max_ratio).
            Default: None
        keep_ratio (bool): Whether to keep the aspect ratio when resizing the
            image. Default: True
    """

    def __init__(self,
                 img_scale=None,
                 multiscale_mode='range',
                 ratio_range=None,
                 keep_ratio=True,
                 override_scale=False,
                 keys=None):
        if img_scale is None:
            self.img_scale = None
        else:
            if isinstance(img_scale, list):
                self.img_scale = img_scale
            else:
                self.img_scale = [img_scale]
            assert mmcv.is_list_of(self.img_scale, tuple)

        if ratio_range is not None:
            # mode 1: given img_scale=None and a range of image ratio
            # mode 2: given a scale and a range of image ratio
            assert self.img_scale is None or len(self.img_scale) == 1
        else:
            # mode 3 and 4: given multiple scales or a range of scales
            assert multiscale_mode in ['value', 'range']

        self.multiscale_mode = multiscale_mode
        self.ratio_range = ratio_range
        self.keep_ratio = keep_ratio
        self.override_scale = override_scale
        self.keys = keys

    @staticmethod
    def random_select(img_scales):
        """Randomly select an img_scale from given candidates.

        Args:
            img_scales (list[tuple]): Images scales for selection.

        Returns:
            (tuple, int): Returns a tuple ``(img_scale, scale_dix)``,
                where ``img_scale`` is the selected image scale and
                ``scale_idx`` is the selected index in the given candidates.
        """

        assert mmcv.is_list_of(img_scales, tuple)
        scale_idx = np.random.randint(len(img_scales))
        img_scale = img_scales[scale_idx]
        return img_scale, scale_idx

    @staticmethod
    def random_sample(img_scales):
        """Randomly sample an img_scale when ``multiscale_mode=='range'``.

        Args:
            img_scales (list[tuple]): Images scale range for sampling.
                There must be two tuples in img_scales, which specify the lower
                and upper bound of image scales.

        Returns:
            (tuple, None): Returns a tuple ``(img_scale, None)``, where
                ``img_scale`` is sampled scale and None is just a placeholder
                to be consistent with :func:`random_select`.
        """

        assert mmcv.is_list_of(img_scales, tuple) and len(img_scales) == 2
        img_scale_long = [max(s) for s in img_scales]
        img_scale_short = [min(s) for s in img_scales]
        long_edge = np.random.randint(
            min(img_scale_long),
            max(img_scale_long) + 1)
        short_edge = np.random.randint(
            min(img_scale_short),
            max(img_scale_short) + 1)
        img_scale = (long_edge, short_edge)
        return img_scale, None

    @staticmethod
    def random_sample_ratio(img_scale, ratio_range):
        """Randomly sample an img_scale when ``ratio_range`` is specified.

        A ratio will be randomly sampled from the range specified by
        ``ratio_range``. Then it would be multiplied with ``img_scale`` to
        generate sampled scale.

        Args:
            img_scale (tuple): Images scale base to multiply with ratio.
            ratio_range (tuple[float]): The minimum and maximum ratio to scale
                the ``img_scale``.

        Returns:
            (tuple, None): Returns a tuple ``(scale, None)``, where
                ``scale`` is sampled ratio multiplied with ``img_scale`` and
                None is just a placeholder to be consistent with
                :func:`random_select`.
        """

        assert isinstance(img_scale, tuple) and len(img_scale) == 2
        min_ratio, max_ratio = ratio_range
        assert min_ratio <= max_ratio
        ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
        scale = int(img_scale[0] * ratio), int(img_scale[1] * ratio)
        return scale, None

    def _random_scale(self, results):
        """Randomly sample an img_scale according to ``ratio_range`` and
        ``multiscale_mode``.

        If ``ratio_range`` is specified, a ratio will be sampled and be
        multiplied with ``img_scale``.
        If multiple scales are specified by ``img_scale``, a scale will be
        sampled according to ``multiscale_mode``.
        Otherwise, single scale will be used.

        Args:
            results (dict): Result dict from :obj:`dataset`.

        Returns:
            dict: Two new keys 'scale` and 'scale_idx` are added into
                ``results``, which would be used by subsequent pipelines.
        """

        if self.ratio_range is not None:
            if self.img_scale is None:
                h, w = results['img'].shape[:2]
                scale, scale_idx = self.random_sample_ratio((w, h),
                                                            self.ratio_range)
            else:
                scale, scale_idx = self.random_sample_ratio(
                    self.img_scale[0], self.ratio_range)
        elif len(self.img_scale) == 1:
            scale, scale_idx = self.img_scale[0], 0
        elif self.multiscale_mode == 'range':
            scale, scale_idx = self.random_sample(self.img_scale)
        elif self.multiscale_mode == 'value':
            scale, scale_idx = self.random_select(self.img_scale)
        else:
            raise NotImplementedError

        results['scale'] = scale
        results['scale_idx'] = scale_idx

        return results

    def _resize_img(self, results):
        """Resize images with ``results['scale']``."""
        if self.keep_ratio:
            img, scale_factor = mmcv.imrescale(
                results['img'], results['scale'], return_scale=True)
            # the w_scale and h_scale has minor difference
            # a real fix should be done in the mmcv.imrescale in the future
            new_h, new_w = img.shape[:2]
            h, w = results['img'].shape[:2]
            w_scale = new_w / w
            h_scale = new_h / h
        else:
            img, w_scale, h_scale = mmcv.imresize(
                results['img'], results['scale'], return_scale=True)
        scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                                dtype=np.float32)
        results['img'] = img
        results['img_shape'] = img.shape
        results['pad_shape'] = img.shape  # in case that there is no padding
        results['scale_factor'] = scale_factor
        results['keep_ratio'] = self.keep_ratio
        
        return results

    def _resize_seg(self, results):
        """Resize semantic segmentation map with ``results['scale']``."""
        for key in results.get('seg_fields', []):
            if self.keep_ratio:
                gt_seg = mmcv.imrescale(
                    results[key], results['scale'], interpolation='nearest')
            else:
                gt_seg = mmcv.imresize(
                    results[key], results['scale'], interpolation='nearest')
            results[key] = gt_seg
        
        return results

    def __call__(self, results):
        """Call function to resize images, bounding boxes, masks, semantic
        segmentation map.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Resized results, 'img_shape', 'pad_shape', 'scale_factor',
                'keep_ratio' keys are added into result dict.
        """

        if self.keys is None:
            results_tmp = [results]
        else:
            results_tmp = []
            for k in self.keys:
                results_tmp.append(results[k])
        for i, r in enumerate(results_tmp):
            if 'scale' not in r or self.override_scale:
                results_tmp[i] = self._random_scale(results_tmp[i])
            results_tmp[i] = self._resize_img(results_tmp[i])
            results_tmp[i] = self._resize_seg(results_tmp[i])
        if self.keys is None:
            results = results_tmp[0]
        else:
            for i, k in enumerate(self.keys):
                results[k] = results_tmp[i]
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(img_scale={self.img_scale}, '
                     f'multiscale_mode={self.multiscale_mode}, '
                     f'ratio_range={self.ratio_range}, '
                     f'keep_ratio={self.keep_ratio})')
        return repr_str


@PIPELINES.register_module()
class RandomFlip(object):
    """Flip the image & seg.

    If the input dict contains the key "flip", then the flag will be used,
    otherwise it will be randomly decided by a ratio specified in the init
    method.

    Args:
        prob (float, optional): The flipping probability. Default: None.
        direction(str, optional): The flipping direction. Options are
            'horizontal' and 'vertical'. Default: 'horizontal'.
    """

    @deprecated_api_warning({'flip_ratio': 'prob'}, cls_name='RandomFlip')
    def __init__(self, prob=None, direction='horizontal', keys=None):
        self.prob = prob
        self.direction = direction
        if prob is not None:
            assert prob >= 0 and prob <= 1
        assert direction in ['horizontal', 'vertical']
        self.keys = keys

    def __call__(self, results):
        """Call function to flip bounding boxes, masks, semantic segmentation
        maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Flipped results, 'flip', 'flip_direction' keys are added into
                result dict.
        """

        if self.keys is None:
            results_tmp = [results]
        else:
            results_tmp = []
            for k in self.keys:
                results_tmp.append(results[k])
        
        for i, r in enumerate(results_tmp):
            if 'flip' not in r:
                flip = True if np.random.rand() < self.prob else False
                results_tmp[i]['flip'] = flip
            if 'flip_direction' not in r:
                results_tmp[i]['flip_direction'] = self.direction
            if results_tmp[i]['flip']:
                # flip image
                results_tmp[i]['img'] = mmcv.imflip(
                    r['img'], direction=results_tmp[i]['flip_direction'])

                # Flip the stylized image.
                if 'img_stylized' in r:
                    results_tmp[i]['img_stylized'] = mmcv.imflip(
                        r['img_stylized'], direction=results_tmp[i]['flip_direction'])
                
                # flip segs
                for key in r.get('seg_fields', []):
                    # use copy() to make numpy stride positive
                    results_tmp[i][key] = mmcv.imflip(
                        r[key], direction=results_tmp[i]['flip_direction']).copy()
        
        if self.keys is None:
            results = results_tmp[0]
        else:
            for i, k in enumerate(self.keys):
                results[k] = results_tmp[i]

        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(prob={self.prob})'


@PIPELINES.register_module()
class Pad(object):
    """Pad the image & mask.

    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.
    Added keys are "pad_shape", "pad_fixed_size", "pad_size_divisor",

    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value. Default: 0.
        seg_pad_val (float, optional): Padding value of segmentation map.
            Default: 255.
    """

    def __init__(self,
                 size=None,
                 size_divisor=None,
                 pad_val=0,
                 seg_pad_val=255,
                 keys=None):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        self.seg_pad_val = seg_pad_val
        self.keys = keys
        # only one of size and size_divisor should be valid
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None

    def _pad_img(self, results):
        """Pad images according to ``self.size``."""
        if self.size is not None:
            padded_img = mmcv.impad(
                results['img'], shape=self.size, pad_val=self.pad_val)
            if 'img_stylized' in results:
                padded_img_stylized = mmcv.impad(
                    results['img_stylized'], shape=self.size, pad_val=self.pad_val)
        elif self.size_divisor is not None:
            padded_img = mmcv.impad_to_multiple(
                results['img'], self.size_divisor, pad_val=self.pad_val)
            if 'img_stylized' in results:
                padded_img_stylized = mmcv.impad_to_multiple(
                    results['img_stylized'], self.size_divisor, pad_val=self.pad_val)
        results['img'] = padded_img
        if 'img_stylized' in results:
            results['img_stylized'] = padded_img_stylized
        results['pad_shape'] = padded_img.shape
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor

        return results

    def _pad_seg(self, results):
        """Pad masks according to ``results['pad_shape']``."""
        for key in results.get('seg_fields', []):
            results[key] = mmcv.impad(
                results[key],
                shape=results['pad_shape'][:2],
                pad_val=self.seg_pad_val)
        
        return results

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Updated result dict.
        """

        if self.keys is None:
            results_tmp = [results]
        else:
            results_tmp = []
            for k in self.keys:
                results_tmp.append(results[k])
        
        for i, r in enumerate(results_tmp):
            results_tmp[i] = self._pad_img(results_tmp[i])
            results_tmp[i] = self._pad_seg(results_tmp[i])
        
        if self.keys is None:
            results = results_tmp[0]
        else:
            for i, k in enumerate(self.keys):
                results[k] = results_tmp[i]        
        
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, size_divisor={self.size_divisor}, ' \
                    f'pad_val={self.pad_val})'
        return repr_str


@PIPELINES.register_module()
class Normalize(object):
    """Normalize the image.

    Added key is "img_norm_cfg".

    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True, keys=None):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb
        self.keys = keys

    def __call__(self, results):
        """Call function to normalize images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """

        if self.keys is None:
            results_tmp = [results]
        else:
            results_tmp = []
            for k in self.keys:
                results_tmp.append(results[k])
        
        for i, r in enumerate(results_tmp):
            results_tmp[i]['img'] = mmcv.imnormalize(r['img'], self.mean, self.std,
                                            self.to_rgb)
            if 'img_stylized' in r:
                results_tmp[i]['img_stylized'] = mmcv.imnormalize(
                    r['img_stylized'],
                    self.mean,
                    self.std,
                    self.to_rgb)
            results_tmp[i]['img_norm_cfg'] = dict(
                mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        
        if self.keys is None:
            results = results_tmp[0]
        else:
            for i, k in enumerate(self.keys):
                results[k] = results_tmp[i]        
        
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, std={self.std}, to_rgb=' \
                    f'{self.to_rgb})'
        return repr_str


@PIPELINES.register_module()
class Rerange(object):
    """Rerange the image pixel value.

    Args:
        min_value (float or int): Minimum value of the reranged image.
            Default: 0.
        max_value (float or int): Maximum value of the reranged image.
            Default: 255.
    """

    def __init__(self, min_value=0, max_value=255):
        assert isinstance(min_value, float) or isinstance(min_value, int)
        assert isinstance(max_value, float) or isinstance(max_value, int)
        assert min_value < max_value
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, results):
        """Call function to rerange images.

        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Reranged results.
        """

        img = results['img']
        img_min_value = np.min(img)
        img_max_value = np.max(img)

        assert img_min_value < img_max_value
        # rerange to [0, 1]
        img = (img - img_min_value) / (img_max_value - img_min_value)
        # rerange to [min_value, max_value]
        img = img * (self.max_value - self.min_value) + self.min_value
        results['img'] = img

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(min_value={self.min_value}, max_value={self.max_value})'
        return repr_str


@PIPELINES.register_module()
class CLAHE(object):
    """Use CLAHE method to process the image.

    See `ZUIDERVELD,K. Contrast Limited Adaptive Histogram Equalization[J].
    Graphics Gems, 1994:474-485.` for more information.

    Args:
        clip_limit (float): Threshold for contrast limiting. Default: 40.0.
        tile_grid_size (tuple[int]): Size of grid for histogram equalization.
            Input image will be divided into equally sized rectangular tiles.
            It defines the number of tiles in row and column. Default: (8, 8).
    """

    def __init__(self, clip_limit=40.0, tile_grid_size=(8, 8)):
        assert isinstance(clip_limit, (float, int))
        self.clip_limit = clip_limit
        assert is_tuple_of(tile_grid_size, int)
        assert len(tile_grid_size) == 2
        self.tile_grid_size = tile_grid_size

    def __call__(self, results):
        """Call function to Use CLAHE method process images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Processed results.
        """

        for i in range(results['img'].shape[2]):
            results['img'][:, :, i] = mmcv.clahe(
                np.array(results['img'][:, :, i], dtype=np.uint8),
                self.clip_limit, self.tile_grid_size)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(clip_limit={self.clip_limit}, '\
                    f'tile_grid_size={self.tile_grid_size})'
        return repr_str


@PIPELINES.register_module()
class RandomCrop(object):
    """Random crop the image & seg.

    Args:
        crop_size (tuple): Expected size after cropping, (h, w).
        cat_max_ratio (float): The maximum ratio that single category could
            occupy.
    """

    def __init__(self, crop_size, cat_max_ratio=1., ignore_index=255, keys=None):
        assert crop_size[0] > 0 and crop_size[1] > 0
        self.crop_size = crop_size
        self.cat_max_ratio = cat_max_ratio
        self.ignore_index = ignore_index
        self.keys = keys

    def get_crop_bbox(self, img):
        """Randomly get a crop bounding box."""
        margin_h = max(img.shape[0] - self.crop_size[0], 0)
        margin_w = max(img.shape[1] - self.crop_size[1], 0)
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)
        crop_y1, crop_y2 = offset_h, offset_h + self.crop_size[0]
        crop_x1, crop_x2 = offset_w, offset_w + self.crop_size[1]

        return crop_y1, crop_y2, crop_x1, crop_x2

    def crop(self, img, crop_bbox):
        """Crop from ``img``"""
        crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
        img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
        return img

    def __call__(self, results):
        """Call function to randomly crop images, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """

        if self.keys is None:
            results_tmp = [results]
        else:
            results_tmp = []
            for k in self.keys:
                results_tmp.append(results[k])
        
        for i, r in enumerate(results_tmp):
            img = r['img']
            crop_bbox = self.get_crop_bbox(img)
            if self.cat_max_ratio < 1.:
                # Repeat 10 times
                for _ in range(10):
                    seg_temp = self.crop(r['gt_semantic_seg'], crop_bbox)
                    labels, cnt = np.unique(seg_temp, return_counts=True)
                    cnt = cnt[labels != self.ignore_index]
                    if len(cnt) > 1 and np.max(cnt) / np.sum(
                            cnt) < self.cat_max_ratio:
                        break
                    crop_bbox = self.get_crop_bbox(img)

            # crop the image
            img = self.crop(img, crop_bbox)
            img_shape = img.shape
            results_tmp[i]['img'] = img
            results_tmp[i]['img_shape'] = img_shape

            # crop the stylized image
            if 'img_stylized' in r:
                img_stylized = r['img_stylized']
                img_stylized = self.crop(img_stylized, crop_bbox)
                results_tmp[i]['img_stylized'] = img_stylized
            
            # crop semantic seg
            for key in r.get('seg_fields', []):
                results_tmp[i][key] = self.crop(r[key], crop_bbox)

        if self.keys is None:
            results = results_tmp[0]
        else:
            for i, k in enumerate(self.keys):
                results[k] = results_tmp[i]
        
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(crop_size={self.crop_size})'


@PIPELINES.register_module()
class CentralCrop(object):
    """Central crop the image & seg.

    Args:
        crop_size (tuple): Expected size after cropping, (h, w).
        cat_max_ratio (float): The maximum ratio that single category could
            occupy.
    """

    def __init__(self, crop_size, keys=None):
        assert crop_size[0] > 0 and crop_size[1] > 0
        self.crop_size = crop_size
        self.keys = keys

    def get_crop_bbox(self, img):
        """Get the central crop bounding box."""
        margin_h = max(img.shape[0] - self.crop_size[0], 0)
        margin_w = max(img.shape[1] - self.crop_size[1], 0)
        offset_h = np.floor(margin_h / 2).astype(int)
        offset_w = np.floor(margin_w / 2).astype(int)
        crop_y1, crop_y2 = offset_h, offset_h + self.crop_size[0]
        crop_x1, crop_x2 = offset_w, offset_w + self.crop_size[1]

        return crop_y1, crop_y2, crop_x1, crop_x2

    def crop(self, img, crop_bbox):
        """Crop from ``img``"""
        crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
        img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
        return img

    def __call__(self, results):
        """Call function to randomly crop images, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """

        if self.keys is None:
            results_tmp = [results]
        else:
            results_tmp = []
            for k in self.keys:
                results_tmp.append(results[k])
        
        for i, r in enumerate(results_tmp):
            img = r['img']
            crop_bbox = self.get_crop_bbox(img)
            # crop the image
            img = self.crop(img, crop_bbox)
            img_shape = img.shape
            results_tmp[i]['img'] = img
            results_tmp[i]['img_shape'] = img_shape

            # crop the stylized image
            if 'img_stylized' in r:
                img_stylized = r['img_stylized']
                img_stylized = self.crop(img_stylized, crop_bbox)
                results_tmp[i]['img_stylized'] = img_stylized
            
            # crop semantic seg
            for key in r.get('seg_fields', []):
                results_tmp[i][key] = self.crop(r[key], crop_bbox)

        if self.keys is None:
            results = results_tmp[0]
        else:
            for i, k in enumerate(self.keys):
                results[k] = results_tmp[i]
        
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(crop_size={self.crop_size})'


@PIPELINES.register_module()
class RandomRotate(object):
    """Rotate the image & seg.

    Args:
        prob (float): The rotation probability.
        degree (float, tuple[float]): Range of degrees to select from. If
            degree is a number instead of tuple like (min, max),
            the range of degree will be (``-degree``, ``+degree``)
        pad_val (float, optional): Padding value of image. Default: 0.
        seg_pad_val (float, optional): Padding value of segmentation map.
            Default: 255.
        center (tuple[float], optional): Center point (w, h) of the rotation in
            the source image. If not specified, the center of the image will be
            used. Default: None.
        auto_bound (bool): Whether to adjust the image size to cover the whole
            rotated image. Default: False
    """

    def __init__(self,
                 prob,
                 degree,
                 pad_val=0,
                 seg_pad_val=255,
                 center=None,
                 auto_bound=False):
        self.prob = prob
        assert prob >= 0 and prob <= 1
        if isinstance(degree, (float, int)):
            assert degree > 0, f'degree {degree} should be positive'
            self.degree = (-degree, degree)
        else:
            self.degree = degree
        assert len(self.degree) == 2, f'degree {self.degree} should be a ' \
                                      f'tuple of (min, max)'
        self.pal_val = pad_val
        self.seg_pad_val = seg_pad_val
        self.center = center
        self.auto_bound = auto_bound

    def __call__(self, results):
        """Call function to rotate image, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Rotated results.
        """

        rotate = True if np.random.rand() < self.prob else False
        degree = np.random.uniform(min(*self.degree), max(*self.degree))
        if rotate:
            # rotate image
            results['img'] = mmcv.imrotate(
                results['img'],
                angle=degree,
                border_value=self.pal_val,
                center=self.center,
                auto_bound=self.auto_bound)

            # rotate segs
            for key in results.get('seg_fields', []):
                results[key] = mmcv.imrotate(
                    results[key],
                    angle=degree,
                    border_value=self.seg_pad_val,
                    center=self.center,
                    auto_bound=self.auto_bound,
                    interpolation='nearest')
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob}, ' \
                    f'degree={self.degree}, ' \
                    f'pad_val={self.pal_val}, ' \
                    f'seg_pad_val={self.seg_pad_val}, ' \
                    f'center={self.center}, ' \
                    f'auto_bound={self.auto_bound})'
        return repr_str


@PIPELINES.register_module()
class RGB2Gray(object):
    """Convert RGB image to grayscale image.

    This transform calculate the weighted mean of input image channels with
    ``weights`` and then expand the channels to ``out_channels``. When
    ``out_channels`` is None, the number of output channels is the same as
    input channels.

    Args:
        out_channels (int): Expected number of output channels after
            transforming. Default: None.
        weights (tuple[float]): The weights to calculate the weighted mean.
            Default: (0.299, 0.587, 0.114).
    """

    def __init__(self, out_channels=None, weights=(0.299, 0.587, 0.114)):
        assert out_channels is None or out_channels > 0
        self.out_channels = out_channels
        assert isinstance(weights, tuple)
        for item in weights:
            assert isinstance(item, (float, int))
        self.weights = weights

    def __call__(self, results):
        """Call function to convert RGB image to grayscale image.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with grayscale image.
        """
        img = results['img']
        assert len(img.shape) == 3
        assert img.shape[2] == len(self.weights)
        weights = np.array(self.weights).reshape((1, 1, -1))
        img = (img * weights).sum(2, keepdims=True)
        if self.out_channels is None:
            img = img.repeat(weights.shape[2], axis=2)
        else:
            img = img.repeat(self.out_channels, axis=2)

        results['img'] = img
        results['img_shape'] = img.shape

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(out_channels={self.out_channels}, ' \
                    f'weights={self.weights})'
        return repr_str


@PIPELINES.register_module()
class AdjustGamma(object):
    """Using gamma correction to process the image.

    Args:
        gamma (float or int): Gamma value used in gamma correction.
            Default: 1.0.
    """

    def __init__(self, gamma=1.0):
        assert isinstance(gamma, float) or isinstance(gamma, int)
        assert gamma > 0
        self.gamma = gamma
        inv_gamma = 1.0 / gamma
        self.table = np.array([(i / 255.0)**inv_gamma * 255
                               for i in np.arange(256)]).astype('uint8')

    def __call__(self, results):
        """Call function to process the image with gamma correction.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Processed results.
        """

        results['img'] = mmcv.lut_transform(
            np.array(results['img'], dtype=np.uint8), self.table)

        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(gamma={self.gamma})'


@PIPELINES.register_module()
class SegRescale(object):
    """Rescale semantic segmentation maps.

    Args:
        scale_factor (float): The scale factor of the final output.
    """

    def __init__(self, scale_factor=1):
        self.scale_factor = scale_factor

    def __call__(self, results):
        """Call function to scale the semantic segmentation map.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with semantic segmentation map scaled.
        """
        for key in results.get('seg_fields', []):
            if self.scale_factor != 1:
                results[key] = mmcv.imrescale(
                    results[key], self.scale_factor, interpolation='nearest')
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(scale_factor={self.scale_factor})'


@PIPELINES.register_module()
class PhotoMetricDistortion(object):
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.

    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)

    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def convert(self, img, alpha=1, beta=0):
        """Multiple with alpha and add beat with clip."""
        img = img.astype(np.float32) * alpha + beta
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)

    def brightness(self, img):
        """Brightness distortion."""
        if random.randint(2):
            return self.convert(
                img,
                beta=random.uniform(-self.brightness_delta,
                                    self.brightness_delta))
        return img

    def contrast(self, img):
        """Contrast distortion."""
        if random.randint(2):
            return self.convert(
                img,
                alpha=random.uniform(self.contrast_lower, self.contrast_upper))
        return img

    def saturation(self, img):
        """Saturation distortion."""
        if random.randint(2):
            img = mmcv.bgr2hsv(img)
            img[:, :, 1] = self.convert(
                img[:, :, 1],
                alpha=random.uniform(self.saturation_lower,
                                     self.saturation_upper))
            img = mmcv.hsv2bgr(img)
        return img

    def hue(self, img):
        """Hue distortion."""
        if random.randint(2):
            img = mmcv.bgr2hsv(img)
            img[:, :,
                0] = (img[:, :, 0].astype(int) +
                      random.randint(-self.hue_delta, self.hue_delta)) % 180
            img = mmcv.hsv2bgr(img)
        return img

    def __call__(self, results):
        """Call function to perform photometric distortion on images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with images distorted.
        """

        img = results['img']
        # random brightness
        img = self.brightness(img)

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        mode = random.randint(2)
        if mode == 1:
            img = self.contrast(img)

        # random saturation
        img = self.saturation(img)

        # random hue
        img = self.hue(img)

        # random contrast
        if mode == 0:
            img = self.contrast(img)

        results['img'] = img
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(brightness_delta={self.brightness_delta}, '
                     f'contrast_range=({self.contrast_lower}, '
                     f'{self.contrast_upper}), '
                     f'saturation_range=({self.saturation_lower}, '
                     f'{self.saturation_upper}), '
                     f'hue_delta={self.hue_delta})')
        return repr_str


@PIPELINES.register_module()
class FDA(object):
    """Perform Fourier Domain Adaptation (FDA) with the source and target images.

    Added keys are src.img_stylized and src.bandwidth for (src, _) in keys.

    Args:
        bandwidth (float): Value for bandwidth of low-frequency band in FDA.
    """

    def __init__(self, bandwidth, keys=None):
        self.bandwidth = bandwidth
        self.keys = keys
        self.timer = Timer()

    def low_freq_mutate_np(self, a_src, a_trg):

        h, w, _ = a_src.shape
        b = (np.floor(np.amin((h,w)) * self.bandwidth)).astype(int)
        c_h = np.floor(h/2.0).astype(int)
        c_w = np.floor(w/2.0).astype(int)
        
        h_trg, w_trg, _ = a_trg.shape
        c_trg_h = np.floor(h_trg/2.0).astype(int)
        c_trg_w = np.floor(w_trg/2.0).astype(int)

        h1 = c_h-b
        h2 = c_h+b+1
        w1 = c_w-b
        w2 = c_w+b+1

        h1_trg = c_trg_h - b
        h2_trg = c_trg_h + b + 1
        w1_trg = c_trg_w - b
        w2_trg = c_trg_w + b + 1

        a_src[h1:h2,w1:w2,:] = a_trg[h1_trg:h2_trg,w1_trg:w2_trg,:]
        a_src = np.fft.ifftshift(a_src, axes=(-3, -2))
        return a_src

    def fda_source_to_target_np(self, src_img, trg_img):
        # exchange magnitude
        # input: src_img, trg_img

        # get fft of both source and target
        fft_src_np = np.fft.fft2(src_img, axes=(-3, -2))
        fft_trg_np = np.fft.fft2(trg_img, axes=(-3, -2))

        # extract amplitude and phase of both ffts
        amp_src, pha_src = np.abs(fft_src_np), np.angle(fft_src_np)
        amp_trg, pha_trg = np.abs(fft_trg_np), np.angle(fft_trg_np)
        
        a_src = np.fft.fftshift(amp_src, axes=(-3, -2))
        a_trg = np.fft.fftshift(amp_trg, axes=(-3, -2))

        # mutate the amplitude part of source with target
        amp_src_ = self.low_freq_mutate_np(a_src, a_trg)

        # mutated fft of source
        fft_src_ = amp_src_ * np.exp(1j * pha_src)

        # get the mutated image
        src_in_trg = np.fft.ifft2(fft_src_, axes=(-3, -2))
        src_in_trg = np.real(src_in_trg)

        return src_in_trg

    def __call__(self, results):
        """Call function to perform FDA.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """

        if self.keys is not None:
            for k in self.keys:
                src, trg = k
                results[src]['img_stylized'] = self.fda_source_to_target_np(
                    results[src]['img'],
                    results[trg]['img'])
                results[src]['bandwidth'] = self.bandwidth
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(bandwidth={self.bandwidth})'
        return repr_str

@PIPELINES.register_module()
class ReinhardTransfer(object):
    """Perform color transfer with the source and target images based on Reinhard's algorithm.
    E. Reinhard, M. Adhikhmin, B. Gooch, P. Shirley, "Color transfer
       between images," in IEEE Computer Graphics and Applications, vol.21,
       no.5, pp.34-41, 2001.

    Added key is src.img_stylized for (src, _) in keys.

    """

    def __init__(self, keys=None):
        self.keys = keys
        # Define conversion matrices.
        self.rgb2lms = np.array([[0.3811, 0.5783, 0.0402],
                            [0.1967, 0.7244, 0.0782],
                            [0.0241, 0.1288, 0.8444]])

        self.lms2lab = np.dot(
            np.array([[1 / (3**0.5), 0, 0],
                    [0, 1 / (6**0.5), 0],
                    [0, 0, 1 / (2**0.5)]]),
            np.array([[1, 1, 1],
                    [1, 1, -2],
                    [1, -1, 0]])
        )
        self.lms2rgb = np.linalg.inv(self.rgb2lms)
        self.lab2lms = np.linalg.inv(self.lms2lab)
        self.timer = Timer()

    def rgb_to_lab(self, im_rgb):
        """Transforms an image from RGB to LAB color space
        Parameters
        ----------
        im_rgb : array_like
            An RGB image
        Returns
        -------
        im_lab : array_like
            LAB representation of the input image `im_rgb`.
        References
        ----------
        D. Ruderman, T. Cronin, and C. Chiao, "Statistics of cone
        responses to natural images: implications for visual coding,"
        J. Opt. Soc. Am. A vol.15, pp.2036-2045, 1998.
        """

        # get input image dimensions
        m = im_rgb.shape[0]
        n = im_rgb.shape[1]

        # calculate im_lms values from RGB
        im_rgb = np.reshape(im_rgb, (m * n, 3))
        im_lms = np.dot(self.rgb2lms, np.transpose(im_rgb))
        im_lms[im_lms == 0] = np.spacing(1)

        # calculate LAB values from im_lms
        im_lab = np.dot(self.lms2lab, np.log(im_lms))

        # reshape to 3-channel image
        im_lab = np.reshape(im_lab.transpose(), (m, n, 3))

        return im_lab    
    
    def lab_to_rgb(self, im_lab):
        """Transforms an image from LAB to RGB color space
        Parameters
        ----------
        im_lab : array_like
            An image in LAB color space
        Returns
        -------
        im_rgb : array_like
            The RGB representation of the input image 'im_lab'.
        References
        ----------
        D. Ruderman, T. Cronin, and C. Chiao, "Statistics of cone
        responses to natural images: implications for visual coding,"
        J. Opt. Soc. Am. A 15, 2036-2045 (1998).
        """

        # get input image dimensions
        m = im_lab.shape[0]
        n = im_lab.shape[1]

        # calculate im_lms values from LAB
        im_lab = np.reshape(im_lab, (m * n, 3))
        im_lms = np.dot(self.lab2lms, np.transpose(im_lab))

        # calculate RGB values from im_lms
        im_lms = np.exp(im_lms)
        im_lms[im_lms == np.spacing(1)] = 0

        im_rgb = np.dot(self.lms2rgb, im_lms)

        # reshape to 3-channel image
        im_rgb = np.reshape(im_rgb.transpose(), (m, n, 3))

        return im_rgb
            
    def reinhard_source_to_target_np(self, src_img, trg_img):

        print(src_img.shape)
        print(trg_img.shape)

        # Convert source and target images to Ruderman's LAB color space.
        src_img_lab = self.rgb_to_lab(src_img)
        trg_img_lab = self.rgb_to_lab(trg_img)
        
        # Computer means and standard deviations per channel for LAB images.
        src_mu = np.mean(src_img_lab, axis=(-3, -2))
        src_sigma = np.std(src_img_lab, axis=(-3, -2))
        trg_mu = np.mean(trg_img_lab, axis=(-3, -2))
        trg_sigma = np.std(trg_img_lab, axis=(-3, -2))

        # Map the source LAB image to zero mean and unit variance and assign it the mean and variance of the target image.
        src_in_trg_lab = (src_img_lab - src_mu) * (trg_sigma / src_sigma) + trg_mu

        # Convert stylized source image back to RGB color space.
        src_in_trg = self.lab_to_rgb(src_in_trg_lab)
        
        # Clamp output RGB image to [0, 255].
        src_in_trg[src_in_trg < 0] = 0
        src_in_trg[src_in_trg > 255] = 255

        return src_in_trg

    def __call__(self, results):
        """Call function to perform Reinhard color transfer.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """

        print('Time since last check: {:6.3f} sec.'.format(self.timer.since_last_check()))
        if self.keys is not None:
            for k in self.keys:
                src, trg = k
                results[src]['img_stylized'] = self.reinhard_source_to_target_np(
                    results[src]['img'],
                    results[trg]['img'])
        print('Time spent on Reinhard: {:6.3f} sec.'.format(self.timer.since_last_check()))
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str
