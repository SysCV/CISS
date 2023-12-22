# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
from .builder import DATASETS
from .cityscapes import CityscapesDataset


@DATASETS.register_module()
class ACDCrefDataset(CityscapesDataset):

    def __init__(self, **kwargs):
        super(ACDCrefDataset, self).__init__(
            img_suffix='_rgb_ref_anon.png',
            seg_map_suffix='_gt_ref_labelTrainIds.png',
            **kwargs)
        self.valid_mask_size = [1080, 1920]
