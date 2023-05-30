# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
from .builder import DATASETS
from .cityscapes import CityscapesDataset


@DATASETS.register_module()
class BDD100KDataset(CityscapesDataset):

    def __init__(self, **kwargs):
        super(BDD100KDataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='_train_id.png',
            **kwargs)
        self.valid_mask_size = [720, 1280]
