from .builder import DATASETS
from .cityscapes import CityscapesDataset


@DATASETS.register_module()
class NighttimeDrivingDataset(CityscapesDataset):
    def __init__(self, **kwargs):
        super().__init__(
            img_suffix='_leftImg8bit.png',
            seg_map_suffix='_gtCoarse_labelTrainIds.png',
            **kwargs)
