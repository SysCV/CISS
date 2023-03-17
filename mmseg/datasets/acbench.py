import copy
import json
import os
import random
from collections import namedtuple
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from .builder import DATASETS
from .pipelines import Compose
from mmseg.core import eval_metrics
import mmcv
import numpy as np
from mmcv.utils import print_log
from prettytable import PrettyTable
from collections import OrderedDict

import numpy as np
import torch
from PIL import Image

# from acbench_transforms import pillow_interp_codes


@DATASETS.register_module()
class ACBenchDataset(torch.utils.data.Dataset):
    """ACBench
    """

    CLASSES = ('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
               'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
               'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
               'bicycle')

    PALETTE = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
               [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
               [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
               [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
               [0, 80, 100], [0, 0, 230], [119, 11, 32]]


    WildDashClass = namedtuple(
        "WildDashClass",
        ["name", "id", "train_id", "supercategory"],
    )
    labels = [
        #       name                             id    trainId     supercategory     
        WildDashClass(  'unlabeled'            ,  0 ,      255   , 'void'            ),
        WildDashClass(  'ego vehicle'          ,  1 ,      255   , 'vehicle'         ),
        WildDashClass(  'overlay'              ,  2 ,      255   , 'void'            ),
        WildDashClass(  'out of roi'           ,  3 ,      255   , 'void'            ),
        WildDashClass(  'static'               ,  4 ,      255   , 'void'            ),
        WildDashClass(  'dynamic'              ,  5 ,      255   , 'void'            ),
        WildDashClass(  'ground'               ,  6 ,      255   , 'void'            ),
        WildDashClass(  'road'                 ,  7 ,        0   , 'flat'            ),
        WildDashClass(  'sidewalk'             ,  8 ,        1   , 'flat'            ),
        WildDashClass(  'parking'              ,  9 ,      255   , 'flat'            ),
        WildDashClass(  'rail track'           , 10 ,      255   , 'flat'            ),
        WildDashClass(  'building'             , 11 ,        2   , 'construction'    ),
        WildDashClass(  'wall'                 , 12 ,        3   , 'construction'    ),
        WildDashClass(  'fence'                , 13 ,        4   , 'construction'    ),
        WildDashClass(  'guard rail'           , 14 ,      255   , 'construction'    ),
        WildDashClass(  'bridge'               , 15 ,      255   , 'construction'    ),
        WildDashClass(  'tunnel'               , 16 ,      255   , 'construction'    ),
        WildDashClass(  'pole'                 , 17 ,        5   , 'object'          ),
        WildDashClass(  'polegroup'            , 18 ,      255   , 'object'          ),
        WildDashClass(  'traffic light'        , 19 ,        6   , 'object'          ),
        WildDashClass(  'traffic sign front'   , 20 ,        7   , 'object'          ),
        WildDashClass(  'vegetation'           , 21 ,        8   , 'nature'          ),
        WildDashClass(  'terrain'              , 22 ,        9   , 'nature'          ),
        WildDashClass(  'sky'                  , 23 ,       10   , 'sky'             ),
        WildDashClass(  'person'               , 24 ,       11   , 'human'           ),
        WildDashClass(  'rider'                , 25 ,       12   , 'human'           ),
        WildDashClass(  'car'                  , 26 ,       13   , 'vehicle'         ),
        WildDashClass(  'truck'                , 27 ,       14   , 'vehicle'         ),
        WildDashClass(  'bus'                  , 28 ,       15   , 'vehicle'         ),
        WildDashClass(  'caravan'              , 29 ,      255   , 'vehicle'         ),
        WildDashClass(  'trailer'              , 30 ,      255   , 'vehicle'         ),
        WildDashClass(  'on rails'             , 31 ,       16   , 'vehicle'         ),
        WildDashClass(  'motorcycle'           , 32 ,       17   , 'vehicle'         ),
        WildDashClass(  'bicycle'              , 33 ,       18   , 'vehicle'         ),
        WildDashClass(  'pickup'               , 34 ,       14   , 'vehicle'         ),
        WildDashClass(  'van'                  , 35 ,       13   , 'vehicle'         ),
        WildDashClass(  'billboard'            , 36 ,      255   , 'object'          ),
        WildDashClass(  'street light'         , 37 ,      255   , 'object'          ),
        WildDashClass(  'road marking'         , 38 ,        0   , 'flat'            ),
        WildDashClass(  'junctionbox'          , 39 ,      255   , 'object'          ),
        WildDashClass(  'mailbox'              , 40 ,      255   , 'object'          ),
        WildDashClass(  'manhole'              , 41 ,        0   , 'object'          ),
        WildDashClass(  'phonebooth'           , 42 ,      255   , 'object'          ),
        WildDashClass(  'pothole'              , 43 ,        0   , 'object'          ),
        WildDashClass(  'bikerack'             , 44 ,      255   , 'object'          ),
        WildDashClass(  'traffic sign frame'   , 45 ,        5   , 'object'          ),
        WildDashClass(  'utility pole'         , 46 ,        5   , 'object'          ),
        WildDashClass(  'motorcyclist'         , 47 ,       12   , 'human'           ),
        WildDashClass(  'bicyclist'            , 48 ,       12   , 'human'           ),
        WildDashClass(  'other rider'          , 49 ,       12   , 'human'           ),
        WildDashClass(  'bird'                 , 50 ,      255   , 'nature'          ),
        WildDashClass(  'ground animal'        , 51 ,      255   , 'nature'          ),
        WildDashClass(  'curb'                 , 52 ,        1   , 'flat'            ),
        WildDashClass(  'traffic sign any'     , 53 ,      255   , 'object'          ),
        WildDashClass(  'traffic sign back'    , 54 ,      255   , 'object'          ),
        WildDashClass(  'trashcan'             , 55 ,      255   , 'object'          ),
        WildDashClass(  'other barrier'        , 56 ,        3   , 'object'          ),
        WildDashClass(  'other vehicle'        , 57 ,      255   , 'vehicle'         ),
        WildDashClass(  'auto rickshaw'        , 58 ,       17   , 'vehicle'         ),
        WildDashClass(  'bench'                , 59 ,      255   , 'object'          ),
        WildDashClass(  'mountain'             , 60 ,      255   , 'nature'          ),
        WildDashClass(  'tram track'           , 61 ,        0   , 'flat'            ),
        WildDashClass(  'wheeled slow'         , 62 ,      255   , 'vehicle'         ),
        WildDashClass(  'boat'                 , 63 ,      255   , 'vehicle'         ),
        WildDashClass(  'bikelane'             , 64 ,        0   , 'flat'            ),
        WildDashClass(  'bikelane sidewalk'    , 65 ,        1   , 'flat'            ),
        WildDashClass(  'banner'               , 66 ,      255   , 'object'          ),
        WildDashClass(  'dashcam mount'        , 67 ,      255   , 'vehicle'         ),
        WildDashClass(  'water'                , 68 ,      255   , 'flat'            ),
        WildDashClass(  'sand'                 , 69 ,      255   , 'flat'            ),
        WildDashClass(  'pedestrian area'      , 70 ,        0   , 'flat'            ),
        WildDashClass(  'fire hydrant'         , 71 ,      255   , 'object'          ),
        WildDashClass(  'cctv camera'          , 72 ,      255   , 'object'          ),
        WildDashClass(  'snow'                 , 73 ,      255   , 'flat'            ),
        WildDashClass(  'catch basin'          , 74 ,        0   , 'object'          ),
        WildDashClass(  'crosswalk plain'      , 75 ,        0   , 'flat'            ),
        WildDashClass(  'crosswalk zebra'      , 76 ,        0   , 'flat'            ),
        WildDashClass(  'manhole sidewalk'     , 77 ,        1   , 'object'          ),
        WildDashClass(  'curb terrain'         , 78 ,        9   , 'flat'            ),
        WildDashClass(  'service lane'         , 79 ,        0   , 'flat'            ),
        WildDashClass(  'curb cut'             , 80 ,        1   , 'flat'            ),
        WildDashClass(  'license plate'        , -1 ,       -1   , 'vehicle'         ),
    ]

    id_to_train_id = np.array([c.train_id for c in labels], dtype=int)

    def __init__(
            self,
            pipeline,
            data_root: str,
            stage: str = "test",
            conditions: List[str] = ["fog", "night", "rain", "snow"],
            load_keys: Union[List[str], str] = ["image", "semantic"],
            transforms: Optional[Callable] = None,
            ignore_index = 255,
            **kwargs
    ) -> None:
        super().__init__()
        self.root = data_root
        self.transforms = transforms
        self.pipeline = Compose(pipeline)
        self.ignore_index = ignore_index
        self.ann_dir = data_root
        self.label_map = None
        self.CLASSES, self.PALETTE = self.get_classes_and_palette()

        assert stage == 'test'
        self.stage = stage
        self.split = 'test'

        if isinstance(load_keys, str):
            self.load_keys = [load_keys]
        else:
            self.load_keys = load_keys

        pan = json.load(open(os.path.join(self.root, 'WildDash2', 'panoptic.json')))
        self.img_to_segments = {img_dict['file_name']: img_dict['segments_info'] for img_dict in pan['annotations']}

        self.conditions = conditions
        for condition in self.conditions: 
            print(f'\nAdding {condition} to dataset\n')
        self.paths = {k: [] for k in ['image', 'semantic', 'dataset']}
        self.paths_night = {k: [] for k in ['image', 'semantic', 'dataset']}
        self.paths_rain = {k: [] for k in ['image', 'semantic', 'dataset']}
        self.paths_snow = {k: [] for k in ['image', 'semantic', 'dataset']}
        self.paths_fog = {k: [] for k in ['image', 'semantic', 'dataset']}

        dirname = os.path.dirname(__file__)

        foggyzurich_img_ids = [i_id.strip() for i_id in open(os.path.join(dirname, 'list/ACBench_FoggyZurich_fog'))]
        for file_name in foggyzurich_img_ids:
            file_path = os.path.join(self.root, 'Foggy_Zurich', file_name)
            semantic_path = file_path.replace('/RGB/', '/gt_labelTrainIds/')
            self.paths_fog['image'].append(file_path)
            self.paths_fog['semantic'].append(semantic_path)
            self.paths_fog['dataset'].append('foggyzurich')

        bdd100k_img_ids_snow = [i_id.strip() for i_id in open(os.path.join(dirname, 'list/ACBench_BDD100k_snow'))]
        for file_name in bdd100k_img_ids_snow:
            file_path = os.path.join(self.root, 'bdd100k', file_name)
            semantic_name = file_name.replace('images/10k/', 'labels/sem_seg/masks/')
            semantic_name = semantic_name.replace('.jpg', '.png')
            semantic_path = os.path.join(self.root, 'bdd100k', semantic_name)
            self.paths_snow['image'].append(file_path)
            self.paths_snow['semantic'].append(semantic_path)
            self.paths_snow['dataset'].append('bdd100k')

        wilddash_img_ids_fog = [i_id.strip() for i_id in open(os.path.join(dirname, 'list/ACBench_WildDash_fog'))]
        for file_name in wilddash_img_ids_fog:
            file_path = os.path.join(self.root, 'WildDash2', file_name)
            semantic_name = file_name.replace('images/', 'panoptic/')
            semantic_name = semantic_name.replace('.jpg', '.png')
            semantic_path = os.path.join(self.root, 'WildDash2', semantic_name)
            self.paths_fog['image'].append(file_path)
            self.paths_fog['semantic'].append(semantic_path)
            self.paths_fog['dataset'].append('wilddash')

        wilddash_img_ids_night = [i_id.strip() for i_id in open(os.path.join(dirname, 'list/ACBench_WildDash_night'))]
        for file_name in wilddash_img_ids_night:
            file_path = os.path.join(self.root, 'WildDash2', file_name)
            semantic_name = file_name.replace('images/', 'panoptic/')
            semantic_name = semantic_name.replace('.jpg', '.png')
            semantic_path = os.path.join(self.root, 'WildDash2', semantic_name)
            self.paths_night['image'].append(file_path)
            self.paths_night['semantic'].append(semantic_path)
            self.paths_night['dataset'].append('wilddash')
        
        wilddash_img_ids_rain = [i_id.strip() for i_id in open(os.path.join(dirname, 'list/ACBench_WildDash_rain'))]
        for file_name in wilddash_img_ids_rain:
            file_path = os.path.join(self.root, 'WildDash2', file_name)
            semantic_name = file_name.replace('images/', 'panoptic/')
            semantic_name = semantic_name.replace('.jpg', '.png')
            semantic_path = os.path.join(self.root, 'WildDash2', semantic_name)
            self.paths_rain['image'].append(file_path)
            self.paths_rain['semantic'].append(semantic_path)
            self.paths_rain['dataset'].append('wilddash')

        wilddash_img_ids_snow = [i_id.strip() for i_id in open(os.path.join(dirname, 'list/ACBench_WildDash_snow'))]
        for file_name in wilddash_img_ids_snow:
            file_path = os.path.join(self.root, 'WildDash2', file_name)
            semantic_name = file_name.replace('images/', 'panoptic/')
            semantic_name = semantic_name.replace('.jpg', '.png')
            semantic_path = os.path.join(self.root, 'WildDash2', semantic_name)
            self.paths_snow['image'].append(file_path)
            self.paths_snow['semantic'].append(semantic_path)
            self.paths_snow['dataset'].append('wilddash')

        bdd100k_img_ids_fog = [i_id.strip() for i_id in open(os.path.join(dirname, 'list/ACBench_BDD100k_fog'))]
        for file_name in bdd100k_img_ids_fog:
            file_path = os.path.join(self.root, 'bdd100k', file_name)
            semantic_name = file_name.replace('images/10k/', 'labels/sem_seg/masks/')
            semantic_name = semantic_name.replace('.jpg', '.png')
            semantic_path = os.path.join(self.root, 'bdd100k', semantic_name)
            self.paths_fog['image'].append(file_path)
            self.paths_fog['semantic'].append(semantic_path)
            self.paths_fog['dataset'].append('bdd100k')

        bdd100k_img_ids_night = [i_id.strip() for i_id in open(os.path.join(dirname, 'list/ACBench_BDD100k_night'))]
        for file_name in bdd100k_img_ids_night:
            file_path = os.path.join(self.root, 'bdd100k', file_name)
            semantic_name = file_name.replace('images/10k/', 'labels/sem_seg/masks/')
            semantic_name = semantic_name.replace('.jpg', '.png')
            semantic_path = os.path.join(self.root, 'bdd100k', semantic_name)
            self.paths_night['image'].append(file_path)
            self.paths_night['semantic'].append(semantic_path)
            self.paths_night['dataset'].append('bdd100k')

        bdd100k_img_ids_rain = [i_id.strip() for i_id in open(os.path.join(dirname, 'list/ACBench_BDD100k_rain'))]
        for file_name in bdd100k_img_ids_rain:
            file_path = os.path.join(self.root, 'bdd100k', file_name)
            semantic_name = file_name.replace('images/10k/', 'labels/sem_seg/masks/')
            semantic_name = semantic_name.replace('.jpg', '.png')
            semantic_path = os.path.join(self.root, 'bdd100k', semantic_name)
            self.paths_rain['image'].append(file_path)
            self.paths_rain['semantic'].append(semantic_path)
            self.paths_rain['dataset'].append('bdd100k')

        foggydriving_img_ids = [i_id.strip() for i_id in open(os.path.join(dirname, 'list/ACBench_FoggyDriving_fog'))]
        for file_name in foggydriving_img_ids:
            file_path = os.path.join(self.root, 'Foggy_Driving', file_name)
            if 'test_extra' in file_name:
                semantic_name = file_name.replace('leftImg8bit/test_extra/', 'gtCoarse/test_extra/')
                semantic_name = semantic_name.replace('_leftImg8bit.png', '_gtCoarse_labelTrainIds.png')
            else:
                semantic_name = file_name.replace('leftImg8bit/test/', 'gtFine/test/')
                semantic_name = semantic_name.replace('_leftImg8bit.png', '_gtFine_labelTrainIds.png')
            semantic_path = os.path.join(self.root, 'Foggy_Driving', semantic_name)
            self.paths_fog['image'].append(file_path)
            self.paths_fog['semantic'].append(semantic_path)
            self.paths_fog['dataset'].append('foggydriving')

        custom_img_ids = [
            'train_000001.jpg',
            'train_000002.jpg',
            'train_000003.jpg',
        ]
        for file_name in custom_img_ids:
            file_path = os.path.join(dirname, 'list', file_name)
            semantic_path = file_path.replace('.jpg', '_mask.png')
            self.paths_rain['image'].append(file_path)
            self.paths_rain['semantic'].append(semantic_path)
            self.paths_rain['dataset'].append('custom')

        self.len_fog = len(self.paths_fog['image'])
        self.len_night = len(self.paths_night['image'])
        self.len_rain = len(self.paths_rain['image'])
        self.len_snow = len(self.paths_snow['image'])

        for c in self.conditions:
            if c == "fog":
                self.paths['image'].extend(self.paths_fog['image'])
                self.paths['semantic'].extend(self.paths_fog['semantic'])
                self.paths['dataset'].extend(self.paths_fog['dataset'])
            elif c == "night":
                self.paths['image'].extend(self.paths_night['image'])
                self.paths['semantic'].extend(self.paths_night['semantic'])
                self.paths['dataset'].extend(self.paths_night['dataset'])
            elif c == "rain":
                self.paths['image'].extend(self.paths_rain['image'])
                self.paths['semantic'].extend(self.paths_rain['semantic'])
                self.paths['dataset'].extend(self.paths_rain['dataset'])
            elif c == "snow":
                self.paths['image'].extend(self.paths_snow['image'])
                self.paths['semantic'].extend(self.paths_snow['semantic'])
                self.paths['dataset'].extend(self.paths_snow['dataset'])
                            
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item. Otherwise target is a json object if target_type="polygon", else the image segmentation.
        """

        sample: Any = {}

        filename = self.paths['image'][index].split('/')[-1]
        sample['filename'] = filename

        dataset = self.paths['dataset'][index]
        for k in self.load_keys:
            if k == 'image':
                data = Image.open(self.paths[k][index]).convert('RGB')
            elif k == 'semantic':
                data = Image.open(self.paths[k][index])
                if dataset == 'wilddash':
                    data = self.encode_semantic_map(
                        data, filename.replace('.jpg', '.png'))

            else:
                raise ValueError('invalid load_key')
            sample[k] = data

        if self.transforms is not None:
            sample = self.transforms(sample)

        img_info = dict(ann=dict(seg_map=self.paths["semantic"][index]),
        filename=self.paths["image"][index])
        results = dict(img_info=img_info)
        self.pre_pipeline(results)
        results = self.pipeline(results)
        return results


    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['seg_fields'] = []
        results['img_prefix'] = None #self.img_dir
        results['seg_prefix'] = None #self.ann_dir

    def __len__(self) -> int:
        return len(next(iter(self.paths.values())))

    def encode_semantic_map(self, semseg, filename):
        pan_format = np.array(semseg, dtype=np.uint32)
        pan = self.rgb2id(pan_format)
        semantic = np.zeros(pan.shape, dtype=np.uint8)
        for segm_info in self.img_to_segments[filename]:
            semantic[pan == segm_info['id']] = segm_info['category_id']
        semantic = self.id_to_train_id[semantic.astype(int)]
        return semantic.astype(np.uint8)

    @staticmethod
    def rgb2id(color):
        if isinstance(color, np.ndarray) and len(color.shape) == 3:
            if color.dtype == np.uint8:
                color = color.astype(np.int32)
            return color[:, :, 0] + 256 * color[:, :, 1] + 256 * 256 * color[:, :, 2]
        return int(color[0] + 256 * color[1] + 256 * 256 * color[2])

    def evaluate(self,
                 results,
                 metric='mIoU',
                 logger=None,
                 imgfile_prefix=None,
                 efficient_test=False):
        """Evaluation in Cityscapes/default protocol.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            imgfile_prefix (str | None): The prefix of output image file,
                for cityscapes evaluation only. It includes the file path and
                the prefix of filename, e.g., "a/b/prefix".
                If results are evaluated with cityscapes protocol, it would be
                the prefix of output png files. The output files would be
                png images under folder "a/b/prefix/xxx.png", where "xxx" is
                the image name of cityscapes. If not specified, a temp file
                will be created for evaluation.
                Default: None.

        Returns:
            dict[str, float]: Cityscapes/default metrics.
        """

        eval_results = dict()
        metrics = metric.copy() if isinstance(metric, list) else [metric]
        eval_results.update(
            self.standard_evaluate(results, metrics, logger, efficient_test))

        return eval_results

    def standard_evaluate(self,
                 results,
                 metric='mIoU',
                 logger=None,
                 efficient_test=False,
                 **kwargs):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. 'mIoU',
                'mDice' and 'mFscore' are supported.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.

        Returns:
            dict[str, float]: Default metrics.
        """

        if isinstance(metric, str):
            metric = [metric]
        allowed_metrics = ['mIoU', 'mDice', 'mFscore']
        if not set(metric).issubset(set(allowed_metrics)):
            raise KeyError('metric {} is not supported'.format(metric))
        eval_results = {}
        gt_seg_maps = self.get_gt_seg_maps(efficient_test)
        if self.CLASSES is None:
            num_classes = len(
                reduce(np.union1d, [np.unique(_) for _ in gt_seg_maps]))
        else:
            num_classes = len(self.CLASSES)
        ret_metrics = eval_metrics(
            results,
            gt_seg_maps,
            num_classes,
            self.ignore_index,
            metric,
            label_map=self.label_map,
            reduce_zero_label=False)

        if self.CLASSES is None:
            class_names = tuple(range(num_classes))
        else:
            class_names = self.CLASSES

        # summary table
        ret_metrics_summary = OrderedDict({
            ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })

        # each class table
        ret_metrics.pop('aAcc', None)
        ret_metrics_class = OrderedDict({
            ret_metric: np.round(ret_metric_value * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })
        ret_metrics_class.update({'Class': class_names})
        ret_metrics_class.move_to_end('Class', last=False)

        # for logger
        class_table_data = PrettyTable()
        for key, val in ret_metrics_class.items():
            class_table_data.add_column(key, val)

        summary_table_data = PrettyTable()
        for key, val in ret_metrics_summary.items():
            if key == 'aAcc':
                summary_table_data.add_column(key, [val])
            else:
                summary_table_data.add_column('m' + key, [val])

        print_log('per class results:', logger)
        print_log('\n' + class_table_data.get_string(), logger=logger)
        print_log('Summary:', logger)
        print_log('\n' + summary_table_data.get_string(), logger=logger)

        # each metric dict
        for key, value in ret_metrics_summary.items():
            if key == 'aAcc':
                eval_results[key] = value / 100.0
            else:
                eval_results['m' + key] = value / 100.0

        ret_metrics_class.pop('Class', None)
        for key, value in ret_metrics_class.items():
            eval_results.update({
                key + '.' + str(name): value[idx] / 100.0
                for idx, name in enumerate(class_names)
            })

        if mmcv.is_list_of(results, str):
            for file_name in results:
                os.remove(file_name)
        return eval_results


    def get_gt_seg_maps(self, efficient_test=False):
        """Get ground truth segmentation maps for evaluation."""
        gt_seg_maps = []
        for seg_map, dataset in zip(self.paths['semantic'], self.paths['dataset']): # TODO: here I have to load it 
            if efficient_test:
                gt_seg_map = seg_map
            else:
                gt_seg_map = mmcv.imread(
                    seg_map, flag='unchanged', backend='pillow')
                if  dataset == 'wilddash':
                    gt_seg_map = self.encode_semantic_map(
                        gt_seg_map, seg_map.split('/')[-1].replace('.jpg', '.png'))
                
            gt_seg_maps.append(gt_seg_map)
        return gt_seg_maps

    def results2img(self, results, imgfile_prefix, to_label_id):
        """Write the segmentation results to images.

        Args:
            results (list[list | tuple | ndarray]): Testing results of the
                dataset.
            imgfile_prefix (str): The filename prefix of the png files.
                If the prefix is "somepath/xxx",
                the png files will be named "somepath/xxx.png".
            to_label_id (bool): whether convert output to label_id for
                submission

        Returns:
            list[str: str]: result txt files which contains corresponding
            semantic segmentation images.
        """
        mmcv.mkdir_or_exist(imgfile_prefix)
        result_files = []
        prog_bar = mmcv.ProgressBar(len(self))
        for idx in range(len(self)):
            result = results[idx]
            if to_label_id:
                result = self._convert_to_label_id(result)
            filename = self.paths["image"][idx] # self.img_infos[idx]['filename']
            basename = osp.splitext(osp.basename(filename))[0]

            png_filename = osp.join(imgfile_prefix, f'{basename}.png')

            output = Image.fromarray(result.astype(np.uint8)).convert('P')
            import cityscapesscripts.helpers.labels as CSLabels
            palette = np.zeros((len(CSLabels.id2label), 3), dtype=np.uint8)
            if to_label_id:
                for label_id, label in CSLabels.id2label.items():
                    palette[label_id] = label.color
            else:
                palette = np.array(self.PALETTE, dtype=np.uint8)

            output.putpalette(palette)
            output.save(png_filename)
            result_files.append(png_filename)
            prog_bar.update()

        return result_files

    def format_results(self, results, imgfile_prefix=None, to_label_id=True):
        """Format the results into dir (standard format for Cityscapes
        evaluation).

        Args:
            results (list): Testing results of the dataset.
            imgfile_prefix (str | None): The prefix of images files. It
                includes the file path and the prefix of filename, e.g.,
                "a/b/prefix". If not specified, a temp file will be created.
                Default: None.
            to_label_id (bool): whether convert output to label_id for
                submission. Default: False

        Returns:
            tuple: (result_files, tmp_dir), result_files is a list containing
                the image paths, tmp_dir is the temporal directory created
                for saving json/png files when img_prefix is not specified.
        """

        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: '
            f'{len(results)} != {len(self)}')

        if imgfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            imgfile_prefix = tmp_dir.name
        else:
            tmp_dir = None
        result_files = self.results2img(results, imgfile_prefix, to_label_id)

        return result_files, tmp_dir


    @staticmethod
    def _convert_to_label_id(result):
        """Convert trainId to id for cityscapes."""
        if isinstance(result, str):
            result = np.load(result)
        import cityscapesscripts.helpers.labels as CSLabels
        result_copy = result.copy()
        for trainId, label in CSLabels.trainId2label.items():
            result_copy[result == trainId] = label.id

        return result_copy

    def get_classes_and_palette(self, classes=None, palette=None):
        """Get class names of current dataset.

        Args:
            classes (Sequence[str] | str | None): If classes is None, use
                default CLASSES defined by builtin dataset. If classes is a
                string, take it as a file name. The file contains the name of
                classes where each line contains one class name. If classes is
                a tuple or list, override the CLASSES defined by the dataset.
            palette (Sequence[Sequence[int]]] | np.ndarray | None):
                The palette of segmentation map. If None is given, random
                palette will be generated. Default: None
        """
        if classes is None:
            self.custom_classes = False
            return self.CLASSES, self.PALETTE

        self.custom_classes = True
        if isinstance(classes, str):
            # take it as a file path
            class_names = mmcv.list_from_file(classes)
        elif isinstance(classes, (tuple, list)):
            class_names = classes
        else:
            raise ValueError(f'Unsupported type {type(classes)} of classes.')

        if self.CLASSES:
            if not set(classes).issubset(self.CLASSES):
                raise ValueError('classes is not a subset of CLASSES.')

            # dictionary, its keys are the old label ids and its values
            # are the new label ids.
            # used for changing pixel labels in load_annotations.
            self.label_map = {}
            for i, c in enumerate(self.CLASSES):
                if c not in class_names:
                    self.label_map[i] = -1
                else:
                    self.label_map[i] = classes.index(c)

        palette = self.get_palette_for_custom_classes(class_names, palette)

        return class_names, palette
