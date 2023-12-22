# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications: Add class stats computation

import argparse
import json
import os.path as osp

import mmcv
import numpy as np
from PIL import Image


def sample_class_stats_from_label(label_file):

    if 'train/' in label_file:
        pil_label = Image.open(label_file)
        label = np.asarray(pil_label)
        sample_class_stats = {}
        for c in range(19):
            n = int(np.sum(label == c))
            if n > 0:
                sample_class_stats[int(c)] = n
        sample_class_stats['file'] = label_file
        return sample_class_stats
    else:
        return None


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert Cityscapes annotations to TrainIds')
    parser.add_argument('acdcref_path', help='ACDC ref data path')
    parser.add_argument('--gt-dir', default='gt', type=str)
    parser.add_argument('-o', '--out-dir', help='output path')
    parser.add_argument(
        '--nproc', default=1, type=int, help='number of process')
    args = parser.parse_args()
    return args


def save_class_stats(out_dir, sample_class_stats):
    sample_class_stats = [e for e in sample_class_stats if e is not None]
    with open(osp.join(out_dir, 'sample_class_stats.json'), 'w') as of:
        json.dump(sample_class_stats, of, indent=2)

    sample_class_stats_dict = {}
    for stats in sample_class_stats:
        f = stats.pop('file')
        sample_class_stats_dict[f] = stats
    with open(osp.join(out_dir, 'sample_class_stats_dict.json'), 'w') as of:
        json.dump(sample_class_stats_dict, of, indent=2)

    samples_with_class = {}
    for file, stats in sample_class_stats_dict.items():
        for c, n in stats.items():
            if c not in samples_with_class:
                samples_with_class[c] = [(file, n)]
            else:
                samples_with_class[c].append((file, n))
    with open(osp.join(out_dir, 'samples_with_class.json'), 'w') as of:
        json.dump(samples_with_class, of, indent=2)


def main():
    args = parse_args()
    acdcref_path = args.acdcref_path
    out_dir = args.out_dir if args.out_dir else acdcref_path
    mmcv.mkdir_or_exist(out_dir)

    gt_dir = osp.join(acdcref_path, args.gt_dir)

    label_files = []
    for gt_label in mmcv.scandir(gt_dir, '_gt_ref_labelTrainIds.png', recursive=True):
        gt_label_file = osp.join(gt_dir, gt_label)
        label_files.append(gt_label_file)

    only_postprocessing = False
    if not only_postprocessing:
        if args.nproc > 1:
            sample_class_stats = mmcv.track_parallel_progress(
                sample_class_stats_from_label, label_files, args.nproc)
        else:
            sample_class_stats = mmcv.track_progress(sample_class_stats_from_label,
                                                     label_files)
    else:
        with open(osp.join(out_dir, 'sample_class_stats.json'), 'r') as of:
            sample_class_stats = json.load(of)

    save_class_stats(out_dir, sample_class_stats)

    split_names = ['train', 'val']

    for split in split_names:
        filenames = []
        for gt_label in mmcv.scandir(
                osp.join(gt_dir, split), '_labelTrainIds.png', recursive=True):
            filenames.append(gt_label.replace('_gt_ref_labelTrainIds.png', ''))
        with open(osp.join(out_dir, f'{split}.txt'), 'w') as f:
            f.writelines(f + '\n' for f in filenames)


if __name__ == '__main__':
    main()
