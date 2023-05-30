# Condition-Invariant Semantic Segmentation

A novel unsupervised domain adaptation method for semantic segmentation models, tailored for condition-level adaptation

**by [Christos Sakaridis](https://people.ee.ethz.ch/csakarid/), [David Bruggemann](https://scholar.google.com/citations?user=uX2PrWMAAAAJ&hl=en&oi=ao), [Fisher Yu](https://www.yf.io/) and [Luc Van Gool](https://vision.ee.ethz.ch/people-details.OTAyMzM=.TGlzdC8zMjcxLC0xOTcxNDY1MTc4.html)**

**CISS**: [**Paper**][paper_pdf] | [**arXiv**][arxiv]

## Overview

Adaptation of semantic segmentation networks to different visual conditions from those for which ground-truth annotations are available at training is vital for robust perception in autonomous cars and robots. However, previous work has shown that most feature-level adaptation methods, which employ adversarial training and are validated on synthetic-to-real adaptation, provide marginal gains in normal-to-adverse condition-level adaptation, being outperformed by simple pixel-level adaptation via stylization. Motivated by these findings, we propose to leverage stylization in performing feature-level adaptation by aligning the deep features extracted by the encoder of the network from the original and the stylized view of each input image with a novel feature invariance loss. In this way, **we encourage the encoder to extract features that are invariant to the style of the input**, allowing the decoder to focus on parsing these features and not on further abstracting from the specific style of the input.

We implement our method, named **Condition-Invariant Semantic Segmentation (CISS)**, on the top-performing domain adaptation architecture and demonstrate a significant improvement over previous state-of-the-art methods both on **Cityscapes→ACDC** and **Cityscapes→Dark Zurich adaptation**. In particular, **CISS is ranked first among all published unsupervised domain adaptation (UDA) methods** on the [public ACDC leaderboard](https://acdc.vision.ee.ethz.ch/benchmarks#semanticSegmentation). Our method is also shown to generalize well to domains unseen during training, outperforming competing domain adaptation approaches on BDD100K-night and Nighttime Driving.

This repository includes the source code for CISS.

For more information on CISS, please check our [paper][paper_pdf].


## License

This software is made available for non-commercial use under a creative commons [license](LICENSE.txt). You can find a summary of the license [here][cc_license].


## Contents

1. [Requirements](#requirements)
2. [Datasets](#datasets-and-initial-weights)
3. [Testing](#testing)
4. [Training](#training)
5. [Checkpoints](#checkpoints)
6. [Framework Structure](#framework-structure)
7. [Acknowledgments](#acknowledgments)
8. [Citation](#citation)

## Requirements

For implementing CISS, we use two virtual Python environments, one for training and the other for testing. The former involves Pytorch 1.7.1 and Python 3.8.5, while the latter involves Pytorch 1.11.1 and Python 3.10.4. These virtual environments should be set up based on the provided `requirements.txt` and `requirements_test.txt` files as follows:

- Virtual environment for training:
```shell
python -m venv ~/venv/CISS
source ~/venv/CISS/bin/activate
pip install -r requirements_train.txt -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.3.7  # requires the other packages to be installed first
```

- Virtual environment for testing:
```shell
python -m venv ~/venv/CISS_test
source ~/venv/CISS_test/bin/activate
pip install -r requirements_test.txt -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.3.7  # requires the other packages to be installed first
```

We have run training of CISS successfully on NVIDIA Tesla V100-SXM2-32GB, NVIDIA A100 40GB PCIe, and NVIDIA A100 80GB PCIe.

Testing is less memory-intensive and we have run it successfully on the above GPUs and on NVIDIA GeForce RTX 3090 and NVIDIA TITAN RTX.

## Datasets and Initial Weights

**Cityscapes**: Download leftImg8bit_trainvaltest.zip and
gt_trainvaltest.zip from [here](https://www.cityscapes-dataset.com/downloads/)
and extract them to `data/cityscapes`.

**ACDC**: Download rgb_anon_trainvaltest.zip and
gt_trainval.zip from [here](https://acdc.vision.ee.ethz.ch/download) and
extract them to `data/acdc`. Further, please restructure the folders from
`condition/split/sequence/` to `split/` using the following commands:

```shell
rsync -a data/acdc/rgb_anon/*/train/*/* data/acdc/rgb_anon/train/
rsync -a data/acdc/rgb_anon/*/val/*/* data/acdc/rgb_anon/val/
rsync -a data/acdc/gt/*/train/*/*_labelTrainIds.png data/acdc/gt/train/
rsync -a data/acdc/gt/*/val/*/*_labelTrainIds.png data/acdc/gt/val/
```

**Dark Zurich**: Download the Dark_Zurich_train_anon.zip
and Dark_Zurich_val_anon.zip from
[here](https://www.trace.ethz.ch/publications/2019/GCMA_UIoU/) and extract it
to `data/dark_zurich`.

The final folder structure should look like this:

```none
CISS
├── ...
├── data
│   ├── acdc
│   │   ├── gt
│   │   │   ├── train
│   │   │   ├── val
│   │   ├── rgb_anon
│   │   │   ├── train
│   │   │   ├── val
│   ├── cityscapes
│   │   ├── leftImg8bit
│   │   │   ├── train
│   │   │   ├── val
│   │   ├── gtFine
│   │   │   ├── train
│   │   │   ├── val
│   ├── dark_zurich
│   │   ├── gt
│   │   │   ├── val
│   │   ├── rgb_anon
│   │   │   ├── train
│   │   │   ├── val
├── ...
```

For generalization evaluation, you need to also download **BDD100K-night** and **Nighttime Driving** which are used as test sets in that scenario.

**Data Preprocessing**: Finally, run the following scripts to convert the label IDs to the
train IDs and to generate the class index for RCS:

```shell
python tools/convert_datasets/cityscapes.py data/cityscapes --nproc 8
```

Further, please download the MiT weights from SegFormer using the
following script. If problems occur with the automatic download, please follow
the instructions for a manual download within the script.

```shell
sh tools/download_checkpoints.sh
```

## Testing

The provided CISS checkpoint trained on Cityscapes→ACDC (already downloaded by `tools/download_checkpoints.sh`) can be tested on the
Cityscapes validation set using:

```shell
sh test.sh work_dirs/csHR2acdcHR_ciss_9fcab
```

The predictions are saved for inspection to
`work_dirs/csHR2acdcHR_ciss_9fcab/preds`
and the mIoU of the model is printed to the console. The provided checkpoint
should achieve 68.67% mIoU on the validation set of ACDC. Refer to the end of
`work_dirs/csHR2acdcHR_ciss_9fcab/20230303_190813.log` for
more information, such as the IoU scores for individual classes.

The main results for Cityscapes→ACDC and Cityscapes→Dark Zurich in the paper are reported on
the test split of the respective target dataset. To generate the predictions for the test
set, please run:

```shell
source ~/venv/CISS_test/bin/activate
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
python -m tools.test path/to/config_file path/to/checkpoint_file --test-set --format-only --eval-options imgfile_prefix=path/to/trainids_dir to_label_id=False --show-dir path/to/preds_dir --opacity 1

```

The predictions can be submitted to the public evaluation server of the
respective dataset to obtain the test score.

## Training

For convenience, we provide an [annotated config file](configs/ciss/csHR2acdcHR_ciss.py) of the final CISS instantiation for the main, Cityscapes→ACDC experiment. A training job can be launched using:

```shell
source ~/venv/CISS/bin/activate
python run_experiments.py --config configs/ciss/csHR2acdcHR_ciss.py
```

The logs and checkpoints are stored under `work_dirs/`.

For the other experiments in our paper, we use a script to automatically generate the configs and train with them:

```shell
python run_experiments.py --exp <ID>
```

More information about the available experiments and their assigned IDs can be found in [experiments.py](experiments.py). The generated configs will be stored in `configs/generated/`.

Moreover, we provide bash scripts which prepare execution in compute clusters with SLURM and LSF architecture before calling the main training script [run_experiments.py](run_experiments.py). Users can adapt these scripts according to their own configuration of computing infrastructure.

## Checkpoints

Below, we provide checkpoints of CISS for different benchmarks.

* [CISS for Cityscapes→ACDC](https://data.vision.ee.ethz.ch/csakarid/shared/CISS/csHR2acdcHR_ciss_9fcab.tar.gz)
* [CISS for Cityscapes→Dark Zurich](https://data.vision.ee.ethz.ch/csakarid/shared/CISS/csHR2dzurHR_ciss_56306.tar.gz)

The checkpoints come with the training logs. Please note that:

* The logs provide the mIoU on validation sets. For Cityscapes→ACDC and Cityscapes→Dark Zurich the main results reported in the paper (besides those belonging to ablations studies) are calculated on the respective test sets. For Dark Zurich, the performance significantly differs between validation and test set. Please read the section above on how to obtain the test mIoU.

## Framework Structure

This repository is based on [mmsegmentation version 0.16.0](https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0). For more information about the framework structure and the config system, please refer to the [mmsegmentation documentation](https://mmsegmentation.readthedocs.io/en/latest/index.html) and the [mmcv documentation](https://mmcv.readthedocs.ihttps://arxiv.org/abs/2007.08702o/en/v1.3.7/index.html).

The most relevant files for CISS are:

* Model:
  - [mmseg/models/uda/dacs_ciss.py](mmseg/models/uda/dacs_ciss.py):
    Implementation of the feature invariance loss of CISS.
  - [mmseg/models/decode_heads/hrda_head.py](mmseg/models/decode_heads/hrda_head.py), [mmseg/models/segmentors/hrda_encoder_decoder.py](mmseg/models/segmentors/hrda_encoder_decoder.py):
    Implementation of upgraded HRDA decoder and head processing the encoder features used in CISS at multiple resolutions.
* Data:
  - [mmseg/datasets/custom_dual.py](mmseg/datasets/custom_dual.py), [mmseg/datasets/uda_dataset_dual.py](mmseg/datasets/uda_dataset_dual.py):
    Generic dataset classes that allow to consistently load pairs of images from two input datasets and process them jointly via stylization, in particular in the context of UDA where a source and a target dataset are used.
  - [mmseg/datasets/pipelines/transforms.py](mmseg/datasets/pipelines/transforms.py):
     Includes implementation of the two shallow stylization methods with which CISS is implemented, namely Fourier Domain Adaptation and the color transfer approach proposed by Reinhard et al. Stylization is implemented as part of the data loading process.
* Experiments and configurations:
  - [experiments.py](experiments.py):
    Implementation of all experiments with CISS which are presented in the [paper][paper_pdf] by specification of the proper configurations.
  - [configs/ciss/csHR2acdcHR_ciss.py](configs/ciss/csHR2acdcHR_ciss.py):
    Annotated config file for the final CISS instantiation.

Other relevant files are:
* [mmseg/models/segmentors/encoder_decoder.py](mmseg/models/segmentors/encoder_decoder.py): outputs additional features which are required by CISS
* [mmseg/datasets/pipelines/formating.py](mmseg/datasets/pipelines/formating.py), [mmseg/datasets/pipelines/loading.py](mmseg/datasets/pipelines/loading.py): modified implementation to enable during data loading a joint consistent processing of pairs of images where each member of the pair comes from a different dataset, as is the case with UDA, which involves a source and a target dataset.
* [configs/_base_/datasets/uda_cityscapesHR_to_acdcHR_1024x1024_fda.py](configs/_base_/datasets/uda_cityscapesHR_to_acdcHR_1024x1024_fda.py), [configs/_base_/datasets/uda_cityscapesHR_to_acdcHR_1024x1024_reinhard.py](configs/_base_/datasets/uda_cityscapesHR_to_acdcHR_1024x1024_reinhard.py): full configs for datasets in the main Cityscapes→ACDC experiment, with FDA-based stylization and Reinhard stylization respectively.
* [configs/_base_/uda/dacs_a999_fdthings_ciss_src_ceorig_inv_trg_ceorigorig_invorigorigstylizedstylized.py](configs/_base_/uda/dacs_a999_fdthings_ciss_src_ceorig_inv_trg_ceorigorig_invorigorigstylizedstylized.py): full UDA configuration for the generic fully-fledged version of CISS.

## Acknowledgments

This work is funded by Toyota Motor Europe via the research project TRACE-Zürich.

The source code for CISS is built on top of the following open-source projects. We thank their authors for making the respective source code repositories publicly available.

* [HRDA](https://github.com/lhoyer/HRDA)
* [DAFormer](https://github.com/lhoyer/DAFormer)
* [MMSegmentation](https://github.com/open-mmlab/mmsegmentation)
* [SegFormer](https://github.com/NVlabs/SegFormer)
* [DACS](https://github.com/vikolss/DACS)

## Citation

If you use the source code of CISS in your work or find CISS useful in your research in general, please cite our publication as

```
@article{sakaridis2023ciss,
  title={Condition-Invariant Semantic Segmentation},
  author={Sakaridis, Christos and Bruggemann, David and Yu, Fisher and Van Gool, Luc},
  journal = {ArXiv e-prints},
  archivePrefix = "arXiv",
  eprint = {2305.17349},
  primaryClass = "cs.CV",
  year=2023
}
```

[cc_license]: <http://creativecommons.org/licenses/by-nc/4.0/>
[paper_pdf]: <https://arxiv.org/pdf/2305.17349.pdf>
[project_page]: <https://www.trace.ethz.ch/publications/2023/CISS>
[arxiv]: <https://arxiv.org/abs/2305.17349>
