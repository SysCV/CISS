# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
#!/bin/bash

# Instructions for Manual Download:
#
# Please, download the [MiT weights](https://drive.google.com/drive/folders/1b7bwrInTW4VLEm27YawHOAMSMikga2Ia?usp=sharing)
# pretrained on ImageNet-1K provided by the official
# [SegFormer repository](https://github.com/NVlabs/SegFormer) and put them in a
# folder `pretrained/` within this project. Only mit_b5.pth is necessary.
#
# Please, download the checkpoint of CISS on Cityscapes->ACDC from
# [here](https://data.vision.ee.ethz.ch/csakarid/shared/CISS/csHR2acdcHR_ciss_9fcab.tar.gz).
# and extract it to `work_dirs/csHR2acdcHR_ciss_9fcab/`

# Automatic Downloads:
set -e  # exit when any command fails
mkdir -p pretrained/
cd pretrained/
gdown --id 1d7I50jVjtCddnhpf-lqj8-f13UyCzoW1  # MiT-B5 weights
cd ../

mkdir -p work_dirs/
cd work_dirs/
mkdir csHR2acdcHR_ciss_9fcab/
cd csHR2acdcHR_ciss_9fcab/
curl -O https://data.vision.ee.ethz.ch/csakarid/shared/CISS/csHR2acdcHR_ciss_9fcab.tar.gz  # CISS on Cityscapes->ACDC
tar -xzf csHR2acdcHR_ciss_9fcab.tar.gz
rm csHR2acdcHR_ciss_9fcab.tar.gz
cd ../../
