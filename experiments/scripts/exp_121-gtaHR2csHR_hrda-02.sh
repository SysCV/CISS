#!/bin/bash
#
# Specify time limit.
#BSUB -W 48:00
#
# Specify memory limit.
#BSUB -R "rusage[mem=30000]"
#
# Specify disk limit on local scratch.
#BSUB -R "rusage[scratch=200000]"
#
# Specify number of required GPUs.
#BSUB -R "rusage[ngpus_excl_p=1]"
#
# Specify GPU memory.
#BSUB -R "select[gpu_mtotal0>=30000]"
#
# Specify file for logging standard output.
#BSUB -o ../logs/exp_121-gtaHR2csHR_hrda-02-01.o
#
# Specify file for logging standard error.
#BSUB -e ../logs/exp_121-gtaHR2csHR_hrda-02-01.e

/bin/echo Starting on: `date`

# Experiment ID.
EXP_ID="121"

# Specify directories.
export TMPDIR="${TMPDIR}"
export SOURCE_DIR="/cluster/home/csakarid/code/SysCV/CISS"
export SOURCE_DATASET="gta"
export TARGET_DATASET="cityscapes"
export DIR_SOURCE_DATASET="${TMPDIR}/${SOURCE_DATASET}"
export DIR_TARGET_DATASET="${TMPDIR}/${TARGET_DATASET}"
export TAR_SOURCE_DATASET="/cluster/work/cvl/csakarid/data/GTA5/GTA5.tar.gz"
export TAR_TARGET_DATASET="/cluster/work/cvl/csakarid/data/Cityscapes/Cityscapes.tar.gz"

# Perform initialization operations for the experiment.
cd ${SOURCE_DIR}
source /cluster/home/csakarid/CISS/bin/activate
./experiments/scripts/initialization.sh
python tools/convert_datasets/gta.py ${DIR_SOURCE_DATASET} --nproc 8
python tools/convert_datasets/cityscapes.py ${DIR_TARGET_DATASET} --nproc 8

# Run the experiment.
python run_experiments.py --exp ${EXP_ID}

/bin/echo Finished on: `date`

