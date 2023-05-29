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
# Specify GPU type.
#BSUB -R "select[gpu_model0==TITANRTX]"
#
# Specify file for logging standard output.
#BSUB -o ../logs/exp_50-csHR2acdcHR_hrda-%J.o
#
# Specify file for logging standard error.
#BSUB -e ../logs/exp_50-csHR2acdcHR_hrda-%J.e

/bin/echo Starting on: `date`

# Experiment ID.
EXP_ID="50"

# Specify directories.
export TMPDIR="${TMPDIR}"
export SOURCE_DIR="/cluster/home/csakarid/code/SysCV/CISS"
export SOURCE_DATASET="cityscapes"
export TARGET_DATASET="acdc"
export DIR_SOURCE_DATASET="${TMPDIR}/${SOURCE_DATASET}"
export DIR_TARGET_DATASET="${TMPDIR}/${TARGET_DATASET}"
export TAR_SOURCE_DATASET="/cluster/work/cvl/csakarid/data/Cityscapes/Cityscapes.tar.gz"
export TAR_TARGET_DATASET="/cluster/work/cvl/csakarid/data/ACDC/ACDC_splits.tar.gz"

# Perform initialization operations for the experiment.
cd ${SOURCE_DIR}
./experiments/scripts/initialization.sh # ${DIR_SOURCE_DATASET} ${DIR_TARGET_DATASET} ${TAR_SOURCE_DATASET} ${TAR_TARGET_DATASET} ${SOURCE_DATASET} ${TARGET_DATASET}

# Run the experiment.
python run_experiments.py --exp ${EXP_ID}

# Perform finalization operations.
./experiments/scripts/finalization.sh

/bin/echo Finished on: `date`

