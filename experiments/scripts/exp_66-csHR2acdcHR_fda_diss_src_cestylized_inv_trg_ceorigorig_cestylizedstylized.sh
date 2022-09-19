#!/bin/bash
#
# Specify time limit.
#BSUB -W 48:00
#
# Specify number of CPU cores.
#BSUB -n 8
#
# Specify memory limit per CPU core.
#BSUB -R "rusage[mem=8192]"
#
# Specify disk limit on local scratch.
#BSUB -R "rusage[scratch=300000]"
#
# Specify number of required GPUs.
#BSUB -R "rusage[ngpus_excl_p=1]"
#
# Specify GPU type and number of required GPUs.
#BSUB -R "select[gpu_model0==TeslaV100_SXM2_32GB]"
#
# Specify file for logging standard output.
#BSUB -o ../logs/exp_66-csHR2acdcHR_fda_diss_src_cestylized_inv_trg_ceorigorig_cestylizedstylized.o
#
# Specify file for logging standard error.
#BSUB -e ../logs/exp_66-csHR2acdcHR_fda_diss_src_cestylized_inv_trg_ceorigorig_cestylizedstylized.e

/bin/echo Starting on: `date`

# Experiment ID.
EXP_ID="66"

# Specify directories.
export TMPDIR="${TMPDIR}"
export SOURCE_DIR="/cluster/home/csakarid/code/SysCV/DISS"
export SOURCE_DATASET="cityscapes"
export TARGET_DATASET="acdc"
export DIR_SOURCE_DATASET="${TMPDIR}/${SOURCE_DATASET}"
export DIR_TARGET_DATASET="${TMPDIR}/${TARGET_DATASET}"
export TAR_SOURCE_DATASET="/cluster/work/cvl/csakarid/data/Cityscapes/Cityscapes.tar.gz"
export TAR_TARGET_DATASET="/cluster/work/cvl/csakarid/data/ACDC/ACDC_splits.tar.gz"

# Perform initialization operations for the experiment.
cd ${SOURCE_DIR}
source /cluster/home/csakarid/DISS/bin/activate
./experiments/scripts/initialization.sh
python tools/convert_datasets/cityscapes.py ${DIR_SOURCE_DATASET} --nproc 8

# Run the experiment.
python run_experiments.py --exp ${EXP_ID}

# Deactivate virtual environment for DISS.
deactivate

/bin/echo Finished on: `date`

