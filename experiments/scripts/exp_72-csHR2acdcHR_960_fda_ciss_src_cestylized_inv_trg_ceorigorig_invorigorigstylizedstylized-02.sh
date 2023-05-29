#!/bin/bash
#
# Specify time limit.
#BSUB -W 120:00
#
# Specify number of CPU cores.
#BSUB -n 8
#
# Specify memory limit per CPU core.
#BSUB -R "rusage[mem=8192]"
#
# Specify disk limit on local scratch.
#BSUB -R "rusage[scratch=150000]"
#
# Specify number of required GPUs.
#BSUB -R "rusage[ngpus_excl_p=1]"
#
# Specify GPU type and number of required GPUs.
#BSUB -R "select[gpu_mtotal0>=30000]"
#
# Specify file for logging standard output.
#BSUB -o ../logs/exp_72-csHR2acdcHR_960_fda_ciss_src_cestylized_inv_trg_ceorigorig_invorigorigstylizedstylized-02-02.o
#
# Specify file for logging standard error.
#BSUB -e ../logs/exp_72-csHR2acdcHR_960_fda_ciss_src_cestylized_inv_trg_ceorigorig_invorigorigstylizedstylized-02-02.e
#
# Specify jobname.
#BSUB -J exp_72-csHR2acdcHR_960_fda_ciss_src_cestylized_inv_trg_ceorigorig_invorigorigstylizedstylized-02-02
#
# Specify job for which we need to wait before running.
#BSUB -w ended(exp_72-csHR2acdcHR_960_fda_ciss_src_cestylized_inv_trg_ceorigorig_invorigorigstylizedstylized-02-01)

/bin/echo Starting on: `date`

# Experiment ID.
EXP_ID="72"

# Specify directories.
export TMPDIR="${TMPDIR}"
export SOURCE_DIR="/cluster/home/csakarid/code/SysCV/CISS"
export SOURCE_DATASET="cityscapes"
export TARGET_DATASET="acdc"
export DIR_SOURCE_DATASET="${TMPDIR}/${SOURCE_DATASET}"
export DIR_TARGET_DATASET="${TMPDIR}/${TARGET_DATASET}"
export TAR_SOURCE_DATASET="/cluster/work/cvl/csakarid/data/Cityscapes/Cityscapes.tar.gz"
export TAR_TARGET_DATASET="/cluster/work/cvl/csakarid/data/ACDC/ACDC_splits.tar.gz"

# Specify checkpoint to resume from.
export CHECKPOINT_RESUME="/cluster/work/cvl/csakarid/results/CISS/local-exp"${EXP_ID}"//latest.pth"

# Perform initialization operations for the experiment.
cd ${SOURCE_DIR}
source /cluster/home/csakarid/CISS/bin/activate
./experiments/scripts/initialization.sh
python tools/convert_datasets/cityscapes.py ${DIR_SOURCE_DATASET} --nproc 8

# Run the experiment.
python run_experiments.py --exp ${EXP_ID} --resume-from ${CHECKPOINT_RESUME}

# Deactivate virtual environment for CISS.
deactivate

/bin/echo Finished on: `date`

