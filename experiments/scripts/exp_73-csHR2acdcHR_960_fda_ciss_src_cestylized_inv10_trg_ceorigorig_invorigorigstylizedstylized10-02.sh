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
#BSUB -R "rusage[scratch=40000]"
#
# Specify number of required GPUs.
#BSUB -R "rusage[ngpus_excl_p=1]"
#
# Specify GPU type and number of required GPUs.
#BSUB -R "select[gpu_mtotal0>=30000]"
#
# Specify file for logging standard output.
#BSUB -o ../logs/exp_73-csHR2acdcHR_960_fda_diss_src_cestylized_inv10_trg_ceorigorig_invorigorigstylizedstylized10-02-02.o
#
# Specify file for logging standard error.
#BSUB -e ../logs/exp_73-csHR2acdcHR_960_fda_diss_src_cestylized_inv10_trg_ceorigorig_invorigorigstylizedstylized10-02-02.e
#
# Specify jobname.
#BSUB -J exp_73-csHR2acdcHR_960_fda_diss_src_cestylized_inv10_trg_ceorigorig_invorigorigstylizedstylized10-02-02

/bin/echo Starting on: `date`

# Experiment ID.
EXP_ID="73"

# Specify directories.
export TMPDIR="${TMPDIR}"
export SOURCE_DIR="/cluster/home/csakarid/code/SysCV/DISS"
export SOURCE_DATASET="cityscapes"
export TARGET_DATASET="acdc"
export DIR_SOURCE_DATASET="${TMPDIR}/${SOURCE_DATASET}"
export DIR_TARGET_DATASET="${TMPDIR}/${TARGET_DATASET}"
export TAR_SOURCE_DATASET="/cluster/work/cvl/csakarid/data/Cityscapes/Cityscapes.tar.gz"
export TAR_TARGET_DATASET="/cluster/work/cvl/csakarid/data/ACDC/ACDC_splits.tar.gz"

# Specify checkpoint to resume from.
export CHECKPOINT_RESUME="/cluster/work/cvl/csakarid/results/DISS/local-exp"${EXP_ID}"/220921_1738_csHR2acdcHR_960x960_dacs_a999_fdthings_diss_src_cestylized_inv10_trg_ceorigorig_invorigorigstylizedstylized10_rcs001-20_hrda1-480-01_daformer_sepaspp_sl_mitb5_poly10warm_s0_14671/latest.pth"

# Perform initialization operations for the experiment.
cd ${SOURCE_DIR}
source /cluster/home/csakarid/DISS/bin/activate
./experiments/scripts/initialization.sh
python tools/convert_datasets/cityscapes.py ${DIR_SOURCE_DATASET} --nproc 8

# Run the experiment.
python run_experiments.py --exp ${EXP_ID} --resume-from ${CHECKPOINT_RESUME}

# Deactivate virtual environment for DISS.
deactivate

/bin/echo Finished on: `date`

