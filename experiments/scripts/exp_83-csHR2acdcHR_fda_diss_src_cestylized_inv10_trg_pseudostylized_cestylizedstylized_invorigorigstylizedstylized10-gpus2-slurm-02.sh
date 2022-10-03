#!/bin/bash
#
# Specify time limit.
#SBATCH --time=120:00:00
#
# Specify number of CPU cores.
#SBATCH -n 8
#
# Specify memory limit per CPU core.
#SBATCH --mem-per-cpu=8192
#
# Specify disk limit on local scratch.
#SBATCH --tmp=300000
#
# Specify GPU type and number of required GPUs.
#SBATCH --gpus=rtx_3090:2
#
# Specify file for logging standard output.
#SBATCH --output=../logs/exp_83-csHR2acdcHR_fda_diss_src_cestylized_inv10_trg_pseudostylized_cestylizedstylized_invorigorigstylizedstylized10-gpus2-slurm-01-02.o
#
# Specify file for logging standard error.
#SBATCH --error=../logs/exp_83-csHR2acdcHR_fda_diss_src_cestylized_inv10_trg_pseudostylized_cestylizedstylized_invorigorigstylizedstylized10-gpus2-slurm-01-02.e
#
# Specify open mode for log files.
#SBATCH --open-mode=append
#
# Specify jobname.
#SBATCH --job-name=exp_83-csHR2acdcHR_fda_diss_src_cestylized_inv10_trg_pseudostylized_cestylizedstylized_invorigorigstylizedstylized10-gpus2-slurm
#
# Specify dependency.
#SBATCH --dependency=singleton

/bin/echo Starting on: `date`

# Experiment ID.
EXP_ID="83"

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
export CHECKPOINT_RESUME="/cluster/work/cvl/csakarid/results/DISS/local-exp"${EXP_ID}"/220921_1836_csHR2acdcHR_1024x1024_dacs_a999_fdthings_diss_src_cestylized_inv10_trg_pseudostylized_cestylizedstylized_invorigorigstylizedstylized10_rcs001-20_hrda1-512-01_daformer_sepaspp_sl_mitb5_poly10warm_s2_42ea2/latest.pth"
export SEED_TO_RESUME_FROM="2"

# Perform initialization operations for the experiment.
cd ${SOURCE_DIR}
source /cluster/home/csakarid/DISS/bin/activate
./experiments/scripts/initialization.sh
python tools/convert_datasets/cityscapes.py ${DIR_SOURCE_DATASET} --nproc 8

# Run the experiment.
python run_experiments.py --exp ${EXP_ID} --resume-from ${CHECKPOINT_RESUME} --seed-to-resume-from ${SEED_TO_RESUME_FROM}

# Deactivate virtual environment for DISS.
deactivate

/bin/echo Finished on: `date`
