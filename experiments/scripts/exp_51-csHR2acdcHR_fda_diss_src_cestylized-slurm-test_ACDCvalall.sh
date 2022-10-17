#!/bin/bash
#
# Specify time limit.
#SBATCH --time=3:59:59
#
# Specify number of CPU cores.
#SBATCH -n 8
#
# Specify memory limit per CPU core.
#SBATCH --mem-per-cpu=8192
#
# Specify disk limit on local scratch.
#SBATCH --tmp=80000
#
# Specify GPU type and number of required GPUs.
#SBATCH --gpus=rtx_3090:1
#
# Specify file for logging standard output.
#SBATCH --output=../logs/exp_51-csHR2acdcHR_fda_diss_src_cestylized-slurm-test_ACDCvalall.o
#
# Specify file for logging standard error.
#SBATCH --error=../logs/exp_51-csHR2acdcHR_fda_diss_src_cestylized-slurm-test_ACDCvalall.e
#
# Specify open mode for log files.
#SBATCH --open-mode=append
#
# Specify jobname.
#SBATCH --job-name=exp_51-csHR2acdcHR_fda_diss_src_cestylized-slurm-test_ACDCvalall

/bin/echo Starting on: `date`

# Experiment ID.
EXP_ID="51"

# Specify directories.
export TMPDIR="${TMPDIR}"
export SOURCE_DIR="/cluster/home/csakarid/code/SysCV/DISS"
export TARGET_DATASET="acdc"
export DIR_TARGET_DATASET="${TMPDIR}/${TARGET_DATASET}"
export TAR_TARGET_DATASET="/cluster/work/cvl/csakarid/data/ACDC/ACDC_splits.tar.gz"
export TEST_ROOT="/cluster/work/cvl/csakarid/results/DISS/local-exp51/220911_0247_csHR2acdcHR_1024x1024_dacs_a999_fdthings_diss_src_cestylized_rcs001-20_hrda1-512-01_daformer_sepaspp_sl_mitb5_poly10warm_s1_813de"
export CONFIG_FILE="${TEST_ROOT}/*${TEST_ROOT: -1}.json"
export CHECKPOINT_FILE="${TEST_ROOT}/latest.pth"
export SHOW_DIR="${TEST_ROOT}/ACDC_val_all/preds"
export TRAINIDS_DIR="${TEST_ROOT}/ACDC_val_all/labelTrainIds"

# Perform initialization operations for the experiment.
cd ${SOURCE_DIR}
module load gcc/8.2.0 python_gpu/3.10.4 eth_proxy pigz
./experiments/scripts/initialization_test_torch_1_11.sh
source /cluster/home/csakarid/DISS_torch_1_9/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256

# Run the experiment.
echo 'Config File:' $CONFIG_FILE
echo 'Checkpoint File:' $CHECKPOINT_FILE
echo 'Predictions Output Directory:' $SHOW_DIR
python -m tools.test ${CONFIG_FILE} ${CHECKPOINT_FILE} --data-root ${DIR_TARGET_DATASET} --format-only --eval-options imgfile_prefix=${TRAINIDS_DIR} to_label_id=False --show-dir ${SHOW_DIR} --opacity 1

# Perform finalization operations.
deactivate

/bin/echo Finished on: `date`

