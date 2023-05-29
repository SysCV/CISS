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
#SBATCH --gpus=titan_rtx:1
#
# Specify file for logging standard output.
#SBATCH --output=../logs/exp_130-csHR2acdcHR_fda_ciss_src_ceorig_trg_ceorigorig_invorigorigstylizedstylized-ablation_inv_weights-slurm-01-25-test_ACDCtestall.o
#
# Specify file for logging standard error.
#SBATCH --error=../logs/exp_130-csHR2acdcHR_fda_ciss_src_ceorig_trg_ceorigorig_invorigorigstylizedstylized-ablation_inv_weights-slurm-01-25-test_ACDCtestall.e
#
# Specify open mode for log files.
#SBATCH --open-mode=append

/bin/echo Starting on: `date`

# Experiment ID.
EXP_ID="130"

# Specify directories.
export TMPDIR="${TMPDIR}"
export SOURCE_DIR="/cluster/home/csakarid/code/SysCV/CISS"
export TARGET_DATASET="acdc"
export DIR_TARGET_DATASET="${TMPDIR}/${TARGET_DATASET}"
export TAR_TARGET_DATASET="/cluster/work/cvl/csakarid/data/ACDC/ACDC_splits.tar.gz"
export TEST_ROOT="/cluster/work/cvl/csakarid/results/CISS/local-exp130/230222_1126_csHR2acdcHR_1024x1024_dacs_a999_fdthings_ciss_src_ceorig_trg_ceorigorig_invorigorigstylizedstylized_rcs001-20_hrda1-512-01_daformer_sepaspp_sl_mitb5_poly10warm_s2_99cde"
export CONFIG_FILE="${TEST_ROOT}/*${TEST_ROOT: -1}.json"
export CHECKPOINT_FILE="${TEST_ROOT}/latest.pth"
export SHOW_DIR="${TEST_ROOT}/ACDC_test_all/preds"
export TRAINIDS_DIR="${TEST_ROOT}/ACDC_test_all/labelTrainIds"

# Perform initialization operations for the experiment.
cd ${SOURCE_DIR}
module load gcc/8.2.0 python_gpu/3.10.4 eth_proxy pigz
./experiments/scripts/initialization_test_torch_1_11.sh
source /cluster/home/csakarid/CISS_torch_1_9/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256

# Run the experiment.
echo 'Config File:' $CONFIG_FILE
echo 'Checkpoint File:' $CHECKPOINT_FILE
echo 'Predictions Output Directory:' $SHOW_DIR
python -m tools.test ${CONFIG_FILE} ${CHECKPOINT_FILE} --data-root ${DIR_TARGET_DATASET} --test-set --format-only --eval-options imgfile_prefix=${TRAINIDS_DIR} to_label_id=False --show-dir ${SHOW_DIR} --opacity 1

# Perform finalization operations.
deactivate

/bin/echo Finished on: `date`

