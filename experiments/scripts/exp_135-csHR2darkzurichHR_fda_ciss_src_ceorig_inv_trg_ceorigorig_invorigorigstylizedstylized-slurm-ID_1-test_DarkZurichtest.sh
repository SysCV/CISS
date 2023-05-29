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
#SBATCH --output=../logs/exp_135-csHR2darkzurichHR_fda_ciss_src_ceorig_inv_trg_ceorigorig_invorigorigstylizedstylized-slurm-02-1-test_DarkZurichtest.o
#
# Specify file for logging standard error.
#SBATCH --error=../logs/exp_135-csHR2darkzurichHR_fda_ciss_src_ceorig_inv_trg_ceorigorig_invorigorigstylizedstylized-slurm-02-1-test_DarkZurichtest.e
#
# Specify open mode for log files.
#SBATCH --open-mode=append

/bin/echo Starting on: `date`

# Specify directories.
export TMPDIR="${TMPDIR}"
export SOURCE_DIR="/cluster/home/csakarid/code/SysCV/CISS"
export TARGET_DATASET="darkzurich"
export DIR_TARGET_DATASET="${TMPDIR}/${TARGET_DATASET}"
export TAR_TARGET_DATASET="/cluster/work/cvl/csakarid/data/Dark_Zurich/Dark_Zurich.tar.gz"
export TEST_ROOT="/cluster/work/cvl/csakarid/results/CISS/local-exp135/230306_1111_csHR2dzurHR_1024x1024_dacs_a999_fdthings_ciss_src_ceorig_inv_trg_ceorigorig_invorigorigstylizedstylized_rcs001-20_hrda1-512-01_daformer_sepaspp_sl_mitb5_poly10warm_s1_56306"
export CONFIG_FILE="${TEST_ROOT}/*${TEST_ROOT: -1}.json"
export CHECKPOINT_FILE="${TEST_ROOT}/iter_32000_saved.pth"
export SHOW_DIR="${TEST_ROOT}/Dark_Zurich_test/preds"
export TRAINIDS_DIR="${TEST_ROOT}/Dark_Zurich_test/labelTrainIds"

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

# Deactivate virtual environment for CISS.
deactivate

/bin/echo Finished on: `date`

