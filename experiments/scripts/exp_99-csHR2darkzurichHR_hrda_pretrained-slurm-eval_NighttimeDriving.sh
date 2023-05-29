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
# Specify required GPU memory.
# SBATCH --gres=gpumem:50g
#
# Specify file for logging standard output.
#SBATCH --output=../logs/exp_99-csHR2darkzurichHR_hrda_pretrained-slurm-eval_NighttimeDriving-04.o
#
# Specify file for logging standard error.
#SBATCH --error=../logs/exp_99-csHR2darkzurichHR_hrda_pretrained-slurm-eval_NighttimeDriving-04.e
#
# Specify open mode for log files.
#SBATCH --open-mode=append

/bin/echo Starting on: `date`

# Experiment ID.
EXP_ID="99"

# Specify directories.
export TMPDIR="${TMPDIR}"
export SOURCE_DIR="/cluster/home/csakarid/code/SysCV/DISS"
export TARGET_DATASET="NighttimeDrivingTest"
export DIR_TARGET_DATASET="${TMPDIR}/${TARGET_DATASET}"
export TAR_TARGET_DATASET="/cluster/work/cvl/csakarid/data/Nighttime_Driving/Nighttime_Driving.tar.gz"
export TEST_ROOT="/cluster/work/cvl/csakarid/results/DISS/local-exp99/221024_0439_csHR2dzurHR_1024x1024_dacs_a999_fdthings_rcs001-20_hrda1-512-01_daformer_sepaspp_sl_mitb5_poly10warm_s0_76827"
export CONFIG_FILE="/cluster/work/cvl/csakarid/results/DISS/local-exp99/221024_0439_csHR2dzurHR_1024x1024_dacs_a999_fdthings_rcs001-20_hrda1-512-01_daformer_sepaspp_sl_mitb5_poly10warm_s0_76827/230306_1111_csHR2dzurHR_1024x1024_dacs_a999_fdthings_diss_src_ceorig_inv_trg_ceorigorig_invorigorigstylizedstylized_rcs001-20_hrda1-512-01_daformer_sepaspp_sl_mitb5_poly10warm_s1_56306_test_nighttimedriving.json"
export CHECKPOINT_FILE="${TEST_ROOT}/csHR2dzurHR_hrda_97e26/iter_40000_relevant.pth"

# Perform initialization operations for the experiment.
cd ${SOURCE_DIR}
module load gcc/8.2.0 python_gpu/3.10.4 eth_proxy pigz
./experiments/scripts/initialization_test_torch_1_11.sh
source /cluster/home/csakarid/DISS_torch_1_9/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256

# Run the experiment.
echo 'Config File:' $CONFIG_FILE
echo 'Checkpoint File:' $CHECKPOINT_FILE
python -m tools.test ${CONFIG_FILE} ${CHECKPOINT_FILE} --data-root ${DIR_TARGET_DATASET} --eval mIoU

# Perform finalization operations.
deactivate

/bin/echo Finished on: `date`

