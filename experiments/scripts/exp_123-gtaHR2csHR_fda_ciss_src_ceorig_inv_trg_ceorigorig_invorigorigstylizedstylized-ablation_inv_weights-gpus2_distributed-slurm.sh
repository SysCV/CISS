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
#SBATCH --tmp=80000
#
# Specify GPU type and number of required GPUs.
#SBATCH --gpus=rtx_3090:2
#
# Specify range of tasks for job array.
#SBATCH --array=1,12
#
# Specify file for logging standard output.
#SBATCH --output=../logs/exp_123-gtaHR2csHR_fda_ciss_src_ceorig_inv_trg_ceorigorig_invorigorigstylizedstylized-ablation_inv_weights-gpus2_distributed-slurm-01-%a.o
#
# Specify file for logging standard error.
#SBATCH --error=../logs/exp_123-gtaHR2csHR_fda_ciss_src_ceorig_inv_trg_ceorigorig_invorigorigstylizedstylized-ablation_inv_weights-gpus2_distributed-slurm-01-%a.e
#
# Specify open mode for log files.
#SBATCH --open-mode=append
#
# Specify jobname and range of tasks for job array.
#SBATCH --job-name=exp_123-gtaHR2csHR_fda_ciss_src_ceorig_inv_trg_ceorigorig_invorigorigstylizedstylized-ablation_inv_weights-gpus2_distributed-slurm

/bin/echo Starting on: `date`

# Experiment ID.
EXP_ID="123"

# Specify directories.
export TMPDIR="${TMPDIR}"
export SOURCE_DIR="/cluster/home/csakarid/code/SysCV/CISS"
export SOURCE_DATASET="gta"
export TARGET_DATASET="cityscapes"
export DIR_SOURCE_DATASET="${TMPDIR}/${SOURCE_DATASET}"
export DIR_TARGET_DATASET="${TMPDIR}/${TARGET_DATASET}"
export TAR_SOURCE_DATASET="/cluster/work/cvl/csakarid/data/GTA5/GTA5.tar.gz"
export TAR_TARGET_DATASET="/cluster/work/cvl/csakarid/data/Cityscapes/Cityscapes.tar.gz"

# Export task ID.
export SLURM_ARRAY_TASK_ID="${SLURM_ARRAY_TASK_ID}"

TASK_ORDER=(0 0 0 0 0 0 0 0 0 0 0 0 1)

# Specify checkpoint to resume from.
export CHECKPOINT_RESUME=("/cluster/work/cvl/csakarid/results/CISS/local-exp120/230127_2134_gtaHR2csHR_1024x1024_dacs_a999_fdthings_ciss_src_ceorig_inv_trg_ceorigorig_invorigorigstylizedstylized_rcs001-20_cpl2_hrda1-512-01_daformer_sepaspp_sl_mitb5_poly10warm_s0_62a61/latest.pth" "/cluster/work/cvl/csakarid/results/CISS/local-exp120/230128_0337_gtaHR2csHR_1024x1024_dacs_a999_fdthings_ciss_src_ceorig_inv_trg_ceorigorig_invorigorigstylizedstylized_rcs001-20_cpl2_hrda1-512-01_daformer_sepaspp_sl_mitb5_poly10warm_s0_3cde2/latest.pth")

# Perform initialization operations for the experiment.
cd ${SOURCE_DIR}
module load gcc/8.2.0 python_gpu/3.10.4 eth_proxy pigz
./experiments/scripts/initialization_torch_1_11.sh
source /cluster/home/csakarid/CISS_torch_1_9/bin/activate
python tools/convert_datasets/gta.py ${DIR_SOURCE_DATASET} --nproc 8
python tools/convert_datasets/cityscapes.py ${DIR_TARGET_DATASET} --nproc 8
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256

# Calculate masterport for torch.distributed based on task ID.
MASTERPORT_BASE="29515"
MASTERPORT_THIS_TASK=$((${MASTERPORT_BASE}+${SLURM_ARRAY_TASK_ID}))

# Run the experiment.
python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --node_rank=0 --master_port=${MASTERPORT_THIS_TASK} run_experiments.py --exp ${EXP_ID} --resume-from ${CHECKPOINT_RESUME[${TASK_ORDER[${SLURM_ARRAY_TASK_ID}]}]}

# Deactivate virtual environment for CISS.
deactivate

/bin/echo Finished on: `date`

