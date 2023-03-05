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
#SBATCH --array=17-33
#
# Specify file for logging standard output.
#SBATCH --output=../logs/exp_111-csHR2darkzurichHR_fda_diss_src_ceorig_inv_trg_ceorigorig_invorigorigstylizedstylized-ablation_inv_weights-gpus2_distributed-slurm-01-%a.o
#
# Specify file for logging standard error.
#SBATCH --error=../logs/exp_111-csHR2darkzurichHR_fda_diss_src_ceorig_inv_trg_ceorigorig_invorigorigstylizedstylized-ablation_inv_weights-gpus2_distributed-slurm-01-%a.e
#
# Specify open mode for log files.
#SBATCH --open-mode=append
#
# Specify jobname and range of tasks for job array.
#SBATCH --job-name=exp_111-csHR2darkzurichHR_fda_diss_src_ceorig_inv_trg_ceorigorig_invorigorigstylizedstylized-ablation_inv_weights-gpus2_distributed-slurm

/bin/echo Starting on: `date`

# Experiment ID.
EXP_ID="111"

# Specify directories.
export TMPDIR="${TMPDIR}"
export SOURCE_DIR="/cluster/home/csakarid/code/SysCV/DISS"
export SOURCE_DATASET="cityscapes"
export TARGET_DATASET="darkzurich"
export DIR_SOURCE_DATASET="${TMPDIR}/${SOURCE_DATASET}"
export DIR_TARGET_DATASET="${TMPDIR}/${TARGET_DATASET}"
export TAR_SOURCE_DATASET="/cluster/work/cvl/csakarid/data/Cityscapes/Cityscapes.tar.gz"
export TAR_TARGET_DATASET="/cluster/work/cvl/csakarid/data/Dark_Zurich/Dark_Zurich.tar.gz"

# Export task ID.
export SLURM_ARRAY_TASK_ID="${SLURM_ARRAY_TASK_ID}"

# Perform initialization operations for the experiment.
cd ${SOURCE_DIR}
module load gcc/8.2.0 python_gpu/3.10.4 eth_proxy pigz
./experiments/scripts/initialization_torch_1_11.sh
source /cluster/home/csakarid/DISS_torch_1_9/bin/activate
python tools/convert_datasets/cityscapes.py ${DIR_SOURCE_DATASET} --nproc 8
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256

# Calculate masterport for torch.distributed based on task ID.
MASTERPORT_BASE="29534"
MASTERPORT_THIS_TASK=$((${MASTERPORT_BASE}+${SLURM_ARRAY_TASK_ID}))

# Run the experiment.
python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --node_rank=0 --master_port=${MASTERPORT_THIS_TASK} run_experiments.py --exp ${EXP_ID}

# Deactivate virtual environment for DISS.
deactivate

/bin/echo Finished on: `date`

