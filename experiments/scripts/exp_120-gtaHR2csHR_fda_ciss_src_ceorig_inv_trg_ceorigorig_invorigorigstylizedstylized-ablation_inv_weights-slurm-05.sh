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
# Specify number of required GPUs.
#SBATCH --gpus=1
#
# Specify required GPU memory.
#SBATCH --gres=gpumem:30g
#
# Specify range of tasks for job array.
#SBATCH --array=4
#
# Specify file for logging standard output.
#SBATCH --output=../logs/exp_120-gtaHR2csHR_fda_ciss_src_ceorig_inv_trg_ceorigorig_invorigorigstylizedstylized-ablation_inv_weights-slurm-05-01-%a.o
#
# Specify file for logging standard error.
#SBATCH --error=../logs/exp_120-gtaHR2csHR_fda_ciss_src_ceorig_inv_trg_ceorigorig_invorigorigstylizedstylized-ablation_inv_weights-slurm-05-01-%a.e
#
# Specify open mode for log files.
#SBATCH --open-mode=append
#
# Specify jobname and range of tasks for job array.
#SBATCH --job-name=exp_120-gtaHR2csHR_fda_ciss_src_ceorig_inv_trg_ceorigorig_invorigorigstylizedstylized-ablation_inv_weights-slurm-05

/bin/echo Starting on: `date`

# Experiment ID.
EXP_ID="120"

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

# Specify checkpoint to resume from.
export CHECKPOINT_RESUME="/cluster/work/cvl/csakarid/results/CISS/local-exp120/230127_2134_gtaHR2csHR_1024x1024_dacs_a999_fdthings_ciss_src_ceorig_inv_trg_ceorigorig_invorigorigstylizedstylized_rcs001-20_cpl2_hrda1-512-01_daformer_sepaspp_sl_mitb5_poly10warm_s0_a2e5a/latest.pth"

# Perform initialization operations for the experiment.
cd ${SOURCE_DIR}
source /cluster/home/csakarid/CISS/bin/activate
./experiments/scripts/initialization.sh
python tools/convert_datasets/gta.py ${DIR_SOURCE_DATASET} --nproc 8
python tools/convert_datasets/cityscapes.py ${DIR_TARGET_DATASET} --nproc 8

# Run the experiment.
python run_experiments.py --exp ${EXP_ID} --resume-from ${CHECKPOINT_RESUME}

/bin/echo Finished on: `date`

