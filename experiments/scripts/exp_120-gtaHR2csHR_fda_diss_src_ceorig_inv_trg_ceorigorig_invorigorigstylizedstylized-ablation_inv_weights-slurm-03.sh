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
#SBATCH --gpus=a100_80gb_pcie:1
#
# Specify range of tasks for job array.
#SBATCH --array=0
#
# Specify file for logging standard output.
#SBATCH --output=../logs/exp_120-gtaHR2csHR_fda_diss_src_ceorig_inv_trg_ceorigorig_invorigorigstylizedstylized-ablation_inv_weights-slurm-03-02-%a.o
#
# Specify file for logging standard error.
#SBATCH --error=../logs/exp_120-gtaHR2csHR_fda_diss_src_ceorig_inv_trg_ceorigorig_invorigorigstylizedstylized-ablation_inv_weights-slurm-03-02-%a.e
#
# Specify open mode for log files.
#SBATCH --open-mode=append
#
# Specify jobname and range of tasks for job array.
#SBATCH --job-name=exp_120-gtaHR2csHR_fda_diss_src_ceorig_inv_trg_ceorigorig_invorigorigstylizedstylized-ablation_inv_weights-slurm-03

/bin/echo Starting on: `date`

# Experiment ID.
EXP_ID="120"

# Specify directories.
export TMPDIR="${TMPDIR}"
export SOURCE_DIR="/cluster/home/csakarid/code/SysCV/DISS"
export SOURCE_DATASET="gta"
export TARGET_DATASET="cityscapes"
export DIR_SOURCE_DATASET="${TMPDIR}/${SOURCE_DATASET}"
export DIR_TARGET_DATASET="${TMPDIR}/${TARGET_DATASET}"
export TAR_SOURCE_DATASET="/cluster/work/cvl/csakarid/data/GTA5/GTA5.tar.gz"
export TAR_TARGET_DATASET="/cluster/work/cvl/csakarid/data/Cityscapes/Cityscapes.tar.gz"

# Export task ID.
export SLURM_ARRAY_TASK_ID="${SLURM_ARRAY_TASK_ID}"

# Perform initialization operations for the experiment.
cd ${SOURCE_DIR}
source /cluster/home/csakarid/DISS/bin/activate
./experiments/scripts/initialization.sh
python tools/convert_datasets/gta.py ${DIR_SOURCE_DATASET} --nproc 8
python tools/convert_datasets/cityscapes.py ${DIR_TARGET_DATASET} --nproc 8

# Run the experiment.
python run_experiments.py --exp ${EXP_ID}

/bin/echo Finished on: `date`

