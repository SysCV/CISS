#!/bin/bash
#
# Specify time limit.
#BSUB -W 47:59
#
# Specify number of CPU cores.
#BSUB -n 8
#
# Specify memory limit per CPU core.
#BSUB -R "rusage[mem=8192]"
#
# Specify disk limit on local scratch per CPU core.
#BSUB -R "rusage[scratch=40000]"
#
# Specify number of required GPUs.
#BSUB -R "rusage[ngpus_excl_p=1]"
#
# Specify GPU type and number of required GPUs.
#BSUB -R "select[gpu_mtotal0>=30000]"
#
# Specify file for logging standard output.
#BSUB -o ../logs/exp_99-csHR2darkzurichHR_hrda-08.o
#
# Specify file for logging standard error.
#BSUB -e ../logs/exp_99-csHR2darkzurichHR_hrda-08.e

/bin/echo Starting on: `date`

# Experiment ID.
EXP_ID="99"

# Specify directories.
export TMPDIR="${TMPDIR}"
export SOURCE_DIR="/cluster/home/csakarid/code/SysCV/CISS"
export SOURCE_DATASET="cityscapes"
export TARGET_DATASET="darkzurich"
export DIR_SOURCE_DATASET="${TMPDIR}/${SOURCE_DATASET}"
export DIR_TARGET_DATASET="${TMPDIR}/${TARGET_DATASET}"
export TAR_SOURCE_DATASET="/cluster/work/cvl/csakarid/data/Cityscapes/Cityscapes.tar.gz"
export TAR_TARGET_DATASET="/cluster/work/cvl/csakarid/data/Dark_Zurich/Dark_Zurich.tar.gz"

# Perform initialization operations for the experiment.
cd ${SOURCE_DIR}
module load gcc/8.2.0 python_gpu/3.10.4 eth_proxy pigz
./experiments/scripts/initialization_torch_1_11.sh
source /cluster/home/csakarid/CISS_torch_1_9/bin/activate
python tools/convert_datasets/cityscapes.py ${DIR_SOURCE_DATASET} --nproc 8

# Run the experiment.
python run_experiments.py --exp ${EXP_ID}

# Deactivate virtual environment for CISS.
deactivate

/bin/echo Finished on: `date`

