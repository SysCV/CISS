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
# Specify disk limit on local scratch.
#BSUB -R "rusage[scratch=40000]"
#
# Specify number of required GPUs.
#BSUB -R "rusage[ngpus_excl_p=2]"
#
# Specify GPU type and number of required GPUs.
#BSUB -R "select[gpu_mtotal0>=20000]"
#
# Specify file for logging standard output.
#BSUB -o ../logs/exp_97-csHR2acdcHR_fda_ciss_src_cestylized_inv-ablation_inv_weights-gpus2_distributed-07-%I.o
#
# Specify file for logging standard error.
#BSUB -e ../logs/exp_97-csHR2acdcHR_fda_ciss_src_cestylized_inv-ablation_inv_weights-gpus2_distributed-07-%I.e
#
# Specify jobname and range of tasks for job array.
#BSUB -J exp_97-csHR2acdcHR_fda_ciss_src_cestylized_inv-ablation_inv_weights-gpus2_distributed[20-26]

/bin/echo Starting on: `date`

# Experiment ID.
EXP_ID="97"

# Specify directories.
export TMPDIR="${TMPDIR}"
export SOURCE_DIR="/cluster/home/csakarid/code/SysCV/CISS"
export SOURCE_DATASET="cityscapes"
export TARGET_DATASET="acdc"
export DIR_SOURCE_DATASET="${TMPDIR}/${SOURCE_DATASET}"
export DIR_TARGET_DATASET="${TMPDIR}/${TARGET_DATASET}"
export TAR_SOURCE_DATASET="/cluster/work/cvl/csakarid/data/Cityscapes/Cityscapes.tar.gz"
export TAR_TARGET_DATASET="/cluster/work/cvl/csakarid/data/ACDC/ACDC_splits.tar.gz"

# Export task ID.
export LSB_JOBINDEX="${LSB_JOBINDEX}"

# Specify checkpoint to resume from.
export CHECKPOINT_RESUME=("/cluster/work/cvl/csakarid/results/CISS/local-exp"${EXP_ID}"/221012_1623_csHR2acdcHR_1024x1024_dacs_a999_fdthings_ciss_src_cestylized_inv_rcs001-20_hrda1-512-01_daformer_sepaspp_sl_mitb5_poly10warm_s0_18157/latest.pth" "/cluster/work/cvl/csakarid/results/CISS/local-exp"${EXP_ID}"/221012_1623_csHR2acdcHR_1024x1024_dacs_a999_fdthings_ciss_src_cestylized_inv_rcs001-20_hrda1-512-01_daformer_sepaspp_sl_mitb5_poly10warm_s0_60a00/latest.pth" "/cluster/work/cvl/csakarid/results/CISS/local-exp"${EXP_ID}"/221012_1630_csHR2acdcHR_1024x1024_dacs_a999_fdthings_ciss_src_cestylized_inv_rcs001-20_hrda1-512-01_daformer_sepaspp_sl_mitb5_poly10warm_s0_c4f0e/latest.pth" "/cluster/work/cvl/csakarid/results/CISS/local-exp"${EXP_ID}"/221012_1630_csHR2acdcHR_1024x1024_dacs_a999_fdthings_ciss_src_cestylized_inv_rcs001-20_hrda1-512-01_daformer_sepaspp_sl_mitb5_poly10warm_s0_48c5b/latest.pth" "/cluster/work/cvl/csakarid/results/CISS/local-exp"${EXP_ID}"/221012_1630_csHR2acdcHR_1024x1024_dacs_a999_fdthings_ciss_src_cestylized_inv_rcs001-20_hrda1-512-01_daformer_sepaspp_sl_mitb5_poly10warm_s0_f581e/latest.pth" "/cluster/work/cvl/csakarid/results/CISS/local-exp"${EXP_ID}"/221012_1630_csHR2acdcHR_1024x1024_dacs_a999_fdthings_ciss_src_cestylized_inv_rcs001-20_hrda1-512-01_daformer_sepaspp_sl_mitb5_poly10warm_s0_3906e/latest.pth" "/cluster/work/cvl/csakarid/results/CISS/local-exp"${EXP_ID}"/221012_1710_csHR2acdcHR_1024x1024_dacs_a999_fdthings_ciss_src_cestylized_inv_rcs001-20_hrda1-512-01_daformer_sepaspp_sl_mitb5_poly10warm_s0_3f1c3/latest.pth")

# Perform initialization operations for the experiment.
cd ${SOURCE_DIR}
module load gcc/8.2.0 python_gpu/3.10.4 eth_proxy pigz
./experiments/scripts/initialization_torch_1_11.sh
source /cluster/home/csakarid/CISS_torch_1_9/bin/activate
python tools/convert_datasets/cityscapes.py ${DIR_SOURCE_DATASET} --nproc 8
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256

# Calculate masterport for torch.distributed based on task ID.
MASTERPORT_BASE="29550"
MASTERPORT_THIS_TASK=$((${MASTERPORT_BASE}+${LSB_JOBINDEX}))

# Run the experiment.
python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --node_rank=0 --master_port=${MASTERPORT_THIS_TASK} run_experiments.py --exp ${EXP_ID} --resume-from ${CHECKPOINT_RESUME[((${LSB_JOBINDEX}-20))]}

# Deactivate virtual environment for CISS.
deactivate

/bin/echo Finished on: `date`

