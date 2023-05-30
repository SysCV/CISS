# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
#!/bin/bash

# Load modules.
# module load gcc/8.2.0 python_gpu/3.10.4 eth_proxy pigz

# Activate virtual environment for CISS.
# source /cluster/home/csakarid/CISS/bin/activate

# Copy datasets to local scratch of compute node.
/bin/echo Starting dataset copying on: `date`

# Copy source dataset.
mkdir ${DIR_SOURCE_DATASET}
tar -I pigz -xf ${TAR_SOURCE_DATASET} -C ${DIR_SOURCE_DATASET}/

# Copy target dataset.
mkdir ${DIR_TARGET_DATASET}
tar -I pigz -xf ${TAR_TARGET_DATASET} -C ${DIR_TARGET_DATASET}/

/bin/echo Finished dataset copying on: `date`

# Create symlinks in the data directory of the repository to the data in the compute
# node scratch.
# ln -s ${DIR_SOURCE_DATASET} ${SOURCE_DIR}/data
# ln -s ${DIR_TARGET_DATASET} ${SOURCE_DIR}/data

# Create temporary directory for storing the results of the experiment.
# mkdir ${3}

# Copy the pre-trained model to local scratch of compute node.
# mkdir -p ${TMPDIR}/${4}
# rsync -aq ${5} ${TMPDIR}/${4}
