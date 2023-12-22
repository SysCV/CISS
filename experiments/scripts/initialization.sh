#!/bin/bash
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

# Load modules.
module load gcc/6.3.0 python eth_proxy pigz

# Copy datasets to local scratch of compute node.
/bin/echo Starting dataset copying on: `date`

# Copy source dataset.
mkdir ${DIR_SOURCE_DATASET}
tar -I pigz -xf ${TAR_SOURCE_DATASET} -C ${DIR_SOURCE_DATASET}/

# Sym-link source dataset.
ln -s ${DIR_SOURCE_DATASET}/ data/

# Copy target dataset.
mkdir ${DIR_TARGET_DATASET}
tar -I pigz -xf ${TAR_TARGET_DATASET} -C ${DIR_TARGET_DATASET}/

# Sym-link target dataset.
ln -s ${DIR_TARGET_DATASET}/ data/

/bin/echo Finished dataset copying on: `date`
