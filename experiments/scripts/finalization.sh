#!/bin/bash

# Remove symlinks in the data directory of the repository to the data in the compute
# node scratch.
rm ${SOURCE_DIR}/data/${SOURCE_DATASET}
rm ${SOURCE_DIR}/data/${TARGET_DATASET}

# Create directory in work for permanently storing the results of the experiment.
# mkdir -p ${2}

# Copy results from local scratch of compute node to work.
# /bin/echo Starting result copying on: `date`

# Zip results first. The pwd is identical to the directory that is zipped.
# zip -r ${1}/results.zip ./

# Copy only the zip to work.
# rsync -auq ${1}/results.zip ${2}

# /bin/echo Finished result copying on: `date`

