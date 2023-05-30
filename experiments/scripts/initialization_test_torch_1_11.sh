# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
#!/bin/bash

# Copy datasets to local scratch of compute node.
/bin/echo Starting dataset copying on: `date`

# Copy target dataset.
mkdir ${DIR_TARGET_DATASET}
tar -I pigz -xf ${TAR_TARGET_DATASET} -C ${DIR_TARGET_DATASET}/

/bin/echo Finished dataset copying on: `date`
